use crate::store::{ImageData, Store, StoreEvent};
use annot_core::model::*;
use annot_core::proto::*;
use eframe::egui::{
    self, Align2, Color32, CursorIcon, FontId, Key, Modifiers, PointerButton, Pos2, Rect, Sense,
    Stroke, Vec2,
};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};

const HIT_TOL: f32 = 8.0; // screen pixels
const CLOSE_TOL: f32 = 10.0; // screen pixels

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
    Select,
    /// add / remove / move the points of the selected polygon or polyline
    Points,
    BBox,
    Polygon,
    Polyline,
    /// SAM: click an object, get a polygon
    Magic,
}

/// Prompt of the Magic (SAM) tool for the current image.
#[derive(Default)]
struct Magic {
    /// (image point, positive?)
    points: Vec<([f32; 2], bool)>,
    bbox: Option<[f32; 4]>,
    /// id of the newest request sent; older answers are ignored
    seq: u64,
    /// prompt changed but not sent yet (image still being prepared)
    unsent: bool,
    /// SAM's answer: polygon, or [min, max] corners when a box was requested
    shape: Option<Vec<[f32; 2]>>,
    shape_is_box: bool,
}

impl Magic {
    fn is_empty(&self) -> bool {
        self.points.is_empty() && self.bbox.is_none()
    }
}

/// "Auto-label all unlabeled" running in the AI worker (desktop) or on the lab server.
struct BatchJob {
    id: u64,
    done: usize,
    total: usize,
    added: usize,
}

#[derive(Clone, Copy)]
enum Drag {
    None,
    NewBox { start: [f32; 2], cur: [f32; 2] },
    Vertex { shape: usize, idx: usize },
    Move { shape: usize, last: [f32; 2] },
    Pan,
}

/// Select tool: what a point edit (add / remove) under the mouse would act on.
/// Only the selected polygon / polyline is considered.
#[derive(Clone, Copy)]
enum PointHover {
    /// existing vertex: right-click / Backspace removes it
    Vertex { shape: usize, idx: usize },
    /// on an edge: right-click / double-click / I inserts a vertex at `at` after `edge`
    Edge { shape: usize, edge: usize, at: [f32; 2] },
}

struct Loaded {
    name: String,
    tex: egui::TextureHandle,
    w: u32,
    h: u32,
}

struct ExportDlg {
    open: bool,
    out: String,
    val_ratio: f32,
    include_unlabeled: bool,
    keep_split: bool,
    /// keep each folder (video / sequence) entirely in train or in val
    by_folder: bool,
}

/// Folder tree of the dataset, for the navigation pane.
#[derive(Default)]
struct DirNode {
    id: usize, // pre-order index (root = 0), indexes the per-frame counts
    name: String,
    path: String,
    dirs: Vec<DirNode>,
    files: Vec<usize>,
}

/// One visible row of the (virtualized) folder tree.
enum Row<'a> {
    Dir(&'a DirNode, usize),
    File(usize, usize),
}

impl DirNode {
    fn build(files: &[String]) -> DirNode {
        let mut root = DirNode::default();
        for (i, f) in files.iter().enumerate() {
            let parts: Vec<&str> = f.split('/').collect();
            let mut node = &mut root;
            for k in 0..parts.len() - 1 {
                let idx = match node.dirs.iter().position(|n| n.name == parts[k]) {
                    Some(p) => p,
                    None => {
                        node.dirs.push(DirNode {
                            name: parts[k].to_string(),
                            path: parts[..=k].join("/"),
                            ..Default::default()
                        });
                        node.dirs.len() - 1
                    }
                };
                node = &mut node.dirs[idx];
            }
            node.files.push(i);
        }
        let mut next = 0;
        root.number(&mut next);
        root
    }

    fn number(&mut self, next: &mut usize) {
        self.id = *next;
        *next += 1;
        for d in &mut self.dirs {
            d.number(next);
        }
    }

    /// Fill `out[id] = (labeled, total)` for this folder and all below it.
    fn counts(&self, labeled: &[bool], out: &mut Vec<(usize, usize)>) -> (usize, usize) {
        let mut c = (self.files.iter().filter(|i| labeled[**i]).count(), self.files.len());
        for d in &self.dirs {
            let x = d.counts(labeled, out);
            c.0 += x.0;
            c.1 += x.1;
        }
        if out.len() <= self.id {
            out.resize(self.id + 1, (0, 0));
        }
        out[self.id] = c;
        c
    }

    /// Visible rows: folders, and the contents of expanded folders.
    fn flatten<'a>(&'a self, depth: usize, expanded: &HashSet<String>, out: &mut Vec<Row<'a>>) {
        for d in &self.dirs {
            out.push(Row::Dir(d, depth));
            if expanded.contains(&d.path) {
                d.flatten(depth + 1, expanded, out);
            }
        }
        for &i in &self.files {
            out.push(Row::File(i, depth));
        }
    }

    fn all_paths(&self, out: &mut HashSet<String>) {
        for d in &self.dirs {
            out.insert(d.path.clone());
            d.all_paths(out);
        }
    }
}

/// Folder paths containing image `rel` ("a/b/c.jpg" -> "a", "a/b").
fn ancestors(rel: &str) -> impl Iterator<Item = &str> {
    rel.match_indices('/').map(move |(i, _)| &rel[..i])
}

pub struct App {
    store: Box<dyn Store>,
    /// the open dataset (desktop: folder path; lab: as reported by the server)
    source: Option<String>,
    files: Vec<String>,
    cur: usize,
    loaded: Option<Loaded>,
    project: Project,
    tool: Tool,
    class: usize,
    new_class: String,
    selected: Option<usize>,
    drawing: Vec<[f32; 2]>,
    drag: Drag,
    zoom: f32,
    offset: Vec2,
    fit_pending: bool,
    /// snapshots of the current image's shapes; None = the image had no annotation entry
    undo: Vec<Option<Vec<Annot>>>,
    /// images whose label file must be rewritten at the next save (e.g. class deleted)
    stale: HashSet<String>,
    dirty: bool,
    labels_stale: bool,
    last_save: f64,
    /// Select tool: vertex / edge of the selected shape under the mouse (updated every frame)
    point_hover: Option<PointHover>,
    /// (shape, point) clicked last: Del removes this point instead of the whole shape.
    /// Only valid while that shape is the selected one (see `selected_point`).
    sel_point: Option<(usize, usize)>,
    /// last failure of the periodic / forced save (None after a clean save)
    save_error: Option<String>,
    /// last failure writing the current image's label file
    label_error: Option<String>,
    status: String,
    export: ExportDlg,
    cursor_img: Option<[f32; 2]>,
    scroll_list: bool,
    show_labels: bool,
    tree: DirNode,
    filter: String,
    unlabeled_only: bool,
    /// current image's label file must be (re)written
    label_dirty: bool,
    expanded: HashSet<String>,
    nav_dir: i64,
    last_draw_tool: Tool,
    about_open: bool,
    /// a text field had keyboard focus last frame (shortcuts are off while typing)
    typing: bool,
    /// polygon triangulations, keyed by a hash of the points (zoom/pan independent)
    tri_cache: RefCell<HashMap<u64, Vec<u32>>>,
    /// running export: (done, total)
    export_progress: Option<(usize, usize)>,
    // ---- AI assist
    ai_started: bool,
    ai_settings: AiSettings,
    /// engine / model in use, or why AI is unavailable
    ai_status: String,
    ai_models: Vec<String>,
    /// lab: YOLO models in the server's models folder
    yolo_models: Vec<String>,
    /// AI keys of images whose SAM embedding is ready in the worker
    encoded: HashSet<String>,
    magic: Magic,
    batch: Option<BatchJob>,
    last_encode_ms: u128,
    /// id of the newest auto-label batch (results of older ones are ignored)
    batch_seq: u64,
    /// (AI key, reason) when SAM could not prepare that image
    encode_error: Option<(String, String)>,
    /// single-image YOLO result waiting for its new classes (lab: added by the server first)
    pending_dets: Option<(String, Vec<Detection>, u32, u32)>,
    /// image requested from the store but not arrived yet: (index, image left)
    pending_load: Option<(usize, Option<String>)>,
    // ---- lab server
    /// user name (lab); asked once, remembered by the browser
    user: String,
    user_edit: String,
    admin: bool,
    admin_path: String,
    connected: bool,
    locks: BTreeMap<String, String>,
    users: Vec<String>,
    /// labels sent to the server and not confirmed yet
    unacked: HashSet<String>,
    /// lab: text being typed into the Rename field (sent on Enter / focus loss)
    rename_edit: Option<String>,
    /// lab, on the server machine: the "switch dataset folder" window is open
    admin_dialog: bool,
    ctx: egui::Context,
}

fn class_color(i: usize) -> Color32 {
    let h = (i as f32 * 0.618_034 + 0.05) % 1.0;
    egui::ecolor::Hsva::new(h, 0.8, 1.0, 1.0).into()
}

impl App {
    /// `open`: desktop: a folder or image to open right away.
    pub fn new(cc: &eframe::CreationContext<'_>, store: Box<dyn Store>, open: Option<String>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let user: String = cc.storage.and_then(|s| eframe::get_value(s, "user")).unwrap_or_default();
        let lab = store.is_lab();
        let mut app = App {
            store,
            source: None,
            files: vec![],
            cur: 0,
            loaded: None,
            project: Project::default(),
            tool: Tool::BBox,
            class: 0,
            new_class: String::new(),
            selected: None,
            drawing: vec![],
            drag: Drag::None,
            zoom: 1.0,
            offset: Vec2::ZERO,
            fit_pending: true,
            undo: vec![],
            dirty: false,
            labels_stale: false,
            last_save: 0.0,
            point_hover: None,
            sel_point: None,
            save_error: None,
            label_error: None,
            status: if lab { "Connecting to the lab server...".into() } else { "Open an image folder to start.".into() },
            export: ExportDlg {
                open: false,
                out: String::new(),
                val_ratio: 0.2,
                include_unlabeled: false,
                keep_split: true,
                by_folder: false,
            },
            cursor_img: None,
            scroll_list: false,
            show_labels: true,
            tree: DirNode::default(),
            filter: String::new(),
            unlabeled_only: false,
            label_dirty: false,
            expanded: HashSet::new(),
            nav_dir: 1,
            last_draw_tool: Tool::BBox,
            about_open: false,
            typing: false,
            tri_cache: RefCell::new(HashMap::new()),
            stale: HashSet::new(),
            export_progress: None,
            ai_started: false,
            ai_settings: if lab { AiSettings::default() } else { AiSettings::load() },
            ai_status: "not started (select the Magic tool or Auto-label)".into(),
            ai_models: vec![],
            yolo_models: vec![],
            encoded: HashSet::new(),
            magic: Magic::default(),
            batch: None,
            last_encode_ms: 0,
            batch_seq: 0,
            encode_error: None,
            pending_dets: None,
            pending_load: None,
            user_edit: user.clone(),
            user,
            admin: false,
            admin_path: String::new(),
            connected: false,
            locks: BTreeMap::new(),
            users: vec![],
            unacked: HashSet::new(),
            rename_edit: None,
            admin_dialog: false,
            ctx: cc.egui_ctx.clone(),
        };
        if lab && !app.user.is_empty() {
            let u = app.user.clone();
            app.store.hello(&u);
        }
        if let Some(p) = open {
            app.open_path(&cc.egui_ctx, &p);
        }
        app
    }

    fn lab(&self) -> bool {
        self.store.is_lab()
    }

    /// Lab: ask the server to change the label format / polyline width for everyone.
    fn send_settings(&mut self, mode: Option<ExportMode>, width: Option<f32>, frames: Option<NewFrames>) {
        let p = &self.project;
        self.store.op(ClassOp::Settings {
            export_mode: mode.unwrap_or(p.export_mode),
            polyline_width: width.unwrap_or(p.polyline_width),
            new_frames: frames.unwrap_or(p.new_frames),
        });
    }

    /// Lab: someone else is editing the current image, so it is shown read-only.
    fn locked_by_other(&self) -> Option<&str> {
        let l = self.loaded.as_ref()?;
        self.locks.get(&l.name).filter(|u| **u != self.user).map(|s| s.as_str())
    }

    // ------------------------------------------------------------ files

    /// Desktop: open a folder, or jump to an image (opening its folder if needed).
    fn open_path(&mut self, ctx: &egui::Context, path: &str) {
        let p = path.replace('\\', "/");
        // an image inside the open dataset: just jump to it
        if let Some(src) = self.source.as_ref().map(|s| s.replace('\\', "/")) {
            if let Some(rel) = p.strip_prefix(&format!("{}/", src.trim_end_matches('/'))) {
                if let Some(i) = self.files.iter().position(|f| f == rel) {
                    self.load_image(ctx, i);
                    return;
                }
            }
        }
        self.open_dataset(ctx, path);
    }

    /// Ask the store to open a dataset; `on_opened` continues when it is there.
    fn open_dataset(&mut self, ctx: &egui::Context, path: &str) {
        self.accept_magic();
        self.save_current(false);
        if let Some(b) = self.batch.take() {
            self.store.ai(AiReq::CancelBatch { id: b.id });
        }
        self.magic = Magic { seq: self.magic.seq + 1, ..Default::default() };
        self.store.open(path);
        self.pump(ctx);
    }

    fn on_opened(&mut self, ctx: &egui::Context, res: Result<OpenedData, String>, dirty: bool) {
        let d = match res {
            Ok(d) => d,
            Err(e) => {
                self.status = e;
                return;
            }
        };
        // `open` may have named an image: jump to it after loading its folder
        self.project = d.project;
        self.files = d.files;
        self.tree = DirNode::build(&self.files);
        // start with the top-level folders expanded
        self.expanded = self.tree.dirs.iter().map(|d| d.path.clone()).collect();
        self.source = Some(d.label);
        self.loaded = None;
        self.pending_load = None;
        self.undo.clear();
        self.stale.clear();
        self.selected = None;
        self.drag = Drag::None;
        self.class = 0;
        self.dirty = dirty;
        self.label_dirty = false;
        self.labels_stale = false;
        self.unacked.clear();
        self.encoded.clear();
        self.status = d.status;
        if !self.files.is_empty() {
            // resume at the first image that has no annotation yet
            let start = self.files.iter().position(|f| !self.project.images.contains_key(f)).unwrap_or(0);
            self.load_image(ctx, start);
        }
    }

    fn load_image(&mut self, ctx: &egui::Context, idx: usize) {
        if idx >= self.files.len() {
            return;
        }
        // the frame being left: the first choice to copy labels from
        let left = self.loaded.as_ref().map(|l| l.name.clone());
        // keep a polygon / polyline that is still being drawn, and a Magic result
        if !self.drawing.is_empty() {
            self.finish_drawing();
        }
        self.accept_magic();
        self.end_drag();
        self.save_current(false);
        let name = self.files[idx].clone();
        self.cur = idx;
        self.selected = None;
        self.sel_point = None;
        self.drawing.clear();
        self.undo.clear();
        self.tri_cache.borrow_mut().clear();
        self.drag = Drag::None;
        self.fit_pending = true;
        self.scroll_list = true;
        for a in ancestors(&name) {
            self.expanded.insert(a.to_string());
        }
        let next = idx as i64 + self.nav_dir;
        let next = (next >= 0 && (next as usize) < self.files.len()).then(|| self.files[next as usize].clone());
        self.pending_load = Some((idx, left));
        self.store.load_image(&name, next.as_deref());
        // with the Magic tool, SAM prepares the next image in the background too
        if let (Some(n), true) = (&next, self.tool == Tool::Magic && self.ai_started) {
            self.store.ai(AiReq::Encode { key: n.clone() });
        }
        self.pump(ctx);
    }

    fn on_image(&mut self, ctx: &egui::Context, name: String, res: Result<ImageData, String>) {
        let Some((idx, left)) = self.pending_load.take_if(|(i, _)| self.files.get(*i) == Some(&name)) else {
            return; // an image the user has already left
        };
        match res {
            Ok(img) => {
                let ci = egui::ColorImage::from_rgba_unmultiplied(img.size, &img.rgba);
                let tex = ctx.load_texture(&name, ci, egui::TextureOptions::LINEAR);
                let (w, h) = (img.w, img.h);
                // (normalized positions are unchanged, so the label file needs no rewrite)
                if let Some(a) = self.project.images.get_mut(&name) {
                    if (a.width, a.height) != (w, h) {
                        rescale(a, w, h);
                        self.dirty = true;
                    }
                } else if let Some(s) = img.label_text {
                    // a YOLO label file that appeared since the dataset was opened
                    let shapes = parse_yolo(&s, w, h);
                    self.project.images.insert(name.clone(), ImageAnn { width: w, height: h, shapes });
                    self.project.ensure_classes();
                    self.dirty = true;
                }
                self.loaded = Some(Loaded { name, tex, w, h });
                if self.tool == Tool::Magic {
                    self.request_encode();
                }
                if !self.project.images.contains_key(&self.files[idx]) && self.locked_by_other().is_none() {
                    self.prefill_new_frame(idx, left.as_deref());
                }
            }
            Err(e) => {
                self.loaded = None;
                self.status = format!("Cannot load {name}: {e}");
            }
        }
    }

    fn goto(&mut self, ctx: &egui::Context, delta: i64) {
        if self.files.is_empty() {
            return;
        }
        let n = self.files.len() as i64;
        self.nav_dir = delta.signum();
        if self.filtering() {
            // with a filter / "Unlabeled only", A/D step through the matching images
            let mut i = self.cur as i64 + delta.signum();
            while i >= 0 && i < n {
                if self.matches_filter(i as usize) {
                    self.load_image(ctx, i as usize);
                    return;
                }
                i += delta.signum();
            }
            self.status = "No more matching images in this direction.".into();
            return;
        }
        let i = (self.cur as i64 + delta).clamp(0, n - 1) as usize;
        if i != self.cur || self.loaded.is_none() {
            self.load_image(ctx, i);
        }
    }

    fn filtering(&self) -> bool {
        !self.filter.trim().is_empty() || self.unlabeled_only
    }

    fn matches_filter(&self, i: usize) -> bool {
        let f = self.filter.trim().to_lowercase();
        let rel = &self.files[i];
        (f.is_empty() || rel.to_lowercase().contains(&f))
            && (!self.unlabeled_only || !self.project.images.contains_key(rel))
    }

    fn save_current(&mut self, force: bool) {
        if self.source.is_none() {
            return;
        }
        if self.lab() {
            // the server keeps annotations.json and rewrites other label files itself
            if self.label_dirty || force {
                self.write_current_label();
            }
            self.dirty = false;
            self.stale.clear();
            self.labels_stale = false;
            return;
        }
        if !self.dirty && !self.labels_stale && self.stale.is_empty() && !force {
            return;
        }
        let mut errs = vec![];
        // label files first, annotations.json last: on the next start a label file newer than
        // annotations.json means "not recorded in the project file" and is re-read
        let names: Vec<String> = if self.labels_stale {
            self.project.images.keys().cloned().collect()
        } else {
            self.stale.drain().collect()
        };
        self.labels_stale = false;
        let mut failed = 0;
        for name in names {
            let Some(a) = self.project.images.get(&name) else { continue };
            if let Err(e) = self.store.write_label(&name, Some(a), &self.project) {
                failed += 1;
                if errs.len() < 3 {
                    errs.push(e);
                }
                self.stale.insert(name); // try again at the next save
            }
        }
        if failed > 3 {
            errs.push(format!("... {failed} label files could not be written"));
        }
        if self.label_dirty || force {
            self.write_current_label();
        }
        let saved = self.store.save_project(&self.project);
        if let Err(e) = &saved {
            errs.push(e.clone());
        }
        self.dirty = false;
        self.save_error = (!errs.is_empty()).then(|| errs.join("; "));
        if !errs.is_empty() {
            self.status = format!("Save error: {}", errs.join("; "));
        } else if force {
            self.status = format!("Saved. {}", saved.unwrap_or_default());
        }
    }

    /// Write the YOLO .txt of the current image right away.
    fn write_current_label(&mut self) {
        self.label_dirty = false;
        let Some(l) = &self.loaded else { return };
        if self.locked_by_other().is_some() {
            return; // view only
        }
        let name = l.name.clone();
        let ann = self.project.images.get(&name);
        if ann.is_none() && !self.lab() {
            return;
        }
        let n = ann.map_or(0, |a| a.shapes.len());
        match self.store.write_label(&name, ann, &self.project) {
            Ok(shown) => {
                if self.lab() {
                    self.unacked.insert(name);
                } else {
                    self.status = format!("✔ {n} objects -> {shown}");
                }
                self.label_error = None;
            }
            Err(e) => {
                self.status = format!("Save error: {e}");
                self.label_error = Some(e);
                self.stale.insert(name); // try again at the next save
            }
        }
    }

    /// A never-labeled frame was opened: fill it as chosen under "Video / frame sequence".
    fn prefill_new_frame(&mut self, idx: usize, left: Option<&str>) {
        match self.project.new_frames {
            NewFrames::Empty => {}
            NewFrames::CopyPrevious => {
                if let Some(src) = self.copy_source(idx, left, false) {
                    let n = self.copy_from(&src);
                    self.status = format!("New frame: {n} shapes copied from {src}. Fix them with Points (E); Ctrl+Z undoes.");
                }
            }
            NewFrames::AutoLabel => {
                if self.ai_settings.yolo.trim().is_empty() {
                    self.status = "New frames: choose your YOLO model under 'AI assist' to auto-label them.".into();
                } else {
                    self.auto_label();
                }
            }
        }
    }

    /// Frame to copy labels from: `left` (the frame just left) if it is in the same folder and
    /// has shapes, else the nearest such frame before `idx` (then after it, if `both_ways`).
    fn copy_source(&self, idx: usize, left: Option<&str>, both_ways: bool) -> Option<String> {
        let name = self.files.get(idx)?;
        let folder = |n: &str| n.rsplit_once('/').map_or("", |x| x.0).to_string();
        let here = folder(name);
        let ok = |n: &str| {
            n != name
                && folder(n) == here
                && self.project.images.get(n).is_some_and(|a| !a.shapes.is_empty())
        };
        if let Some(l) = left.filter(|l| ok(l)) {
            return Some(l.to_string());
        }
        let near = |it: &mut dyn Iterator<Item = usize>| it.map(|i| &self.files[i]).find(|n| ok(n)).cloned();
        near(&mut (0..idx).rev()).or_else(|| if both_ways { near(&mut (idx + 1..self.files.len())) } else { None })
    }

    /// Replace the current frame's shapes with those of `src` (scaled if the size differs).
    fn copy_from(&mut self, src: &str) -> usize {
        let Some(l) = &self.loaded else { return 0 };
        let (name, w, h) = (l.name.clone(), l.w, l.h);
        let Some(a) = self.project.images.get(src) else { return 0 };
        let (sx, sy) = (w as f32 / a.width.max(1) as f32, h as f32 / a.height.max(1) as f32);
        let shapes: Vec<Annot> = a
            .shapes
            .iter()
            .map(|s| Annot { pts: s.pts.iter().map(|p| [p[0] * sx, p[1] * sy]).collect(), ..s.clone() })
            .collect();
        let n = shapes.len();
        self.push_undo();
        self.project.images.insert(name, ImageAnn { width: w, height: h, shapes });
        self.selected = None;
        self.sel_point = None;
        self.touch();
        n
    }

    /// "Copy previous frame" (C): the nearest labeled frame before this one (else after), same folder.
    fn copy_previous(&mut self) {
        if self.loaded.is_none() {
            return;
        }
        match self.copy_source(self.cur, None, true) {
            Some(src) => {
                let n = self.copy_from(&src);
                self.status = format!("{n} shapes copied from {src} (Ctrl+Z to undo).");
            }
            None => self.status = "No labeled frame in this folder to copy from.".into(),
        }
    }

    /// Mark the current image as changed (label file + project file).
    fn touch(&mut self) {
        self.dirty = true;
        self.label_dirty = true;
    }

    // ------------------------------------------------------------ editing

    fn shapes(&self) -> &[Annot] {
        self.loaded
            .as_ref()
            .and_then(|l| self.project.images.get(&l.name))
            .map(|a| a.shapes.as_slice())
            .unwrap_or(&[])
    }

    fn shapes_mut(&mut self) -> Option<&mut Vec<Annot>> {
        let l = self.loaded.as_ref()?;
        let e = self
            .project
            .images
            .entry(l.name.clone())
            .or_insert_with(|| ImageAnn {
                width: l.w,
                height: l.h,
                shapes: vec![],
            });
        Some(&mut e.shapes)
    }

    fn shape_mut(&mut self, i: usize) -> Option<&mut Annot> {
        self.shapes_mut().and_then(|v| v.get_mut(i))
    }

    fn push_undo(&mut self) {
        let s = self
            .loaded
            .as_ref()
            .and_then(|l| self.project.images.get(&l.name))
            .map(|a| a.shapes.clone());
        self.undo.push(s);
        if self.undo.len() > 200 {
            self.undo.remove(0);
        }
    }

    fn undo(&mut self) {
        // a drag in progress refers to shape indices that may not exist after the undo
        self.drag = Drag::None;
        if let Some(prev) = self.undo.pop() {
            match prev {
                Some(shapes) => {
                    if let Some(s) = self.shapes_mut() {
                        *s = shapes;
                    }
                    self.touch();
                }
                None => {
                    // back to "never annotated": not a background image, no label file
                    if let Some(name) = self.loaded.as_ref().map(|l| l.name.clone()) {
                        self.project.images.remove(&name);
                        let _ = self.store.write_label(&name, None, &self.project);
                        if self.lab() {
                            self.unacked.insert(name);
                        }
                    }
                    self.label_dirty = false;
                    self.dirty = true;
                    self.status = "Undone: this image is unlabeled again (label file removed).".into();
                }
            }
            self.selected = None;
        }
    }

    fn add_shape(&mut self, a: Annot) {
        if self.project.classes.is_empty() {
            if self.lab() {
                self.store.op(ClassOp::Add("object".into()));
            } else {
                self.project.classes.push("object".into());
            }
        }
        self.push_undo();
        if let Some(v) = self.shapes_mut() {
            v.push(a);
            // keep the new shape selected so it can be adjusted right away
            self.selected = Some(v.len() - 1);
        }
        self.touch();
    }

    fn delete_shape(&mut self, i: usize) {
        if i < self.shapes().len() {
            self.push_undo();
            if let Some(v) = self.shapes_mut() {
                v.remove(i);
            }
            self.selected = None;
            self.drag = Drag::None;
            self.touch();
        }
    }

    fn set_class(&mut self, c: usize) {
        self.class = c;
        if let Some(sel) = self.selected {
            if sel < self.shapes().len() {
                self.push_undo();
                if let Some(s) = self.shape_mut(sel) {
                    s.class = c;
                }
                self.touch();
            }
        }
    }

    fn set_tool(&mut self, t: Tool) {
        if !matches!(t, Tool::Select | Tool::Points) {
            self.last_draw_tool = t;
        }
        // keep a polygon / polyline that is still being drawn, and a Magic result
        if !self.drawing.is_empty() {
            self.finish_drawing();
        }
        if t != Tool::Magic {
            self.accept_magic();
        }
        self.end_drag();
        self.tool = t;
        self.drawing.clear();
        if t == Tool::Magic {
            self.ensure_ai();
            self.request_encode();
        }
    }

    // ------------------------------------------------------------ AI assist

    fn ensure_ai(&mut self) {
        if !self.ai_started {
            self.ai_started = true;
            // (lab: the server keeps the shared settings; the AI starts with the first request)
            if !self.lab() {
                self.store.ai(AiReq::Settings(self.ai_settings.clone()));
            }
            if !self.ai_status.starts_with("ready") && !self.lab() {
                self.ai_status = "starting...".into();
            }
        }
    }

    fn cur_key(&self) -> Option<String> {
        self.loaded.as_ref().map(|l| l.name.clone())
    }

    /// Ask SAM to prepare the current image (no-op if it already is).
    fn request_encode(&mut self) {
        if !self.ai_started {
            return;
        }
        let Some(key) = self.cur_key() else { return };
        if !self.encoded.contains(&key) {
            self.store.ai(AiReq::Encode { key });
        }
    }

    /// Send the current Magic prompt to SAM (or remember to, once the image is ready).
    fn send_prompt(&mut self) {
        if self.magic.is_empty() {
            self.magic.shape = None;
            self.magic.unsent = false;
            self.magic.seq += 1; // ignore the answer to the prompt that was just emptied
            return;
        }
        let Some(key) = self.cur_key() else { return };
        if !self.ai_started {
            return;
        }
        if !self.encoded.contains(&key) {
            self.magic.unsent = true;
            self.request_encode(); // (again, in case an earlier request was dropped)
            return;
        }
        self.magic.seq += 1;
        self.magic.unsent = false;
        let polygon = self.project.export_mode == ExportMode::Segment;
        self.store.ai(AiReq::Segment {
            key,
            seq: self.magic.seq,
            points: self.magic.points.clone(),
            bbox: self.magic.bbox,
            polygon,
        });
    }

    /// Turn the Magic preview into a normal (editable) shape.
    fn accept_magic(&mut self) {
        let m = std::mem::take(&mut self.magic);
        self.magic.seq = m.seq + 1; // answers to older prompts must stay ignored
        let Some(pts) = m.shape else { return };
        let shape = if m.shape_is_box {
            Annot { class: self.class, kind: Kind::BBox, pts }
        } else {
            Annot { class: self.class, kind: Kind::Polygon, pts }
        };
        self.add_shape(shape);
        self.status = "Magic: object added. Drag its points to adjust, or click the next object.".into();
    }

    fn no_model_msg(&self) -> String {
        if self.lab() {
            "Auto-label needs a YOLO model: choose one under 'AI assist' (right panel). Models are .onnx files in the server's models folder.".into()
        } else {
            "Auto-label needs your YOLO model: choose a .onnx file under 'AI assist' (right panel).".into()
        }
    }

    fn auto_label(&mut self) {
        if self.ai_settings.yolo.trim().is_empty() {
            self.status = self.no_model_msg();
            return;
        }
        let Some(key) = self.cur_key() else { return };
        self.ensure_ai();
        self.store.ai(AiReq::Detect { key });
        self.status = "Auto-label: running the YOLO model...".into();
    }

    fn auto_label_all(&mut self) {
        if self.ai_settings.yolo.trim().is_empty() {
            self.status = self.no_model_msg();
            return;
        }
        let keys: Vec<String> =
            self.files.iter().filter(|f| !self.project.images.contains_key(*f)).cloned().collect();
        if keys.is_empty() {
            self.status = "Every image is already labeled.".into();
            return;
        }
        self.ensure_ai();
        self.batch_seq += 1;
        let job = BatchJob { id: self.batch_seq, done: 0, total: keys.len(), added: 0 };
        self.store.ai(AiReq::Batch { id: job.id, keys });
        self.batch = Some(job);
    }

    /// Classes a YOLO result needs that the project does not have yet.
    fn classes_needed(&self, dets: &[Detection]) -> Vec<String> {
        let mut need = self.project.missing_classes(dets);
        let max_id = dets.iter().filter(|d| d.class_name.is_none()).map(|d| d.class_id).max();
        if let Some(m) = max_id {
            for i in self.project.classes.len() + need.len()..=m {
                need.push(format!("class_{i}"));
            }
        }
        need
    }

    /// Add a single-image YOLO result to the current image.
    fn apply_detections(&mut self, key: String, dets: Vec<Detection>, w: u32, h: u32) {
        if Some(&key) != self.cur_key().as_ref() {
            self.status = "Auto-label result discarded: the image was changed meanwhile.".into();
            return;
        }
        if self.lab() {
            // new classes are added by the server for everyone first (same order everywhere)
            let need = self.classes_needed(&dets);
            if !need.is_empty() {
                for n in need {
                    self.store.op(ClassOp::Add(n));
                }
                self.pending_dets = Some((key, dets, w, h));
                self.status = "Auto-label: adding the model's classes...".into();
                return;
            }
        }
        let before = self.project.classes.clone();
        let shapes = self.project.detections_to_shapes(&dets, w, h);
        if self.project.classes != before {
            self.dirty = true;
        }
        let n = shapes.len();
        if n > 0 {
            self.push_undo();
            if let Some(v) = self.shapes_mut() {
                v.extend(shapes);
            }
            self.selected = None;
            self.touch();
        }
        self.status = format!("Auto-label: {n} objects added (Ctrl+Z to undo). Adjust them with the Selector (V).");
    }

    fn on_ai(&mut self, m: AiResp) {
        match m {
            AiResp::Ready(s) => self.ai_status = s,
            AiResp::Unavailable(e) => {
                self.ai_status = format!("unavailable: {e}");
                self.status = format!("AI: {e}");
            }
            AiResp::Crashed => {
                self.ai_status = "unavailable: the AI worker crashed (restart the app)".into();
                self.status = "AI: the AI worker crashed; restart the app to use AI again.".into();
                self.batch = None;
                self.encoded.clear();
            }
            AiResp::EncodeFailed { key, err } => {
                if Some(&key) == self.cur_key().as_ref() {
                    self.status = format!("AI could not prepare this image: {err}");
                }
                self.encode_error = Some((key, err));
            }
            AiResp::Info(s) => self.status = s,
            AiResp::Error(e) => {
                self.status = format!("AI: {e}");
                if self.ai_status.starts_with("starting") {
                    self.ai_status = format!("unavailable: {e}");
                }
            }
            AiResp::Encoded { key, ms } => {
                if ms > 0 {
                    self.last_encode_ms = ms as u128;
                }
                if self.encode_error.as_ref().is_some_and(|e| e.0 == key) {
                    self.encode_error = None;
                }
                self.encoded.insert(key.clone());
                if self.encoded.len() > 64 {
                    self.encoded.clear(); // the worker keeps only a few anyway
                    self.encoded.insert(key.clone());
                }
                if Some(&key) == self.cur_key().as_ref() && self.magic.unsent {
                    self.send_prompt();
                }
            }
            AiResp::NeedEncode(key) => {
                self.encoded.remove(&key);
                if Some(&key) == self.cur_key().as_ref() && self.tool == Tool::Magic {
                    self.magic.unsent = !self.magic.is_empty();
                    self.request_encode();
                }
            }
            AiResp::Mask { key, seq, shape } => {
                if Some(&key) == self.cur_key().as_ref() && seq == self.magic.seq {
                    if shape.is_none() {
                        self.status = "Magic: no object found there. Try another point or drag a box.".into();
                    }
                    self.magic.shape_is_box = shape.as_ref().is_some_and(|s| s.len() == 2);
                    self.magic.shape = shape;
                }
            }
            AiResp::Detected { key, batch, dets, w, h } => {
                if let Some(id) = batch {
                    // (desktop) a late result of a cancelled batch is dropped
                    let current = self.batch.as_ref().is_some_and(|b| b.id == id);
                    if !current || dets.is_empty() || self.project.images.contains_key(&key) {
                        return;
                    }
                    let shapes = self.project.detections_to_shapes(&dets, w, h);
                    self.project.images.insert(key.clone(), ImageAnn { width: w, height: h, shapes });
                    self.stale.insert(key);
                    self.dirty = true;
                    if let Some(b) = &mut self.batch {
                        b.added += 1;
                    }
                } else {
                    self.apply_detections(key, dets, w, h);
                }
            }
            AiResp::BatchProgress { id, done, total } => {
                if let Some(b) = self.batch.as_mut().filter(|b| b.id == id) {
                    b.done = done;
                    b.total = total;
                }
            }
            AiResp::BatchDone { id, labeled, errors, cancelled } => {
                if self.batch.as_ref().map_or(true, |b| b.id != id) {
                    return; // an older, replaced batch
                }
                let lab = self.lab();
                let added = self.batch.take().map_or(labeled, |b| if lab { labeled } else { b.added });
                self.save_current(false);
                self.status = format!(
                    "Auto-label {}: {added} images labeled{}",
                    if cancelled { "cancelled" } else { "finished" },
                    if errors.is_empty() { String::new() } else { format!("; errors: {}", errors.join("; ")) }
                );
            }
            AiResp::Models { sam, yolo } => {
                self.ai_models = sam;
                self.yolo_models = yolo;
            }
            AiResp::Settings(s) => self.ai_settings = s,
        }
    }

    fn ai_settings_changed(&mut self, sam_changed: bool) {
        if !self.lab() {
            self.ai_settings.save();
        }
        if self.ai_started {
            self.store.ai(AiReq::Settings(self.ai_settings.clone()));
            if sam_changed {
                // a new SAM model needs new embeddings
                self.encoded.clear();
                self.encode_error = None;
                self.magic = Magic { seq: self.magic.seq + 1, ..Default::default() };
                self.ai_status = "loading model...".into();
                if self.tool == Tool::Magic {
                    self.request_encode();
                }
            } else if !self.magic.is_empty() {
                // e.g. Detail changed: refresh the preview
                self.send_prompt();
            }
        }
    }

    // ------------------------------------------------------------ store events

    /// Handle everything the store has for us (local stores answer right away).
    fn pump(&mut self, ctx: &egui::Context) {
        for _ in 0..8 {
            let events = self.store.poll();
            if events.is_empty() {
                break;
            }
            for e in events {
                self.on_event(ctx, e);
            }
        }
    }

    fn on_event(&mut self, ctx: &egui::Context, e: StoreEvent) {
        match e {
            StoreEvent::Opened(res, dirty) => self.on_opened(ctx, res, dirty),
            StoreEvent::Image(name, res) => self.on_image(ctx, name, res),
            StoreEvent::Ai(m) => self.on_ai(m),
            StoreEvent::ExportProgress { done, total } => self.export_progress = Some((done, total)),
            StoreEvent::ExportDone(res) => {
                self.export_progress = None;
                match res {
                    Ok(dest) => {
                        self.status = format!("Exported: {dest}");
                        self.export.open = false;
                    }
                    Err(e) => self.status = format!("Export failed: {e}"),
                }
            }
            StoreEvent::Connected(c) => {
                self.connected = c;
                if !c {
                    self.status = "Connection to the lab server lost; reconnecting...".into();
                } else if !self.user.is_empty() {
                    let u = self.user.clone();
                    self.store.hello(&u);
                }
            }
            StoreEvent::Welcome { admin, user } => {
                self.admin = admin;
                if user != self.user {
                    self.status = format!("The name '{}' is in use; you are '{user}'.", self.user);
                    self.user = user.clone();
                    self.user_edit = user;
                }
            }
            StoreEvent::Locks(l) => {
                let was = self.locked_by_other().map(str::to_string);
                self.locks = l;
                // the other user left this image: it becomes editable (the server gave it to us)
                if was.is_some() && self.locked_by_other().is_none() {
                    if let Some(n) = self.cur_key() {
                        self.status = format!("{n} is free now; you can edit it.");
                    }
                }
            }
            StoreEvent::ImageChanged { name, ann, by } => {
                match ann {
                    Some(a) => {
                        self.project.images.insert(name.clone(), a);
                    }
                    None => {
                        self.project.images.remove(&name);
                    }
                }
                if self.cur_key().as_ref() == Some(&name) {
                    self.selected = None;
                    self.sel_point = None;
                    self.undo.clear();
                    self.status = format!("{name} was changed by {by}.");
                }
            }
            StoreEvent::Op { op, by } => {
                // "pre-fill new frames" is each lab member's own choice
                let keep = self.project.new_frames;
                if let Err(e) = self.project.apply_op(&op) {
                    self.status = format!("Class change by {by} failed here: {e}");
                }
                self.project.new_frames = keep;
                if let ClassOp::Add(n) = &op {
                    if by == self.user {
                        if let Some(i) = self.project.classes.iter().position(|c| c == n.trim()) {
                            self.class = i;
                        }
                    }
                }
                if let ClassOp::Delete(c) = op {
                    if self.class > c || self.class >= self.project.classes.len() {
                        self.class = self.class.saturating_sub(1);
                    }
                    self.undo.clear();
                }
                // a YOLO result was waiting for these classes
                if let Some((key, dets, w, h)) = self.pending_dets.take() {
                    if self.classes_needed(&dets).is_empty() {
                        self.apply_detections(key, dets, w, h);
                    } else {
                        self.pending_dets = Some((key, dets, w, h));
                    }
                }
            }
            StoreEvent::OpFailed(e) => {
                self.status = e;
                self.pending_dets = None;
            }
            StoreEvent::Saved { name, result } => {
                self.unacked.remove(&name);
                match result {
                    Ok(shown) => {
                        self.label_error = None;
                        if self.cur_key().as_ref() == Some(&name) {
                            let n = self.project.images.get(&name).map_or(0, |a| a.shapes.len());
                            self.status = format!("✔ {n} objects -> {shown}");
                        }
                    }
                    Err(e) => {
                        self.status = format!("Save error ({name}): {e}");
                        self.label_error = Some(e);
                    }
                }
            }
            StoreEvent::Users(u) => self.users = u,
            StoreEvent::Status(s) => self.status = s,
        }
    }

    /// Finish an edit drag: tidy the box corners; a box squashed to nothing is undone.
    /// While a polygon / polyline point is dragged: the neighbouring point it would merge
    /// into if released now (within HIT_TOL screen pixels). Returns (shape, neighbour index).
    fn snap_target(&self) -> Option<(usize, usize)> {
        let Drag::Vertex { shape, idx } = self.drag else { return None };
        let s = self.shapes().get(shape).filter(|s| s.kind != Kind::BBox)?;
        let n = s.pts.len();
        let min = if s.kind == Kind::Polygon { 3 } else { 2 };
        if idx >= n || n <= min {
            return None;
        }
        let closed = s.kind == Kind::Polygon;
        let prev = if idx > 0 { Some(idx - 1) } else { closed.then(|| n - 1) };
        let next = if idx + 1 < n { Some(idx + 1) } else { closed.then_some(0) };
        let tol = HIT_TOL / self.zoom;
        [prev, next]
            .into_iter()
            .flatten()
            .map(|j| (j, dist(s.pts[idx], s.pts[j])))
            .filter(|(_, d)| *d <= tol)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(j, _)| (shape, j))
    }

    fn end_drag(&mut self) {
        // snap-merge: a point released onto its neighbour becomes one point
        // (the drag already pushed an undo step, so Ctrl+Z restores both)
        if let (Drag::Vertex { idx, .. }, Some((shape, j))) = (self.drag, self.snap_target()) {
            if let Some(s) = self.shape_mut(shape) {
                s.pts.remove(idx);
            }
            self.sel_point = Some((shape, if j > idx { j - 1 } else { j }));
            self.touch();
            self.status = "Points merged (Ctrl+Z to undo).".into();
        }
        if let Drag::Vertex { shape, .. } | Drag::Move { shape, .. } = self.drag {
            let resizing = matches!(self.drag, Drag::Vertex { .. });
            let mut degenerate = false;
            if let Some(s) = self.shape_mut(shape) {
                s.normalize_box();
                let b = s.bounds();
                degenerate = resizing
                    && s.kind == Kind::BBox
                    && (b[2] - b[0] < 1.0 || b[3] - b[1] < 1.0);
            }
            if degenerate {
                self.undo();
                self.status = "Box became too small, change undone.".into();
            }
        }
        self.drag = Drag::None;
    }

    /// Begin dragging a handle / shape at image point `p`. Returns false if nothing was hit.
    /// Select tool: any shape. Drawing tools: only the selected shape (so new shapes
    /// can still be drawn on top of / next to existing ones).
    fn try_start_edit(&mut self, p: [f32; 2], tol: f32) -> bool {
        let hit = if self.tool == Tool::Points {
            match self.find_point_hover(p, tol) {
                Some(PointHover::Vertex { shape, idx }) => Some((shape, Drag::Vertex { shape, idx })),
                Some(h @ PointHover::Edge { shape, edge, .. }) => {
                    // drag on an edge: add a point there and move it right away
                    // (apply_point_edit already pushed the undo step)
                    self.apply_point_edit(h);
                    self.drag = Drag::Vertex { shape, idx: edge + 1 };
                    return true;
                }
                None => None,
            }
        } else if self.tool == Tool::Select {
            if let Some((si, vi)) = self.hit_vertex(p, tol) {
                Some((si, Drag::Vertex { shape: si, idx: vi }))
            } else {
                self.hit_shape(p, tol)
                    .map(|si| (si, Drag::Move { shape: si, last: p }))
            }
        } else if let Some(si) = self.selected.filter(|s| *s < self.shapes().len()) {
            let s = &self.shapes()[si];
            if let Some(vi) = nearest_handle(s, p, tol) {
                Some((si, Drag::Vertex { shape: si, idx: vi }))
            } else if dist(s.center(), p) <= tol
                || (self.tool != Tool::BBox && shape_contains(s, p, tol))
            {
                Some((si, Drag::Move { shape: si, last: p }))
            } else {
                None
            }
        } else {
            None
        };
        match hit {
            Some((si, d)) => {
                self.push_undo();
                self.selected = Some(si);
                self.sel_point = match d {
                    Drag::Vertex { idx, .. } if self.shapes()[si].kind != Kind::BBox => Some((si, idx)),
                    _ => None,
                };
                self.drag = d;
                true
            }
            None => false,
        }
    }

    /// What a drag at `p` would do, as a cursor icon (None = nothing to edit there).
    fn edit_cursor(&self, p: [f32; 2], tol: f32) -> Option<CursorIcon> {
        let shapes = self.shapes();
        if self.tool == Tool::Points {
            return match self.find_point_hover(p, tol) {
                Some(PointHover::Vertex { .. }) => Some(CursorIcon::PointingHand),
                Some(PointHover::Edge { .. }) => Some(CursorIcon::Crosshair),
                None => self.hit_shape(p, tol).map(|_| CursorIcon::PointingHand),
            };
        }
        let (si, vi) = if self.tool == Tool::Select {
            match self.hit_vertex(p, tol) {
                Some((si, vi)) => (si, Some(vi)),
                None => return self.hit_shape(p, tol).map(|_| CursorIcon::Move),
            }
        } else {
            let si = self.selected.filter(|s| *s < shapes.len())?;
            let s = &shapes[si];
            match nearest_handle(s, p, tol) {
                Some(vi) => (si, Some(vi)),
                None if dist(s.center(), p) <= tol
                    || (self.tool != Tool::BBox && shape_contains(s, p, tol)) =>
                {
                    return Some(CursorIcon::Move)
                }
                None => return None,
            }
        };
        Some(match (shapes[si].kind, vi) {
            (Kind::BBox, Some(0 | 2)) => CursorIcon::ResizeNwSe,
            (Kind::BBox, Some(1 | 3)) => CursorIcon::ResizeNeSw,
            (Kind::BBox, Some(4 | 6)) => CursorIcon::ResizeVertical,
            (Kind::BBox, Some(_)) => CursorIcon::ResizeHorizontal,
            _ => CursorIcon::PointingHand,
        })
    }

    fn finish_drawing(&mut self) {
        let tol = 3.0 / self.zoom;
        let mut pts: Vec<[f32; 2]> = vec![];
        for p in &self.drawing {
            if pts.last().map_or(true, |q| dist(*q, *p) > tol) {
                pts.push(*p);
            }
        }
        let kind = if self.tool == Tool::Polygon {
            Kind::Polygon
        } else {
            Kind::Polyline
        };
        if kind == Kind::Polygon && pts.len() > 3 && dist(pts[0], *pts.last().unwrap()) < tol * 3.0 {
            pts.pop();
        }
        let min = if kind == Kind::Polygon { 3 } else { 2 };
        if pts.len() < min {
            if pts.len() <= 1 {
                self.drawing.clear();
            } else {
                self.status = format!("A {} needs at least {min} points.", kind.name());
            }
            return;
        }
        self.drawing.clear();
        self.add_shape(Annot {
            class: self.class,
            kind,
            pts,
        });
    }

    /// (shape index, handle index). The selected shape gets priority.
    fn hit_vertex(&self, p: [f32; 2], tol: f32) -> Option<(usize, usize)> {
        let shapes = self.shapes();
        let mut order: Vec<usize> = (0..shapes.len()).rev().collect();
        if let Some(s) = self.selected.filter(|s| *s < shapes.len()) {
            order.retain(|i| *i != s);
            order.insert(0, s);
        }
        for si in order {
            if let Some(vi) = nearest_handle(&shapes[si], p, tol) {
                return Some((si, vi));
            }
        }
        None
    }

    /// Vertex or edge of the selected polygon / polyline at image point `p` (vertices win).
    fn find_point_hover(&self, p: [f32; 2], tol: f32) -> Option<PointHover> {
        let si = self.selected?;
        let s = self.shapes().get(si).filter(|s| s.kind != Kind::BBox)?;
        if let Some(idx) = nearest_handle(s, p, tol) {
            return Some(PointHover::Vertex { shape: si, idx });
        }
        edges(s)
            .into_iter()
            .map(|(i, a, b)| (i, a, b, dist_to_segment(p, a, b)))
            .filter(|e| e.3 <= tol)
            .min_by(|x, y| x.3.total_cmp(&y.3))
            .map(|(edge, a, b, _)| {
                // the new point goes exactly onto the edge, not where the mouse is
                let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                let l2 = (dx * dx + dy * dy).max(1e-9);
                let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0);
                PointHover::Edge { shape: si, edge, at: [a[0] + t * dx, a[1] + t * dy] }
            })
    }

    /// The selected point of the selected polygon / polyline, if it is still valid.
    fn selected_point(&self) -> Option<(usize, usize)> {
        let (si, idx) = self.sel_point?;
        let s = self.shapes().get(si)?;
        (self.selected == Some(si) && s.kind != Kind::BBox && idx < s.pts.len()).then_some((si, idx))
    }

    /// Select what is under image point `p`: a point (→ shape + that point) or a shape's area
    /// (→ the whole shape). Nothing there: deselect.
    fn select_at(&mut self, p: Option<[f32; 2]>, tol: f32) {
        let hv = p.and_then(|p| self.hit_vertex(p, tol));
        self.selected = hv.map(|x| x.0).or_else(|| p.and_then(|p| self.hit_shape(p, tol)));
        self.sel_point = hv.filter(|(si, _)| self.shapes()[*si].kind != Kind::BBox);
    }

    /// Add or remove a point of the selected shape as described by `h`.
    fn apply_point_edit(&mut self, h: PointHover) {
        match h {
            PointHover::Vertex { shape, idx } => {
                let Some(s) = self.shapes().get(shape) else { return };
                let min = if s.kind == Kind::Polygon { 3 } else { 2 };
                if idx >= s.pts.len() || s.kind == Kind::BBox {
                    return;
                }
                if s.pts.len() <= min {
                    self.status = format!("Cannot remove: a {} needs at least {min} points.", s.kind.name());
                    return;
                }
                self.push_undo();
                if let Some(s) = self.shape_mut(shape) {
                    s.pts.remove(idx);
                }
            }
            PointHover::Edge { shape, edge, at } => {
                let Some(s) = self.shapes().get(shape) else { return };
                if edge >= s.pts.len() || s.kind == Kind::BBox {
                    return;
                }
                self.push_undo();
                if let Some(s) = self.shape_mut(shape) {
                    s.pts.insert(edge + 1, at);
                }
            }
        }
        self.selected = Some(match h {
            PointHover::Vertex { shape, .. } | PointHover::Edge { shape, .. } => shape,
        });
        self.sel_point = match h {
            PointHover::Vertex { .. } => None,
            PointHover::Edge { shape, edge, .. } => Some((shape, edge + 1)),
        };
        self.point_hover = None; // indices changed: recomputed next frame
        self.touch();
    }

    /// Smallest shape under the point (so small objects inside big ones are reachable).
    fn hit_shape(&self, p: [f32; 2], tol: f32) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (i, s) in self.shapes().iter().enumerate() {
            if shape_contains(s, p, tol) {
                let b = s.bounds();
                let area = (b[2] - b[0]) * (b[3] - b[1]);
                if best.map_or(true, |x| area < x.1) {
                    best = Some((i, area));
                }
            }
        }
        best.map(|b| b.0)
    }

    // ------------------------------------------------------------ input

    fn handle_keys(&mut self, ctx: &egui::Context) {
        if ctx.wants_keyboard_input() {
            return;
        }
        let (undo, save, open, export, autolabel) = ctx.input_mut(|i| {
            (
                i.consume_key(Modifiers::COMMAND, Key::Z),
                i.consume_key(Modifiers::COMMAND, Key::S),
                i.consume_key(Modifiers::COMMAND, Key::O),
                i.consume_key(Modifiers::COMMAND, Key::E),
                i.consume_key(Modifiers::COMMAND, Key::L),
            )
        });
        if autolabel {
            self.auto_label();
        }
        if undo {
            self.undo();
        }
        if save {
            self.save_current(true);
        }
        if open {
            self.pick_dataset(ctx);
        }
        if export {
            self.open_export();
        }
        const NUMS: [Key; 9] = [
            Key::Num1, Key::Num2, Key::Num3, Key::Num4, Key::Num5, Key::Num6, Key::Num7, Key::Num8,
            Key::Num9,
        ];
        let (k, num) = ctx.input(|i| {
            let p = |k| i.key_pressed(k) && !i.modifiers.command;
            (
                [
                    p(Key::V),
                    p(Key::B),
                    p(Key::P),
                    p(Key::L),
                    p(Key::A) || p(Key::ArrowLeft),
                    p(Key::D) || p(Key::ArrowRight),
                    p(Key::Delete),
                    p(Key::Backspace),
                    p(Key::Escape),
                    p(Key::Enter),
                    p(Key::F),
                    p(Key::Q),
                    p(Key::M),
                    p(Key::I) || p(Key::Insert),
                    p(Key::E),
                    p(Key::C),
                ],
                NUMS.iter().position(|k| p(*k)),
            )
        });
        let [v, b, pk, l, prev, next, del, back, esc, enter, f, q, mg, ins, ek, cp] = k;
        if mg {
            self.set_tool(Tool::Magic);
        }
        // Magic prompt: Enter accepts, Backspace removes the last point, Esc clears
        let prompting = self.tool == Tool::Magic && !self.magic.is_empty();
        let (enter, back, esc) = if prompting {
            if enter && self.magic.shape.is_some() {
                self.accept_magic();
            }
            if back {
                if self.magic.points.pop().is_none() {
                    self.magic.bbox = None;
                }
                self.send_prompt();
            }
            if esc {
                self.magic = Magic { seq: self.magic.seq + 1, ..Default::default() };
            }
            (false, false, false)
        } else {
            (enter, back, esc)
        };
        if q {
            // quick toggle: selector <-> last annotation tool
            let t = if self.tool == Tool::Select { self.last_draw_tool } else { Tool::Select };
            self.set_tool(t);
        }
        if v {
            self.set_tool(Tool::Select);
        }
        if ek {
            self.set_tool(Tool::Points);
        }
        if cp {
            self.copy_previous();
        }
        if b {
            self.set_tool(Tool::BBox);
        }
        if pk {
            self.set_tool(Tool::Polygon);
        }
        if l {
            self.set_tool(Tool::Polyline);
        }
        if prev {
            self.goto(ctx, -1);
        }
        if next {
            self.goto(ctx, 1);
        }
        if f {
            self.fit_pending = true;
        }
        if let Some(n) = num {
            if n < self.project.classes.len() {
                self.set_class(n);
            }
        }
        if enter && !self.drawing.is_empty() {
            self.finish_drawing();
        }
        // Edit-points tool: Backspace removes the point under the mouse, I adds one on the edge
        let edit_key = match self.point_hover {
            Some(h @ PointHover::Vertex { .. }) if back => Some(h),
            Some(h @ PointHover::Edge { .. }) if ins => Some(h),
            _ => None,
        };
        if self.tool == Tool::Points && self.drawing.is_empty() && edit_key.is_some() {
            self.apply_point_edit(edit_key.unwrap());
        } else if back && !self.drawing.is_empty() {
            self.drawing.pop();
        } else if del {
            if let Some((si, idx)) = self.selected_point() {
                // delete just that point; the previous one becomes selected, so pressing
                // Del again keeps removing points (never the whole shape)
                let n = self.shapes()[si].pts.len();
                self.apply_point_edit(PointHover::Vertex { shape: si, idx });
                if self.shapes()[si].pts.len() < n {
                    let m = n - 1;
                    self.sel_point = Some((si, if idx == 0 { m - 1 } else { idx - 1 }));
                } else {
                    self.sel_point = Some((si, idx)); // at the minimum: kept, message shown
                }
            } else if let Some(s) = self.selected {
                self.delete_shape(s);
            }
        }
        if esc {
            if !self.drawing.is_empty() {
                self.drawing.clear();
            } else if matches!(self.drag, Drag::NewBox { .. }) {
                self.drag = Drag::None;
            } else if self.selected_point().is_some() {
                self.sel_point = None; // back to the whole shape
            } else {
                self.selected = None;
            }
        }
    }

    // ------------------------------------------------------------ canvas

    fn fit(&mut self, rect: Rect, iw: f32, ih: f32) {
        let z = ((rect.width() - 20.0) / iw)
            .min((rect.height() - 20.0) / ih)
            .max(0.01);
        self.zoom = z;
        self.offset = (rect.size() - Vec2::new(iw * z, ih * z)) / 2.0;
    }

    fn canvas(&mut self, ui: &mut egui::Ui) {
        let (resp, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let rect = resp.rect;
        let painter = painter.with_clip_rect(rect);
        painter.rect_filled(rect, 0.0, Color32::from_gray(28));
        let Some(l) = &self.loaded else {
            let msg = if !self.lab() {
                "Open an image folder (Open button or Ctrl+O), or drag a folder onto this window"
            } else if self.pending_load.is_some() {
                "Loading image..."
            } else if self.source.is_none() && self.connected {
                "Waiting for the lab server's dataset..."
            } else if self.source.is_none() {
                "Connecting to the lab server..."
            } else {
                "Choose an image on the left"
            };
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                msg,
                FontId::proportional(18.0),
                Color32::GRAY,
            );
            return;
        };
        let (iw, ih) = (l.w as f32, l.h as f32);
        let tex_id = l.tex.id();
        if self.fit_pending {
            self.fit(rect, iw, ih);
            self.fit_pending = false;
        }

        let (scroll, zoom_delta, latest, space, press_origin) = ui.input(|i| {
            (
                i.smooth_scroll_delta.y,
                i.zoom_delta(),
                i.pointer.latest_pos(),
                i.key_down(Key::Space),
                i.pointer.press_origin(),
            )
        });

        // zoom around the mouse
        if let Some(m) = resp.hover_pos() {
            let factor = zoom_delta * (scroll * 0.002).exp();
            if (factor - 1.0).abs() > 1e-4 {
                let p = (m - rect.min - self.offset) / self.zoom;
                self.zoom = (self.zoom * factor).clamp(0.02, 60.0);
                self.offset = m - rect.min - p * self.zoom;
            }
        }
        // pan: middle drag, space+left drag, or left drag on empty area in Select mode
        let panning = resp.dragged_by(PointerButton::Middle)
            || (space && resp.dragged_by(PointerButton::Primary))
            || (matches!(self.drag, Drag::Pan) && resp.dragged_by(PointerButton::Primary));
        if panning {
            self.offset += resp.drag_delta();
        }

        let zoom = self.zoom;
        let origin = rect.min + self.offset;
        let to_s = move |p: [f32; 2]| Pos2::new(origin.x + p[0] * zoom, origin.y + p[1] * zoom);
        let to_i = move |p: Pos2| [(p.x - origin.x) / zoom, (p.y - origin.y) / zoom];
        let clampi = move |p: [f32; 2]| [p[0].clamp(0.0, iw), p[1].clamp(0.0, ih)];
        let tol = HIT_TOL / zoom;

        let hover = latest.filter(|p| rect.contains(*p));
        self.cursor_img = hover.map(to_i);
        let pos_img = latest.map(to_i);

        let started = resp.drag_started_by(PointerButton::Primary) && !space;
        let dragging = resp.dragged_by(PointerButton::Primary) && !space;
        let stopped = resp.drag_stopped_by(PointerButton::Primary);
        let clicked = resp.clicked_by(PointerButton::Primary);
        let dbl = resp.double_clicked();
        let rclick = resp.secondary_clicked();

        // ---- drags (all tools): edit a shape, draw a new box, or pan
        if started && self.drawing.is_empty() {
            if let Some(o) = press_origin {
                let p = to_i(o);
                if !self.try_start_edit(p, tol) {
                    match self.tool {
                        // (Magic: the box is a prompt for SAM)
                        Tool::BBox | Tool::Magic => {
                            let s = clampi(p);
                            self.drag = Drag::NewBox { start: s, cur: s };
                        }
                        Tool::Select | Tool::Points => self.drag = Drag::Pan,
                        _ => {}
                    }
                }
            }
        }
        if dragging {
            if let Some(p) = pos_img {
                match self.drag {
                    Drag::NewBox { start, .. } => {
                        self.drag = Drag::NewBox { start, cur: clampi(p) };
                    }
                    Drag::Vertex { shape, idx } => {
                        if let Some(s) = self.shape_mut(shape) {
                            s.set_handle(idx, clampi(p));
                        }
                        self.touch();
                    }
                    Drag::Move { shape, last } => {
                        if let Some(s) = self.shape_mut(shape) {
                            let b = s.bounds();
                            let dx = safe_clamp(p[0] - last[0], -b[0], iw - b[2]);
                            let dy = safe_clamp(p[1] - last[1], -b[1], ih - b[3]);
                            for q in &mut s.pts {
                                q[0] += dx;
                                q[1] += dy;
                            }
                        }
                        self.drag = Drag::Move { shape, last: p };
                        self.touch();
                    }
                    _ => {}
                }
            }
        }
        if stopped {
            match self.drag {
                Drag::NewBox { start, cur } if self.tool == Tool::Magic => {
                    if (start[0] - cur[0]).abs() >= 2.0 && (start[1] - cur[1]).abs() >= 2.0 {
                        self.magic.bbox = Some([
                            start[0].min(cur[0]),
                            start[1].min(cur[1]),
                            start[0].max(cur[0]),
                            start[1].max(cur[1]),
                        ]);
                        self.send_prompt();
                    }
                }
                Drag::NewBox { start, cur } => {
                    if (start[0] - cur[0]).abs() >= 2.0 && (start[1] - cur[1]).abs() >= 2.0 {
                        self.add_shape(Annot {
                            class: self.class,
                            kind: Kind::BBox,
                            pts: vec![
                                [start[0].min(cur[0]), start[1].min(cur[1])],
                                [start[0].max(cur[0]), start[1].max(cur[1])],
                            ],
                        });
                    }
                }
                _ => {}
            }
            self.end_drag();
        }
        let pick = |app: &Self, p: [f32; 2]| {
            app.hit_vertex(p, tol)
                .map(|x| x.0)
                .or_else(|| app.hit_shape(p, tol))
        };

        // ---- clicks (per tool)
        match self.tool {
            Tool::Magic => {
                let inside = |p: [f32; 2]| p[0] >= 0.0 && p[1] >= 0.0 && p[0] <= iw && p[1] <= ih;
                if dbl && self.magic.shape.is_some() {
                    self.accept_magic();
                } else if clicked {
                    // left-click: "this is the object"
                    if let Some(p) = pos_img.filter(|p| inside(*p)) {
                        self.magic.points.push((p, true));
                        self.send_prompt();
                    }
                } else if rclick {
                    if self.magic.is_empty() {
                        // nothing prompted yet: right-click selects a shape (to edit it)
                        self.select_at(pos_img, tol);
                    } else if let Some(p) = pos_img.filter(|p| inside(*p)) {
                        // right-click: "not this part"
                        self.magic.points.push((p, false));
                        self.send_prompt();
                    }
                }
            }
            Tool::BBox => {
                // click (no drag) selects / deselects
                if clicked {
                    self.select_at(pos_img, tol);
                }
            }
            Tool::Polygon | Tool::Polyline => {
                // right-click while not drawing selects / deselects
                if rclick && self.drawing.is_empty() {
                    self.select_at(pos_img, tol);
                }
                if clicked {
                    if let Some(p) = pos_img {
                        let p = clampi(p);
                        let close = self.tool == Tool::Polygon
                            && self.drawing.len() >= 3
                            && to_s(self.drawing[0]).distance(to_s(p)) < CLOSE_TOL;
                        if close {
                            self.finish_drawing();
                        } else {
                            self.drawing.push(p);
                        }
                    }
                }
                if rclick && !self.drawing.is_empty() {
                    // right-click places the last point and finishes
                    if let Some(p) = pos_img {
                        let p = clampi(p);
                        let near_first = to_s(self.drawing[0]).distance(to_s(p)) < CLOSE_TOL;
                        if !(self.tool == Tool::Polygon && near_first) {
                            self.drawing.push(p);
                        }
                    }
                    self.finish_drawing();
                } else if dbl && !self.drawing.is_empty() {
                    self.finish_drawing();
                }
            }
            Tool::Select => {
                // click a point: select that point; click inside / on a shape: the whole shape
                if clicked || rclick {
                    self.select_at(pos_img, tol);
                }
            }
            Tool::Points => {
                // click an edge of the selected shape: add a point; right-click a point:
                // remove it; click / right-click another shape: edit that one instead
                let hover_edit = pos_img.and_then(|p| self.find_point_hover(p, tol));
                let other = pos_img.and_then(|p| pick(self, p));
                if clicked {
                    match hover_edit {
                        Some(h @ PointHover::Edge { .. }) => self.apply_point_edit(h),
                        Some(PointHover::Vertex { shape, idx }) => self.sel_point = Some((shape, idx)),
                        None => {
                            // inside the shape (or another shape): select the whole shape
                            if other.is_some() {
                                self.selected = other;
                                self.sel_point = None;
                            }
                        }
                    }
                }
                if rclick {
                    match hover_edit {
                        Some(h @ PointHover::Vertex { .. }) => self.apply_point_edit(h),
                        Some(PointHover::Edge { .. }) => {}
                        None => {
                            if other.is_some() {
                                self.selected = other;
                                self.sel_point = None;
                            }
                        }
                    }
                }
            }
        }
        // what a point edit would act on now (drawn below; used by Backspace / I)
        self.point_hover = if self.tool == Tool::Points
            && matches!(self.drag, Drag::None)
            && !space
            && !panning
        {
            hover.map(to_i).and_then(|p| self.find_point_hover(p, tol))
        } else {
            None
        };

        // cursor
        if resp.hovered() {
            let icon = if panning || matches!(self.drag, Drag::Pan) {
                CursorIcon::Grabbing
            } else if space {
                CursorIcon::Grab
            } else if let Drag::Vertex { .. } | Drag::Move { .. } = self.drag {
                CursorIcon::Grabbing
            } else {
                let edit = self
                    .cursor_img
                    .filter(|_| self.drawing.is_empty() && !matches!(self.drag, Drag::NewBox { .. }))
                    .and_then(|p| self.edit_cursor(p, tol));
                edit.unwrap_or(if matches!(self.tool, Tool::Select | Tool::Points) {
                    CursorIcon::Default
                } else {
                    CursorIcon::Crosshair
                })
            };
            ui.ctx().set_cursor_icon(icon);
        }

        // ---------------- draw
        painter.image(
            tex_id,
            Rect::from_min_max(to_s([0.0, 0.0]), to_s([iw, ih])),
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
        let band = (self.project.export_mode == ExportMode::Segment)
            .then_some(self.project.polyline_width * zoom);
        let mut tri_cache = self.tri_cache.borrow_mut();
        if tri_cache.len() > 4096 {
            tri_cache.clear();
        }
        for (i, s) in self.shapes().iter().enumerate() {
            // triangulation is independent of zoom / pan, so it is computed once per shape
            let tri: &[u32] = if s.kind == Kind::Polygon {
                tri_cache.entry(pts_key(&s.pts)).or_insert_with(|| {
                    let ip: Vec<Pos2> = s.pts.iter().map(|q| Pos2::new(q[0], q[1])).collect();
                    triangulate(&ip)
                })
            } else {
                &[]
            };
            let label = self
                .project
                .classes
                .get(s.class)
                .cloned()
                .unwrap_or_else(|| format!("#{}", s.class));
            draw_annot(
                &painter,
                s,
                self.selected == Some(i),
                &to_s,
                &label,
                self.show_labels,
                band,
                tri,
            );
        }
        drop(tri_cache);

        // point-edit preview: red ring = right-click / Backspace removes this point,
        // "+" marker = right-click / double-click / I adds a point here
        match self.point_hover {
            Some(PointHover::Vertex { shape, idx }) => {
                if let Some(q) = self.shapes().get(shape).and_then(|s| s.pts.get(idx)) {
                    let c = to_s(*q);
                    let red = Color32::from_rgb(235, 70, 60);
                    painter.circle_stroke(c, 8.0, Stroke::new(2.5_f32, red));
                }
            }
            Some(PointHover::Edge { at, .. }) => {
                let c = to_s(at);
                painter.circle_filled(c, 6.0, Color32::from_black_alpha(160));
                painter.circle_stroke(c, 6.0, Stroke::new(1.5_f32, Color32::WHITE));
                let st = Stroke::new(1.5_f32, Color32::WHITE);
                painter.line_segment([c - Vec2::X * 3.5, c + Vec2::X * 3.5], st);
                painter.line_segment([c - Vec2::Y * 3.5, c + Vec2::Y * 3.5], st);
            }
            None => {}
        }
        // selected point (what Del removes): filled amber dot
        if let Some((si, idx)) = self.selected_point() {
            let q = to_s(self.shapes()[si].pts[idx]);
            painter.circle_filled(q, 6.5, Color32::from_rgb(255, 196, 40));
            painter.circle_stroke(q, 6.5, Stroke::new(1.5_f32, Color32::BLACK));
        }
        // snap-merge preview: releasing now merges the dragged point into this neighbour
        if let Some((shape, j)) = self.snap_target() {
            if let Some(q) = self.shapes().get(shape).and_then(|s| s.pts.get(j)) {
                painter.circle_stroke(to_s(*q), 9.0, Stroke::new(2.5_f32, Color32::from_rgb(90, 220, 120)));
            }
        }

        let cc = class_color(self.class);
        if self.tool == Tool::Magic {
            self.draw_magic(&painter, &to_s, cc, rect);
        }
        if let Drag::NewBox { start, cur } = self.drag {
            let r = Rect::from_two_pos(to_s(start), to_s(cur));
            painter.rect_filled(r, 0.0, cc.gamma_multiply(0.2));
            painter.rect_stroke(r, 0.0, Stroke::new(2.0_f32, cc));
        }
        if !self.drawing.is_empty() {
            let mut sp: Vec<Pos2> = self.drawing.iter().map(|p| to_s(*p)).collect();
            let first = sp[0];
            let near_first = hover.map_or(false, |m| {
                self.tool == Tool::Polygon && sp.len() >= 3 && m.distance(first) < CLOSE_TOL
            });
            if let Some(m) = hover {
                if self.tool == Tool::Polygon && sp.len() >= 2 {
                    painter.extend(egui::Shape::dashed_line(
                        &[m, first],
                        Stroke::new(1.0_f32, cc),
                        6.0,
                        4.0,
                    ));
                }
                sp.push(m);
            }
            painter.add(egui::Shape::line(sp.clone(), Stroke::new(2.0_f32, cc)));
            for p in &self.drawing {
                painter.circle_filled(to_s(*p), 3.5, cc);
            }
            painter.circle_stroke(
                first,
                if near_first { 9.0 } else { 5.0 },
                Stroke::new(2.0_f32, Color32::WHITE),
            );
        }
        // crosshair guide for drawing tools
        if !matches!(self.tool, Tool::Select | Tool::Points) && !space {
            if let Some(m) = hover {
                let img_r = Rect::from_min_max(to_s([0.0, 0.0]), to_s([iw, ih]));
                if img_r.contains(m) {
                    let st = Stroke::new(1.0_f32, Color32::from_white_alpha(90));
                    painter.line_segment([Pos2::new(img_r.left(), m.y), Pos2::new(img_r.right(), m.y)], st);
                    painter.line_segment([Pos2::new(m.x, img_r.top()), Pos2::new(m.x, img_r.bottom())], st);
                }
            }
        }
    }

    /// Magic tool overlay: SAM's answer, the prompt points / box, and readiness.
    fn draw_magic(&self, painter: &egui::Painter, to_s: &dyn Fn([f32; 2]) -> Pos2, cc: Color32, rect: Rect) {
        if let Some(shape) = &self.magic.shape {
            if self.magic.shape_is_box && shape.len() == 2 {
                let r = Rect::from_two_pos(to_s(shape[0]), to_s(shape[1]));
                painter.rect_filled(r, 0.0, cc.gamma_multiply(0.25));
                painter.rect_stroke(r, 0.0, Stroke::new(2.0_f32, cc));
            } else if shape.len() >= 3 {
                let sp: Vec<Pos2> = shape.iter().map(|p| to_s(*p)).collect();
                let mut cache = self.tri_cache.borrow_mut();
                let tri = cache.entry(pts_key(shape)).or_insert_with(|| {
                    let ip: Vec<Pos2> = shape.iter().map(|q| Pos2::new(q[0], q[1])).collect();
                    triangulate(&ip)
                });
                fill_mesh(painter, &sp, cc.gamma_multiply(0.3), tri);
                let mut closed = sp.clone();
                closed.push(sp[0]);
                painter.extend(egui::Shape::dashed_line(&closed, Stroke::new(2.0_f32, cc), 8.0, 4.0));
            }
        }
        if let Some(b) = self.magic.bbox {
            let r = Rect::from_two_pos(to_s([b[0], b[1]]), to_s([b[2], b[3]]));
            painter.extend(egui::Shape::dashed_line(
                &[r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()],
                Stroke::new(1.5_f32, Color32::WHITE),
                6.0,
                4.0,
            ));
        }
        for (p, pos) in &self.magic.points {
            let c = to_s(*p);
            let col = if *pos { Color32::from_rgb(40, 200, 90) } else { Color32::from_rgb(230, 60, 60) };
            painter.circle_filled(c, 7.0, col);
            painter.circle_stroke(c, 7.0, Stroke::new(1.5_f32, Color32::WHITE));
            let st = Stroke::new(2.0_f32, Color32::WHITE);
            painter.line_segment([c - Vec2::new(3.5, 0.0), c + Vec2::new(3.5, 0.0)], st);
            if *pos {
                painter.line_segment([c - Vec2::new(0.0, 3.5), c + Vec2::new(0.0, 3.5)], st);
            }
        }
        // banner: readiness / what to do
        let ready = self.cur_key().is_some_and(|k| self.encoded.contains(&k));
        let failed = self
            .encode_error
            .as_ref()
            .filter(|e| self.cur_key().as_ref() == Some(&e.0))
            .map(|e| e.1.clone());
        let msg = if !self.ai_started || self.ai_status.starts_with("unavailable") {
            format!("Magic tool: AI {}", self.ai_status)
        } else if let Some(err) = failed {
            format!("Magic: AI could not prepare this image: {err}")
        } else if !ready {
            "Magic: AI is preparing this image...".to_string()
        } else if self.magic.is_empty() {
            "Magic: click an object (or drag a box around it)".to_string()
        } else if self.magic.shape.is_some() {
            "Enter / double-click: accept · click: add area · right-click: remove area · Backspace: undo point · Esc: cancel".to_string()
        } else {
            "Magic: segmenting...".to_string()
        };
        let g = painter.layout_no_wrap(msg, FontId::proportional(14.0), Color32::WHITE);
        let pos = Pos2::new(rect.center().x - g.size().x / 2.0, rect.top() + 10.0);
        painter.rect_filled(Rect::from_min_size(pos, g.size()).expand(6.0), 6.0, Color32::from_black_alpha(170));
        painter.galley(pos, g, Color32::WHITE);
    }

    // ------------------------------------------------------------ panels

    fn pick_dataset(&mut self, ctx: &egui::Context) {
        if self.lab() {
            // lab members cannot change the folder; only a browser on the server machine can
            if self.admin {
                self.admin_dialog = true;
            } else {
                self.status = "The dataset is chosen on the lab server.".into();
            }
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(d) = rfd::FileDialog::new().pick_folder() {
            self.open_dataset(ctx, &d.display().to_string());
        }
        let _ = ctx;
    }

    fn open_export(&mut self) {
        let Some(d) = &self.source else { return };
        if self.export.out.is_empty() {
            if self.lab() {
                self.export.out = "yolo_dataset".into();
            } else {
                let d = std::path::Path::new(d);
                let name = d.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "images".into());
                let base = d.parent().unwrap_or(d);
                self.export.out = base.join(format!("{name}_yolo_dataset")).display().to_string();
            }
        }
        self.export.open = true;
    }

    fn top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.add_space(3.0);
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 3.0;
                let has = self.source.is_some();
                // lab: only a browser on the server machine may switch the dataset
                if (!self.lab() || self.admin)
                    && act_button(ui, Act::Open, "Open", true)
                        .on_hover_text(if self.lab() {
                            "Switch the dataset folder for everyone (only on the server machine)"
                        } else {
                            "Open a dataset folder (Ctrl+O)\nAll sub-folders are scanned. You can also drop a folder or image on the window."
                        })
                        .clicked()
                {
                    self.pick_dataset(ctx);
                }
                if act_button(ui, Act::Save, "Save", has)
                    .on_hover_text("Save everything now (Ctrl+S)\nEdits are also saved automatically.")
                    .clicked()
                {
                    self.save_current(true);
                }
                if act_button(ui, Act::Export, "Export", has)
                    .on_hover_text("Export a train/val dataset with data.yaml (Ctrl+E)")
                    .clicked()
                {
                    self.open_export();
                }
                ui.spacing_mut().item_spacing.x = 8.0;
                ui.separator();
                // ---- toolbox
                egui::Frame::group(ui.style())
                    .inner_margin(egui::Margin::same(3.0))
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing.x = 3.0;
                        for (t, label, key, tip) in [
                            (Tool::Select, "Select", "V", "Selector: select, move, resize and delete shapes.\nQ toggles between Selector and the last annotation tool."),
                            (Tool::Points, "Points", "E", "Edit the points of the selected polygon / polyline:\nclick or drag on an edge to add a point (+), right-click a point (red ring) or Backspace to remove it, drag a point to move it.\nDrop a point onto its neighbour to merge them. Click another shape to edit that one."),
                            (Tool::BBox, "Box", "B", "Annotate bounding boxes: drag a rectangle."),
                            (Tool::Polygon, "Polygon", "P", "Annotate instance-segmentation polygons: click points, click the first point (or right/double-click, Enter) to close."),
                            (Tool::Polyline, "Polyline", "L", "Annotate polylines: click points, right/double-click or Enter to finish."),
                            (Tool::Magic, "Magic", "M", "AI segmentation (SAM 2.1): click an object or drag a box around it to get its polygon.\nClick to add an area, right-click to remove one, Enter or double-click to accept.\nThen drag the points to adjust."),
                        ] {
                            if tool_button(ui, self.tool == t, t, label, key)
                                .on_hover_text(tip)
                                .clicked()
                            {
                                self.set_tool(t);
                            }
                        }
                    });
                ui.separator();
                ui.spacing_mut().item_spacing.x = 3.0;
                let has_img = self.loaded.is_some();
                if act_button(ui, Act::Prev, "", has).on_hover_text("Previous image (A or Left arrow)").clicked() {
                    self.goto(ctx, -1);
                }
                if act_button(ui, Act::Next, "", has).on_hover_text("Next image (D or Right arrow)").clicked() {
                    self.goto(ctx, 1);
                }
                if act_button(ui, Act::Fit, "", has_img).on_hover_text("Fit image to window (F)").clicked() {
                    self.fit_pending = true;
                }
                if act_button(ui, Act::Undo, "", has_img).on_hover_text("Undo (Ctrl+Z)").clicked() {
                    self.undo();
                }
                if act_button(ui, Act::Auto, "Auto", has_img)
                    .on_hover_text("Auto-label this image with your YOLO model (Ctrl+L)\nThen adjust the results. Set the model under 'AI assist' in the right panel.")
                    .clicked()
                {
                    self.auto_label();
                }
                ui.spacing_mut().item_spacing.x = 8.0;
                ui.separator();
                let before = self.project.export_mode;
                egui::ComboBox::from_id_salt("mode")
                    .selected_text(match self.project.export_mode {
                        ExportMode::Segment => "Segment",
                        ExportMode::Detect => "Detect",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.project.export_mode,
                            ExportMode::Segment,
                            "Segmentation (polygons)",
                        );
                        ui.selectable_value(
                            &mut self.project.export_mode,
                            ExportMode::Detect,
                            "Detection (boxes)",
                        );
                    })
                    .response
                    .on_hover_text("YOLO label format: Segmentation (polygons) or Detection (boxes)");
                if before != self.project.export_mode {
                    if self.lab() {
                        // applied when the server confirms it (for everyone)
                        let m = std::mem::replace(&mut self.project.export_mode, before);
                        self.send_settings(Some(m), None, None);
                    } else {
                        self.labels_stale = true; // every label file changes format
                        self.touch();
                    }
                }
                ui.label("Line width:")
                    .on_hover_text("Polylines are exported as a thin polygon band of this width (image pixels). 0 = zero-width.");
                let r = ui.add(
                    egui::DragValue::new(&mut self.project.polyline_width)
                        .range(0.0..=500.0)
                        .speed(0.5)
                        .suffix(" px"),
                );
                if self.lab() {
                    // sent once the value is set (not on every step of a drag)
                    if r.drag_stopped() || (r.changed() && !r.dragged()) {
                        let w = self.project.polyline_width;
                        self.send_settings(None, Some(w), None);
                    }
                } else if r.changed() {
                    // only images with polylines are affected
                    for (name, a) in &self.project.images {
                        if a.shapes.iter().any(|s| s.kind == Kind::Polyline) {
                            self.stale.insert(name.clone());
                        }
                    }
                    self.touch();
                }
            });
            ui.add_space(3.0);
        });
    }

    fn about_window(&mut self, ctx: &egui::Context) {
        if !self.about_open {
            return;
        }
        let tex = ctx.data_mut(|d| d.get_temp::<egui::TextureHandle>(egui::Id::new("about_icon")));
        let tex = tex.unwrap_or_else(|| {
            let img = egui::ColorImage::from_rgba_unmultiplied([96, 96], &crate::icon::rgba(96));
            let t = ctx.load_texture("about_icon", img, Default::default());
            ctx.data_mut(|d| d.insert_temp(egui::Id::new("about_icon"), t.clone()));
            t
        });
        egui::Window::new("About")
            .open(&mut self.about_open)
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.add(egui::Image::new((tex.id(), Vec2::splat(96.0))));
                    ui.vertical(|ui| {
                        ui.heading("YOLO Annotator");
                        ui.label(format!("Version {}", env!("CARGO_PKG_VERSION")));
                        ui.label("Box, polygon and polyline annotation\nfor Ultralytics YOLO datasets.");
                        ui.add_space(6.0);
                        ui.strong("© 2026 DotoRobotics");
                        ui.label("Released under the MIT License.");
                    });
                });
                ui.separator();
                egui::CollapsingHeader::new("License text").show(ui, |ui| {
                    egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                        ui.add(egui::Label::new(egui::RichText::new(include_str!("../../../LICENSE")).small().monospace()).wrap());
                    });
                });
                ui.label(
                    egui::RichText::new("Built with Rust, egui/eframe, image and rfd (MIT / Apache-2.0).")
                        .small()
                        .weak(),
                );
            });
    }

    /// Saved / unsaved / failed badge at the start of the status bar.
    fn save_indicator(&self, ui: &mut egui::Ui) {
        let Some(src) = &self.source else { return };
        let pending = if self.lab() { !self.unacked.is_empty() } else { self.labels_stale || !self.stale.is_empty() };
        let (text, color, tip) = if let Some(e) = self.label_error.as_ref().or(self.save_error.as_ref()) {
            ("⚠ Save failed", Color32::from_rgb(235, 90, 80), format!("{e}\nRetried automatically at the next save."))
        } else if self.label_dirty || pending {
            ("● Unsaved", Color32::from_rgb(230, 180, 60), "Changes are being written...".to_string())
        } else if self.lab() {
            ("✔ Saved", Color32::from_rgb(110, 200, 120), format!("Saved on the lab server (dataset {src}).\nLabels are written after every edit."))
        } else {
            let dir = std::path::Path::new(src);
            (
                "✔ Saved",
                Color32::from_rgb(110, 200, 120),
                format!(
                    "YOLO label files are written after every edit.\nLabels: {}\nProject: {}",
                    labels_dir_for(dir).display(),
                    dir.join(PROJECT_FILE).display()
                ),
            )
        };
        ui.label(egui::RichText::new(text).color(color)).on_hover_text(tip);
        ui.separator();
        if self.lab() {
            let (t, c) = if self.connected {
                (format!("🖧 {} · {} online", self.user, self.users.len().max(1)), Color32::from_rgb(110, 200, 120))
            } else {
                ("🖧 offline".to_string(), Color32::from_rgb(235, 90, 80))
            };
            ui.label(egui::RichText::new(t).color(c)).on_hover_text(if self.users.is_empty() {
                "Lab server".to_string()
            } else {
                format!("Working on this dataset:\n{}", self.users.join("\n"))
            });
            ui.separator();
        }
    }

    fn status_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                self.save_indicator(ui);
                if self.source.is_some() && self.project.new_frames != NewFrames::Empty {
                    let t = match self.project.new_frames {
                        NewFrames::CopyPrevious => "⏩ New frames: copy previous",
                        _ => "⏩ New frames: auto-label",
                    };
                    ui.label(egui::RichText::new(t).color(Color32::from_rgb(120, 180, 255)))
                        .on_hover_text("Frames with no labels are pre-filled when opened. Change it under 'Video / frame sequence' (right panel).");
                    ui.separator();
                }
                if let Some(l) = &self.loaded {
                    ui.label(format!(
                        "{}/{}  {}  {}×{}  zoom {:.0}%",
                        self.cur + 1,
                        self.files.len(),
                        l.name,
                        l.w,
                        l.h,
                        self.zoom * 100.0
                    ));
                    if let Some(p) = self.cursor_img {
                        ui.label(format!("({:.0}, {:.0})", p[0], p[1]));
                    }
                    ui.separator();
                }
                let point_sel = self.selected_point().is_some();
                let hint = match self.tool {
                    Tool::Select | Tool::Points if point_sel => "POINT selected (amber) · Del: delete this point · drag: move it · click inside the shape: select the whole shape · Esc: back to the whole shape",
                    Tool::Select => "Click a point: select the point · click inside: select the shape · drag: move · Del: delete the selection · drag point/shape: edit ·  · E: edit points · drag empty: pan",
                    Tool::Points if self.selected.is_none() => "EDIT POINTS · click a polygon or polyline to edit its points · V: back to Select",
                    Tool::Points => "EDIT POINTS · click / drag edge (+): add point · right-click point (red) or Backspace: remove · drag point: move · drop on neighbour: merge · I: add at + · V: back to Select",
                    Tool::BBox => "Drag to draw a box · click a shape: select it · drag its handles / centre grip: resize / move · Del: delete · Esc: deselect",
                    Tool::Polygon => "Click to add points · click 1st point, double-click, Enter, or right-click (adds a last point): close · Backspace: undo point · Esc: cancel · right-click a shape: select it, then drag handles / centre to edit",
                    Tool::Polyline => "Click to add points · double-click, Enter, or right-click (adds a last point): finish · Backspace: undo point · Esc: cancel · right-click a shape: select it, then drag handles / centre to edit",
                    Tool::Magic => "Click an object or drag a box · click: add area · right-click: remove area · Enter / double-click: accept · Backspace: undo point · Esc: cancel",
                };
                ui.weak(hint);
                ui.separator();
                ui.label(&self.status);
            });
        });
    }

    fn file_list(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("files")
            .default_width(260.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.heading("Dataset");
                let Some(root) = self.source.clone() else {
                    ui.weak(if self.lab() {
                        "Waiting for the lab server..."
                    } else {
                        "No folder open.\nUse Open (Ctrl+O) or drag a folder here."
                    });
                    return;
                };
                let root_name =
                    root.replace('\\', "/").rsplit('/').find(|s| !s.is_empty()).unwrap_or(&root).to_string();
                ui.label(egui::RichText::new(format!("Folder: {root_name}")).strong()).on_hover_text(&root);
                // labeled flags once per frame; folder counts from them
                let flags: Vec<bool> =
                    self.files.iter().map(|f| self.project.images.contains_key(f)).collect();
                let mut counts = Vec::new();
                let (labeled, total) = self.tree.counts(&flags, &mut counts);
                ui.label(format!("{labeled} / {total} images labeled"));
                ui.horizontal(|ui| {
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut self.filter)
                            .hint_text("filter by name / folder")
                            .desired_width(150.0),
                    );
                    self.typing |= r.has_focus();
                    if !self.filter.is_empty() && ui.small_button("✖").clicked() {
                        self.filter.clear();
                    }
                });
                let tree_mode = self.filter.trim().is_empty() && !self.unlabeled_only;
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.unlabeled_only, "Unlabeled only");
                    if tree_mode && !self.tree.dirs.is_empty() {
                        if ui.small_button("Expand all").clicked() {
                            self.tree.all_paths(&mut self.expanded);
                        }
                        if ui.small_button("Collapse").clicked() {
                            self.expanded.clear();
                            // keep the current image visible
                            if let Some(f) = self.files.get(self.cur) {
                                self.expanded.extend(ancestors(f).map(String::from));
                            }
                        }
                    }
                });
                ui.separator();

                let scroll = std::mem::take(&mut self.scroll_list);
                let mut goto = None;
                let mut toggle: Option<String> = None;
                let (files, project, cur) = (&self.files, &self.project, self.cur);
                let (locks, me) = (&self.locks, self.user.as_str());
                // every row has the same height, so only the visible ones are laid out
                // (keeps the pane fast with tens of thousands of images)
                let row_h = ui.spacing().interact_size.y;
                let stride = row_h + ui.spacing().item_spacing.y;
                let view_h = ui.available_height();
                if tree_mode {
                    let mut rows = Vec::new();
                    self.tree.flatten(0, &self.expanded, &mut rows);
                    let mut sa = egui::ScrollArea::vertical().auto_shrink([false; 2]);
                    if scroll {
                        if let Some(pos) = rows.iter().position(|r| matches!(r, Row::File(i, _) if *i == cur)) {
                            sa = sa.vertical_scroll_offset((pos as f32 * stride - view_h / 2.0).max(0.0));
                        }
                    }
                    sa.show_rows(ui, row_h, rows.len(), |ui, range| {
                        for row in &rows[range] {
                            match *row {
                                Row::Dir(d, depth) => {
                                    let open = self.expanded.contains(&d.path);
                                    if dir_row(ui, d, depth, open, counts[d.id], row_h) {
                                        toggle = Some(d.path.clone());
                                    }
                                }
                                Row::File(i, depth) => {
                                    ui.horizontal(|ui| {
                                        ui.add_space(depth as f32 * INDENT + 16.0);
                                        file_row(ui, i, files, project, cur, false, locks, me, &mut goto);
                                    });
                                }
                            }
                        }
                    });
                } else {
                    // flat filtered list
                    let f = self.filter.trim().to_lowercase();
                    let idx: Vec<usize> = (0..files.len())
                        .filter(|&i| {
                            (f.is_empty() || files[i].to_lowercase().contains(&f))
                                && (!self.unlabeled_only || !flags[i])
                        })
                        .collect();
                    ui.weak(format!("{} matching", idx.len()));
                    let mut sa = egui::ScrollArea::vertical().auto_shrink([false; 2]);
                    if scroll {
                        if let Some(pos) = idx.iter().position(|&i| i == cur) {
                            sa = sa.vertical_scroll_offset((pos as f32 * stride - view_h / 2.0).max(0.0));
                        }
                    }
                    sa.show_rows(ui, row_h, idx.len(), |ui, range| {
                        for k in range {
                            file_row(ui, idx[k], files, project, cur, true, locks, me, &mut goto);
                        }
                    });
                }
                if let Some(path) = toggle {
                    if !self.expanded.remove(&path) {
                        self.expanded.insert(path);
                    }
                }
                if let Some(i) = goto {
                    self.load_image(ctx, i);
                    self.scroll_list = false;
                }
            });
    }

    fn right_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("right")
            .default_width(270.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
                    ui.heading("Classes");
                    ui.weak("Click or press 1-9. With a shape selected, this changes its class.");
                    let mut pick = None;
                    for (i, name) in self.project.classes.iter().enumerate() {
                        ui.horizontal(|ui| {
                            let (r, _) = ui.allocate_exact_size(Vec2::splat(14.0), Sense::hover());
                            ui.painter().rect_filled(r, 3.0, class_color(i));
                            let key = if i < 9 { format!("[{}] ", i + 1) } else { String::new() };
                            if ui
                                .selectable_label(self.class == i, format!("{key}{i}: {name}"))
                                .clicked()
                            {
                                pick = Some(i);
                            }
                        });
                    }
                    if let Some(i) = pick {
                        self.set_class(i);
                    }
                    ui.horizontal(|ui| {
                        let r = ui.add(
                            egui::TextEdit::singleline(&mut self.new_class)
                                .hint_text("new class name")
                                .desired_width(150.0),
                        );
                        self.typing |= r.has_focus();
                        let enter = r.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
                        if (ui.button("Add").clicked() || enter) && !self.new_class.trim().is_empty() {
                            let name = self.new_class.trim().to_string();
                            if self.lab() {
                                // added by the server for everyone, in one order
                                self.store.op(ClassOp::Add(name));
                            } else {
                                self.project.classes.push(name);
                                self.class = self.project.classes.len() - 1;
                                self.touch();
                            }
                            self.new_class.clear();
                        }
                    });
                    if self.class < self.project.classes.len() && self.lab() {
                        ui.horizontal(|ui| {
                            ui.label("Rename:");
                            let c = self.class;
                            let mut text = self.rename_edit.clone().unwrap_or_else(|| self.project.classes[c].clone());
                            let r = ui.add(egui::TextEdit::singleline(&mut text).desired_width(120.0));
                            self.typing |= r.has_focus();
                            if r.has_focus() {
                                self.rename_edit = Some(text);
                            } else if r.lost_focus() {
                                // sent when done typing; applied when the server confirms it
                                if text.trim() != self.project.classes[c] {
                                    self.store.op(ClassOp::Rename(c, text));
                                }
                                self.rename_edit = None;
                            }
                            if ui.button("Delete").clicked() {
                                self.store.op(ClassOp::Delete(c));
                            }
                        });
                    } else if self.class < self.project.classes.len() {
                        ui.horizontal(|ui| {
                            ui.label("Rename:");
                            let r = ui.add(
                                egui::TextEdit::singleline(&mut self.project.classes[self.class])
                                    .desired_width(120.0),
                            );
                            self.typing |= r.has_focus();
                            if r.changed() {
                                self.dirty = true;
                            }
                            // an empty name would break classes.txt / data.yaml
                            if r.lost_focus() && self.project.classes[self.class].trim().is_empty() {
                                self.project.classes[self.class] = format!("class_{}", self.class);
                            }
                            if ui.button("Delete").clicked() {
                                let c = self.class;
                                let used: usize = self
                                    .project
                                    .images
                                    .values()
                                    .map(|a| a.shapes.iter().filter(|s| s.class == c).count())
                                    .sum();
                                if used > 0 {
                                    self.status = format!(
                                        "Class '{}' is used by {used} objects; reassign or delete them first.",
                                        self.project.classes[c]
                                    );
                                } else {
                                    self.project.classes.remove(c);
                                    for (name, a) in self.project.images.iter_mut() {
                                        let mut changed = false;
                                        for s in &mut a.shapes {
                                            if s.class > c {
                                                s.class -= 1;
                                                changed = true;
                                            }
                                        }
                                        if changed {
                                            self.stale.insert(name.clone());
                                        }
                                    }
                                    self.class = c.saturating_sub(1);
                                    self.undo.clear();
                                    self.touch();
                                }
                            }
                        });
                    }

                    ui.separator();
                    ui.heading("Objects in this image");
                    let mut del = None;
                    let mut sel = None;
                    for (i, s) in self.shapes().iter().enumerate() {
                        ui.horizontal(|ui| {
                            let (r, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
                            ui.painter().rect_filled(r, 2.0, class_color(s.class));
                            let cname = self
                                .project
                                .classes
                                .get(s.class)
                                .map(|s| s.as_str())
                                .unwrap_or("?");
                            if ui
                                .selectable_label(
                                    self.selected == Some(i),
                                    format!("#{i} {} – {cname}", s.kind.name()),
                                )
                                .clicked()
                            {
                                sel = Some(i);
                            }
                            if ui.small_button("✖").on_hover_text("Delete").clicked() {
                                del = Some(i);
                            }
                        });
                    }
                    if let Some(i) = sel {
                        self.selected = Some(i);
                    }
                    if let Some(i) = del {
                        self.delete_shape(i);
                    }
                    if self.loaded.is_some() {
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            if ui
                                .button("Mark as background")
                                .on_hover_text("Image with no objects: writes an empty label file (useful negative example).")
                                .clicked()
                            {
                                self.shapes_mut();
                                self.touch();
                            }
                            if ui.button("Clear all").clicked() && !self.shapes().is_empty() {
                                self.push_undo();
                                if let Some(v) = self.shapes_mut() {
                                    v.clear();
                                }
                                self.selected = None;
                                self.touch();
                            }
                        });
                    }
                    ui.checkbox(&mut self.show_labels, "Show class names on image");

                    ui.separator();
                    self.video_panel(ui);
                    ui.separator();
                    self.ai_panel(ui);

                    ui.separator();
                    egui::CollapsingHeader::new("Help").default_open(false).show(ui, |ui| {
                        if ui.button("ⓘ About YOLO Annotator").clicked() {
                            self.about_open = true;
                        }
                        ui.label(
                            "Mouse\n\
                             • Wheel: zoom   • Middle-drag / Space+drag: pan\n\
                             • Box: drag a rectangle\n\
                             • Polygon: click points, click the first point (or double/right-click) to close\n\
                             • Polyline: click points, double/right-click to finish\n\
                             • Select: click inside a shape to select it, or click a point to select just that point (amber); Del deletes the point if one is selected, otherwise the whole shape\n\
                             • Points (E): edit the points of the selected polygon / polyline: click or drag an edge to add a point, right-click a point (or hover + Backspace) to remove it, drop a point onto its neighbour to merge the two\n\
                             • Video frames: under 'Video / frame sequence' (right panel), new frames can be pre-filled with the previous frame's labels or by your YOLO model; C copies the previous frame's labels. Export: 'Split by folder' keeps each video in train or in val\n\
                             • In the drawing tools the shape you just drew stays selected: drag its handles (box corners or edge midpoints) or its centre grip to adjust. Box tool: click a shape to select it; Polygon / Polyline: right-click a shape. Del deletes the selection, Esc deselects\n\n\
                             Keys\n\
                             V/B/P/L tools · Q toggle Selector <-> last annotation tool · A/D or Left/Right arrows prev/next image · 1-9 class · Del delete · Ctrl+Z undo · Ctrl+O open · Ctrl+S save · Ctrl+E export · F fit · Esc cancel\n\n\
                             Saving\n\
                             Everything autosaves to annotations.json in the image folder, and YOLO .txt files are written to the 'labels' folder next to it (images -> labels). Use 'Export...' to build a train/val split with data.yaml.",
                        );
                    });
                });
            });
    }

    fn video_panel(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Video / frame sequence").default_open(true).show(ui, |ui| {
            ui.label("When a frame with no labels is opened:");
            let before = self.project.new_frames;
            egui::ComboBox::from_id_salt("new_frames")
                .selected_text(before.label())
                .width(220.0)
                .show_ui(ui, |ui| {
                    for m in [NewFrames::Empty, NewFrames::CopyPrevious, NewFrames::AutoLabel] {
                        ui.selectable_value(&mut self.project.new_frames, m, m.label());
                    }
                })
                .response
                .on_hover_text(
                    "Copy: takes the labels of the frame you came from (or the nearest labeled frame in the same folder). Fix them with Points (E).\n\
                     Auto-label: runs your YOLO model (set under AI assist); class names come from the model.\n\
                     Only frames that were never labeled are filled. Ctrl+Z undoes. Saved with this dataset.",
                );
            if self.project.new_frames != before {
                self.dirty = true;
            }
            if ui
                .add_enabled(self.loaded.is_some(), egui::Button::new("Copy previous frame (C)"))
                .on_hover_text("Replace this frame's labels with those of the nearest labeled frame in the same folder (before it, else after it). Ctrl+Z undoes.")
                .clicked()
            {
                self.copy_previous();
            }
        });
    }

    fn ai_panel(&mut self, ui: &mut egui::Ui) {
        let lab = self.lab();
        egui::CollapsingHeader::new("AI assist").default_open(true).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("Status:");
                let mut s = self.ai_status.clone();
                if self.last_encode_ms > 0 && self.ai_started {
                    s.push_str(&format!(" · {} ms/image", self.last_encode_ms));
                }
                ui.weak(s);
            });
            if lab {
                ui.weak("The AI runs on the lab server; its settings apply to everyone.");
            }
            ui.label(egui::RichText::new("Magic tool (M) · SAM 2.1").strong());
            let before = self.ai_settings.clone();
            egui::Grid::new("ai_grid").num_columns(2).spacing([6.0, 4.0]).show(ui, |ui| {
                ui.label("Model:");
                let show = |m: &str| {
                    if m == "auto" {
                        "Auto (by hardware)".to_string()
                    } else {
                        m.replace("sam2.1-", "SAM 2.1 ")
                    }
                };
                egui::ComboBox::from_id_salt("sam_model")
                    .selected_text(show(&self.ai_settings.sam))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.ai_settings.sam, "auto".to_string(), show("auto"));
                        for m in &self.ai_models {
                            ui.selectable_value(&mut self.ai_settings.sam, m.clone(), show(m));
                        }
                    })
                    .response
                    .on_hover_text("tiny: fastest · small: balanced · base-plus: best edges (GPU recommended)");
                ui.end_row();
                if !lab {
                    ui.label("Engine:");
                    egui::ComboBox::from_id_salt("ai_engine")
                        .selected_text(self.ai_settings.engine.label())
                        .show_ui(ui, |ui| {
                            for e in [Engine::Auto, Engine::Cuda, Engine::DirectML, Engine::Cpu] {
                                ui.selectable_value(&mut self.ai_settings.engine, e, e.label());
                            }
                        })
                        .response
                        .on_hover_text("Auto: CUDA if an NVIDIA GPU and the CUDA pack are present, else DirectML (any GPU), else CPU.\nA change takes effect after restarting the app.");
                    ui.end_row();
                }
                ui.label("Detail:");
                ui.add(egui::Slider::new(&mut self.ai_settings.detail, 0.3..=4.0).text("px"))
                    .on_hover_text("Polygon simplification: smaller = more points, follows the edge closer");
                ui.end_row();
            });
            if self.ai_models.is_empty() && self.ai_started && (!lab || !self.user.is_empty() && self.connected) {
                ui.colored_label(
                    Color32::from_rgb(230, 160, 60),
                    if lab { "No SAM models on the lab server." } else { "No SAM models found (models folder next to the exe)." },
                );
            }

            ui.add_space(4.0);
            ui.label(egui::RichText::new("Auto-label · your YOLO model").strong());
            ui.horizontal(|ui| {
                let name = self.ai_settings.yolo.replace('\\', "/").rsplit('/').next().unwrap_or("").to_string();
                let name = if name.is_empty() { "none".to_string() } else { name };
                if lab {
                    // models are files in the server's models folder; no paths from the browser
                    if !self.ai_started {
                        self.ensure_ai(); // fetches the model list
                    }
                    egui::ComboBox::from_id_salt("yolo_model").selected_text(name).show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.ai_settings.yolo, String::new(), "none");
                        for m in &self.yolo_models {
                            ui.selectable_value(&mut self.ai_settings.yolo, m.clone(), m);
                        }
                    });
                } else {
                    ui.label(name).on_hover_text(&self.ai_settings.yolo);
                    #[cfg(not(target_arch = "wasm32"))]
                    if ui.small_button("Browse...").on_hover_text("An Ultralytics model exported with: yolo export model=best.pt format=onnx").clicked() {
                        if let Some(f) = rfd::FileDialog::new().add_filter("ONNX model", &["onnx"]).pick_file() {
                            self.ai_settings.yolo = f.display().to_string();
                        }
                    }
                    if !self.ai_settings.yolo.is_empty() && ui.small_button("✖").clicked() {
                        self.ai_settings.yolo.clear();
                    }
                }
            });
            if lab && self.yolo_models.is_empty() {
                ui.weak("Put YOLO .onnx models into the server's models folder to use them here.");
            }
            ui.add(egui::Slider::new(&mut self.ai_settings.conf, 0.05..=0.95).text("confidence"));
            if self.ai_settings != before {
                let sam_changed = self.ai_settings.sam != before.sam;
                self.ai_settings_changed(sam_changed);
            }
            let has_model = !self.ai_settings.yolo.is_empty();
            let editable = self.locked_by_other().is_none();
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(has_model && self.loaded.is_some() && editable, egui::Button::new("Auto-label image"))
                    .on_hover_text("Ctrl+L. Adds the detections to this image; Ctrl+Z undoes.")
                    .clicked()
                {
                    self.auto_label();
                }
                let unl = self.files.iter().filter(|f| !self.project.images.contains_key(*f)).count();
                if self.batch.is_none()
                    && ui
                        .add_enabled(has_model && unl > 0, egui::Button::new(format!("All unlabeled ({unl})")))
                        .on_hover_text("Runs the model on every unlabeled image in the background.\nImages with no detections stay unlabeled.")
                        .clicked()
                {
                    self.auto_label_all();
                }
            });
            if let Some(b) = &self.batch {
                let (d, id) = (b.done, b.id);
                let added = if lab { String::new() } else { format!(" · {} labeled", b.added) };
                ui.add(egui::ProgressBar::new(d as f32 / b.total.max(1) as f32).text(format!("{d} / {}{added}", b.total)));
                if ui.button("Cancel").clicked() {
                    self.store.ai(AiReq::CancelBatch { id });
                }
                self.ctx.request_repaint_after(std::time::Duration::from_millis(200));
            }
        });
    }

    fn export_window(&mut self, ctx: &egui::Context) {
        if !self.export.open {
            return;
        }
        let lab = self.lab();
        let mut open = true;
        let mut run = false;
        egui::Window::new("Export dataset")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label("Creates an Ultralytics-ready dataset:");
                ui.monospace("out/images/{train,val}  out/labels/{train,val}  out/data.yaml");
                ui.horizontal(|ui| {
                    ui.label(if lab { "Export name:" } else { "Output folder:" });
                    let r = ui.add(egui::TextEdit::singleline(&mut self.export.out).desired_width(320.0));
                    self.typing |= r.has_focus();
                    #[cfg(not(target_arch = "wasm32"))]
                    if !lab && ui.button("Browse...").clicked() {
                        if let Some(d) = rfd::FileDialog::new().pick_folder() {
                            self.export.out = d.display().to_string();
                        }
                    }
                });
                if lab {
                    ui.weak("Written into the exports folder on the lab server.");
                }
                let has_split = self.files.iter().any(|f| folder_split(f).is_some());
                ui.add_enabled(
                    has_split,
                    egui::Checkbox::new(
                        &mut self.export.keep_split,
                        "Keep existing train / val / test folders",
                    ),
                )
                .on_hover_text("Images inside folders named train, val/valid or test keep that split. Other images are split randomly.");
                ui.checkbox(&mut self.export.by_folder, "Split by folder (keep each video / sequence together)")
                    .on_hover_text("For video frames: whole folders go to train or to val, so almost identical neighbouring frames are never on both sides (that would make the val scores look too good).\nWith a single folder, its last frames become val.\nPut each video's frames in its own folder.");
                ui.add(
                    egui::Slider::new(&mut self.export.val_ratio, 0.0..=0.5)
                        .text("validation fraction"),
                );
                ui.checkbox(
                    &mut self.export.include_unlabeled,
                    "Also include unlabeled images (as background)",
                );
                ui.label(format!(
                    "Format: {}",
                    match self.project.export_mode {
                        ExportMode::Segment => "segmentation (boxes written as 4-point polygons)",
                        ExportMode::Detect => "detection (polygons written as their bounding boxes)",
                    }
                ));
                if let Some((d, t)) = self.export_progress {
                    let frac = if t > 0 { d as f32 / t as f32 } else { 0.0 };
                    ui.add(egui::ProgressBar::new(frac).text(format!("copying {d} / {t}")));
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                } else if ui.button("Export").clicked() {
                    run = true;
                }
            });
        if run && self.export_progress.is_none() && self.source.is_some() {
            self.save_current(true);
            self.store.export(ExportOpts {
                out: self.export.out.trim().to_string(),
                val_ratio: self.export.val_ratio,
                include_unlabeled: self.export.include_unlabeled,
                keep_split: self.export.keep_split,
                by_folder: self.export.by_folder,
            });
            self.export_progress = Some((0, 0));
            self.status = "Exporting...".into();
        }
        self.export.open = open && self.export.open;
    }
}

const INDENT: f32 = 14.0;

/// `v.clamp(lo, hi)` that never panics (lo > hi happens for shapes larger than the image).
fn safe_clamp(v: f32, lo: f32, hi: f32) -> f32 {
    if !(v.is_finite() && lo.is_finite() && hi.is_finite()) || lo > hi {
        0.0
    } else {
        v.clamp(lo, hi)
    }
}

/// Toolbox button: drawn icon + name + shortcut key; highlighted when active.
fn tool_button(ui: &mut egui::Ui, active: bool, tool: Tool, label: &str, key: &str) -> egui::Response {
    let vis = ui.visuals().clone();
    let fg = if active { vis.selection.stroke.color } else { vis.text_color() };
    let weak = vis.weak_text_color();
    let g = ui.painter().layout_no_wrap(label.to_owned(), FontId::proportional(14.0), fg);
    let kg = ui.painter().layout_no_wrap(key.to_owned(), FontId::monospace(11.0), weak);
    let key_box = kg.size() + Vec2::new(8.0, 2.0);
    let w = 6.0 + 22.0 + 6.0 + g.size().x + 8.0 + key_box.x + 6.0;
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, 30.0), Sense::click());
    let p = ui.painter();
    let bg = if active {
        vis.selection.bg_fill
    } else if resp.hovered() {
        vis.widgets.hovered.weak_bg_fill
    } else {
        Color32::TRANSPARENT
    };
    p.rect_filled(rect, 5.0, bg);
    if active {
        p.rect_stroke(rect, 5.0, Stroke::new(1.5_f32, vis.selection.stroke.color));
    }
    let icon = Rect::from_min_size(rect.min + Vec2::new(6.0, 4.0), Vec2::splat(22.0));
    draw_tool_icon(p, icon, tool, fg);
    let tx = icon.max.x + 6.0;
    p.galley(Pos2::new(tx, rect.center().y - g.size().y / 2.0), g.clone(), fg);
    let kr = Rect::from_min_size(
        Pos2::new(tx + g.size().x + 8.0, rect.center().y - key_box.y / 2.0),
        key_box,
    );
    p.rect_stroke(kr, 3.0, Stroke::new(1.0_f32, weak));
    p.galley(kr.min + Vec2::new(4.0, 1.0), kg, weak);
    resp
}

fn draw_tool_icon(p: &egui::Painter, r: Rect, tool: Tool, c: Color32) {
    let pt = |x: f32, y: f32| r.min + Vec2::new(x * r.width(), y * r.height());
    let st = Stroke::new(1.8_f32, c);
    match tool {
        Tool::Points => {
            // polygon outline with square nodes and a "+"
            let v = vec![pt(0.08, 0.55), pt(0.36, 0.14), pt(0.78, 0.3), pt(0.66, 0.9)];
            p.add(egui::Shape::closed_line(v.clone(), st));
            for q in v {
                p.rect_filled(Rect::from_center_size(q, Vec2::splat(5.0)), 0.0, c);
            }
            let k = pt(0.86, 0.72);
            let d = r.width() * 0.14;
            let s2 = Stroke::new(2.0_f32, c);
            p.line_segment([k - Vec2::X * d, k + Vec2::X * d], s2);
            p.line_segment([k - Vec2::Y * d, k + Vec2::Y * d], s2);
        }
        Tool::Select => {
            let arrow = [
                pt(0.25, 0.06),
                pt(0.25, 0.84),
                pt(0.44, 0.66),
                pt(0.57, 0.96),
                pt(0.70, 0.90),
                pt(0.57, 0.61),
                pt(0.82, 0.61),
            ];
            fill_polygon(p, &arrow, c);
        }
        Tool::BBox => {
            let rr = Rect::from_min_max(pt(0.1, 0.2), pt(0.9, 0.8));
            p.rect_stroke(rr, 1.0, st);
            for q in [rr.left_top(), rr.right_top(), rr.right_bottom(), rr.left_bottom()] {
                p.circle_filled(q, 2.4, c);
            }
        }
        Tool::Polygon => {
            let v = vec![pt(0.12, 0.5), pt(0.42, 0.1), pt(0.9, 0.3), pt(0.8, 0.88), pt(0.3, 0.84)];
            p.add(egui::Shape::closed_line(v.clone(), st));
            for q in v {
                p.circle_filled(q, 2.4, c);
            }
        }
        Tool::Polyline => {
            let v = vec![pt(0.06, 0.8), pt(0.34, 0.32), pt(0.62, 0.64), pt(0.94, 0.18)];
            p.add(egui::Shape::line(v.clone(), st));
            for q in v {
                p.circle_filled(q, 2.4, c);
            }
        }
        Tool::Magic => {
            // wand + sparkles
            p.line_segment([pt(0.1, 0.92), pt(0.6, 0.42)], Stroke::new(2.6_f32, c));
            sparkle(p, pt(0.7, 0.3), r.width() * 0.2, c);
            sparkle(p, pt(0.3, 0.2), r.width() * 0.11, c);
            sparkle(p, pt(0.88, 0.72), r.width() * 0.1, c);
        }
    }
}

/// Four-pointed star.
fn sparkle(p: &egui::Painter, c: Pos2, s: f32, col: Color32) {
    let k = s * 0.28;
    let v = vec![
        c + Vec2::new(0.0, -s),
        c + Vec2::new(k, -k),
        c + Vec2::new(s, 0.0),
        c + Vec2::new(k, k),
        c + Vec2::new(0.0, s),
        c + Vec2::new(-k, k),
        c + Vec2::new(-s, 0.0),
        c + Vec2::new(-k, -k),
    ];
    fill_polygon(p, &v, col);
}

#[derive(Clone, Copy)]
enum Act {
    Open,
    Save,
    Export,
    Prev,
    Next,
    Fit,
    Undo,
    Auto,
}

/// Top-bar action button: drawn icon (+ optional label). Shortcuts are in the tooltip.
fn act_button(ui: &mut egui::Ui, act: Act, label: &str, enabled: bool) -> egui::Response {
    let vis = ui.visuals().clone();
    let fg = if enabled { vis.text_color() } else { vis.weak_text_color().gamma_multiply(0.6) };
    let g = (!label.is_empty())
        .then(|| ui.painter().layout_no_wrap(label.to_owned(), FontId::proportional(14.0), fg));
    let w = 6.0 + 22.0 + g.as_ref().map_or(0.0, |g| 6.0 + g.size().x + 2.0) + 6.0;
    let sense = if enabled { Sense::click() } else { Sense::hover() };
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, 30.0), sense);
    let p = ui.painter();
    if enabled && resp.hovered() {
        let bg = if resp.is_pointer_button_down_on() {
            vis.widgets.active.weak_bg_fill
        } else {
            vis.widgets.hovered.weak_bg_fill
        };
        p.rect_filled(rect, 5.0, bg);
    }
    let icon = Rect::from_min_size(rect.min + Vec2::new(6.0, 4.0), Vec2::splat(22.0));
    draw_act_icon(p, icon, act, fg);
    if let Some(g) = g {
        p.galley(Pos2::new(icon.max.x + 6.0, rect.center().y - g.size().y / 2.0), g, fg);
    }
    if enabled {
        resp.on_hover_cursor(CursorIcon::PointingHand)
    } else {
        resp
    }
}

fn draw_act_icon(p: &egui::Painter, r: Rect, act: Act, c: Color32) {
    let pt = |x: f32, y: f32| r.min + Vec2::new(x * r.width(), y * r.height());
    let st = Stroke::new(1.8_f32, c);
    let line = |pts: Vec<Pos2>| p.add(egui::Shape::line(pts, st));
    match act {
        Act::Open => {
            // folder
            line(vec![
                pt(0.08, 0.82), pt(0.08, 0.2), pt(0.38, 0.2), pt(0.48, 0.32), pt(0.84, 0.32),
                pt(0.84, 0.44),
            ]);
            p.add(egui::Shape::closed_line(
                vec![pt(0.08, 0.82), pt(0.24, 0.44), pt(0.96, 0.44), pt(0.8, 0.82)],
                st,
            ));
        }
        Act::Save => {
            // floppy disk
            p.add(egui::Shape::closed_line(
                vec![pt(0.12, 0.1), pt(0.74, 0.1), pt(0.9, 0.26), pt(0.9, 0.9), pt(0.12, 0.9)],
                st,
            ));
            p.rect_stroke(Rect::from_min_max(pt(0.3, 0.1), pt(0.66, 0.34)), 0.0, st);
            p.rect_filled(Rect::from_min_max(pt(0.27, 0.56), pt(0.75, 0.9)), 1.0, c);
        }
        Act::Export => {
            // tray with an arrow leaving it
            line(vec![pt(0.1, 0.5), pt(0.1, 0.9), pt(0.9, 0.9), pt(0.9, 0.5)]);
            line(vec![pt(0.5, 0.68), pt(0.5, 0.08)]);
            line(vec![pt(0.28, 0.3), pt(0.5, 0.08), pt(0.72, 0.3)]);
        }
        Act::Prev | Act::Next => {
            let (a, b) = if matches!(act, Act::Prev) { (0.68, 0.3) } else { (0.32, 0.7) };
            p.add(egui::Shape::convex_polygon(
                vec![pt(a, 0.18), pt(b, 0.5), pt(a, 0.82)],
                c,
                Stroke::NONE,
            ));
        }
        Act::Fit => {
            // four corner brackets
            for (x, y, dx, dy) in [
                (0.1, 0.1, 1.0, 1.0),
                (0.9, 0.1, -1.0, 1.0),
                (0.9, 0.9, -1.0, -1.0),
                (0.1, 0.9, 1.0, -1.0),
            ] {
                line(vec![pt(x, y + 0.28 * dy), pt(x, y), pt(x + 0.28 * dx, y)]);
            }
            p.rect_filled(Rect::from_min_max(pt(0.36, 0.36), pt(0.64, 0.64)), 1.0, c);
        }
        Act::Undo => {
            // arc over the top, arrow head on the left pointing down-left (↶)
            let ctr = pt(0.52, 0.58);
            let rad = 0.32 * r.width();
            let at = |deg: f32| {
                let a = deg.to_radians();
                ctr + Vec2::new(a.cos(), -a.sin()) * rad
            };
            line((0..=16).map(|i| at(165.0 - 230.0 * i as f32 / 16.0)).collect());
            let a = 165f32.to_radians();
            let tan = Vec2::new(-a.sin(), -a.cos()); // continuing past the start
            let nrm = Vec2::new(-tan.y, tan.x);
            let s = at(165.0);
            p.add(egui::Shape::convex_polygon(
                vec![s + tan * 5.0, s + nrm * 4.5, s - nrm * 4.5],
                c,
                Stroke::NONE,
            ));
        }
        Act::Auto => {
            // dashed box with sparkles: "boxes appear by themselves"
            let b = Rect::from_min_max(pt(0.08, 0.3), pt(0.72, 0.92));
            p.extend(egui::Shape::dashed_line(
                &[b.left_top(), b.right_top(), b.right_bottom(), b.left_bottom(), b.left_top()],
                Stroke::new(1.6_f32, c),
                3.0,
                2.5,
            ));
            sparkle(p, pt(0.74, 0.26), r.width() * 0.22, c);
        }
    }
}

/// One image row in the navigation pane.
#[allow(clippy::too_many_arguments)]
fn file_row(
    ui: &mut egui::Ui,
    i: usize,
    files: &[String],
    project: &Project,
    cur: usize,
    full_path: bool,
    locks: &BTreeMap<String, String>,
    me: &str,
    goto: &mut Option<usize>,
) {
    let rel = &files[i];
    let name = if full_path { rel.as_str() } else { rel.rsplit('/').next().unwrap_or(rel) };
    let mut text = match project.images.get(rel) {
        Some(a) if a.shapes.is_empty() => format!("✔ {name}  (bg)"),
        Some(a) => format!("✔ {name}  ({})", a.shapes.len()),
        None => format!("    {name}"),
    };
    // lab: someone else is editing this image
    if let Some(u) = locks.get(rel).filter(|u| u.as_str() != me) {
        text.push_str(&format!("  🔒 {u}"));
    }
    if ui.selectable_label(i == cur, text).on_hover_text(rel).clicked() {
        *goto = Some(i);
    }
}

/// A folder row: expand/collapse triangle, name, labeled/total. Returns true when clicked.
fn dir_row(
    ui: &mut egui::Ui,
    d: &DirNode,
    depth: usize,
    open: bool,
    (l, t): (usize, usize),
    row_h: f32,
) -> bool {
    ui.horizontal(|ui| {
        ui.add_space(depth as f32 * INDENT);
        let mark = if l == t && t > 0 { "✔ " } else { "" };
        let color = ui.visuals().strong_text_color();
        let font = egui::TextStyle::Button.resolve(ui.style());
        let g = ui.painter().layout_no_wrap(format!("{mark}{}    {l}/{t}", d.name), font, color);
        let (rect, resp) =
            ui.allocate_exact_size(Vec2::new(18.0 + g.size().x + 6.0, row_h), Sense::click());
        let p = ui.painter();
        if resp.hovered() {
            p.rect_filled(rect, 3.0, ui.visuals().widgets.hovered.weak_bg_fill);
        }
        let c = rect.left_center() + Vec2::new(8.0, 0.0);
        let tri = if open {
            vec![c + Vec2::new(-4.0, -2.0), c + Vec2::new(4.0, -2.0), c + Vec2::new(0.0, 3.0)]
        } else {
            vec![c + Vec2::new(-2.0, -4.0), c + Vec2::new(3.0, 0.0), c + Vec2::new(-2.0, 4.0)]
        };
        p.add(egui::Shape::convex_polygon(tri, color, Stroke::NONE));
        p.galley(Pos2::new(rect.min.x + 18.0, rect.center().y - g.size().y / 2.0), g, color);
        resp.on_hover_text(&d.path).clicked()
    })
    .inner
}

fn nearest_handle(s: &Annot, p: [f32; 2], tol: f32) -> Option<usize> {
    s.handles()
        .iter()
        .enumerate()
        .map(|(i, h)| (i, dist(*h, p)))
        .filter(|(_, d)| *d <= tol)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}

/// Point is inside the shape or within `tol` of its outline.
fn shape_contains(s: &Annot, p: [f32; 2], tol: f32) -> bool {
    match s.kind {
        Kind::BBox => {
            let b = s.bounds();
            p[0] >= b[0] - tol && p[0] <= b[2] + tol && p[1] >= b[1] - tol && p[1] <= b[3] + tol
        }
        Kind::Polygon => {
            point_in_polygon(p, &s.pts)
                || edges(s).iter().any(|e| dist_to_segment(p, e.1, e.2) <= tol)
        }
        Kind::Polyline => edges(s).iter().any(|e| dist_to_segment(p, e.1, e.2) <= tol),
    }
}

/// Ear-clipping triangulation so concave polygons can be filled correctly.
fn triangulate(pts: &[Pos2]) -> Vec<u32> {
    let n = pts.len();
    if n < 3 {
        return vec![];
    }
    let area: f32 = (0..n)
        .map(|i| {
            let (a, b) = (pts[i], pts[(i + 1) % n]);
            a.x * b.y - b.x * a.y
        })
        .sum();
    let ccw = area > 0.0;
    let cross = |a: Pos2, b: Pos2, c: Pos2| (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    let mut idx: Vec<usize> = (0..n).collect();
    let mut out = Vec::with_capacity((n - 2) * 3);
    while idx.len() > 3 {
        let m = idx.len();
        let mut found = false;
        for i in 0..m {
            let (a, b, c) = (idx[(i + m - 1) % m], idx[i], idx[(i + 1) % m]);
            let (pa, pb, pc) = (pts[a], pts[b], pts[c]);
            let cr = cross(pa, pb, pc);
            if cr == 0.0 || (cr > 0.0) != ccw {
                continue;
            }
            let contains = idx.iter().any(|&j| {
                if j == a || j == b || j == c {
                    return false;
                }
                let p = pts[j];
                let d1 = cross(pa, pb, p);
                let d2 = cross(pb, pc, p);
                let d3 = cross(pc, pa, p);
                if ccw {
                    d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0
                } else {
                    d1 <= 0.0 && d2 <= 0.0 && d3 <= 0.0
                }
            });
            if contains {
                continue;
            }
            out.extend([a as u32, b as u32, c as u32]);
            idx.remove(i);
            found = true;
            break;
        }
        if !found {
            break; // degenerate / self-intersecting: fill what we have
        }
    }
    if idx.len() == 3 {
        out.extend(idx.iter().map(|i| *i as u32));
    }
    out
}

fn fill_polygon(p: &egui::Painter, pts: &[Pos2], color: Color32) {
    fill_mesh(p, pts, color, &triangulate(pts));
}

fn fill_mesh(p: &egui::Painter, pts: &[Pos2], color: Color32, indices: &[u32]) {
    if indices.is_empty() || indices.iter().any(|i| *i as usize >= pts.len()) {
        return;
    }
    let mut mesh = egui::Mesh::default();
    for q in pts {
        mesh.colored_vertex(*q, color);
    }
    mesh.indices = indices.to_vec();
    p.add(egui::Shape::mesh(mesh));
}

fn pts_key(pts: &[[f32; 2]]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for q in pts {
        q[0].to_bits().hash(&mut h);
        q[1].to_bits().hash(&mut h);
    }
    h.finish()
}

fn draw_annot(
    p: &egui::Painter,
    s: &Annot,
    selected: bool,
    to_s: &impl Fn([f32; 2]) -> Pos2,
    label: &str,
    show_label: bool,
    polyline_band: Option<f32>,
    tri: &[u32],
) {
    if s.pts.is_empty() {
        return;
    }
    let c = class_color(s.class);
    let w: f32 = if selected { 3.0 } else { 2.0 };
    let stroke = Stroke::new(w, c);
    let fill = c.gamma_multiply(if selected { 0.35 } else { 0.18 });
    let sp: Vec<Pos2> = s.pts.iter().map(|q| to_s(*q)).collect();
    match s.kind {
        Kind::BBox => {
            if sp.len() == 2 {
                let r = Rect::from_two_pos(sp[0], sp[1]);
                p.rect_filled(r, 0.0, fill);
                p.rect_stroke(r, 0.0, stroke);
            }
        }
        Kind::Polygon => {
            fill_mesh(p, &sp, fill, tri);
            p.add(egui::Shape::closed_line(sp.clone(), stroke));
        }
        Kind::Polyline => {
            if let Some(bw) = polyline_band.filter(|b| *b > 1.0) {
                // preview of the exported band
                p.add(egui::Shape::line(sp.clone(), Stroke::new(bw, c.gamma_multiply(0.3))));
            }
            p.add(egui::Shape::line(sp.clone(), stroke));
        }
    }
    if selected {
        for (i, h) in s.handles().into_iter().enumerate() {
            let hp = to_s(h);
            if s.kind == Kind::BBox && i >= 4 {
                // edge-midpoint handle: small square
                let r = Rect::from_center_size(hp, Vec2::splat(8.0));
                p.rect_filled(r, 1.0, Color32::WHITE);
                p.rect_stroke(r, 1.0, Stroke::new(1.5_f32, c));
            } else {
                p.circle_filled(hp, 5.0, Color32::WHITE);
                p.circle_stroke(hp, 5.0, Stroke::new(1.5_f32, c));
            }
        }
        // move grip at the centre
        let g = to_s(s.center());
        let st = Stroke::new(2.0_f32, Color32::WHITE);
        p.circle_filled(g, 7.0, c.gamma_multiply(0.8));
        for (dx, dy) in [(1.0, 0.0), (0.0, 1.0)] {
            let d = Vec2::new(dx, dy) * 5.0;
            p.line_segment([g - d, g + d], st);
        }
    } else if s.kind != Kind::BBox {
        for q in &sp {
            p.circle_filled(*q, 2.5, c);
        }
    }
    if show_label {
        let galley = p.layout_no_wrap(label.to_string(), FontId::proportional(12.0), Color32::BLACK);
        let size = galley.size() + Vec2::new(6.0, 2.0);
        let r = match label_anchor(s, tri) {
            // box: tag sits on its top-left corner
            None => {
                let b = s.bounds();
                Rect::from_min_size(to_s([b[0], b[1]]) - Vec2::new(0.0, size.y), size)
            }
            // polygon / polyline: tag centred on the shape itself
            Some(a) => Rect::from_center_size(to_s(a), size),
        };
        p.rect_filled(r, 2.0, c);
        p.galley(r.min + Vec2::new(3.0, 1.0), galley, Color32::BLACK);
    }
}

/// Where the class tag of a polygon / polyline goes (image coordinates); None = box.
/// Polyline: halfway along its length. Polygon: the area centroid if it lies inside,
/// else the centre of the largest triangle of its triangulation (always inside).
fn label_anchor(s: &Annot, tri: &[u32]) -> Option<[f32; 2]> {
    let pts = &s.pts;
    match s.kind {
        Kind::BBox => None,
        Kind::Polyline => {
            let len: f32 = pts.windows(2).map(|w| dist(w[0], w[1])).sum();
            let mut left = len / 2.0;
            for w in pts.windows(2) {
                let d = dist(w[0], w[1]);
                if d >= left && d > 0.0 {
                    let t = left / d;
                    return Some([w[0][0] + t * (w[1][0] - w[0][0]), w[0][1] + t * (w[1][1] - w[0][1])]);
                }
                left -= d;
            }
            pts.first().copied()
        }
        Kind::Polygon => {
            let n = pts.len();
            let (mut a2, mut cx, mut cy) = (0.0f32, 0.0f32, 0.0f32);
            for i in 0..n {
                let (p, q) = (pts[i], pts[(i + 1) % n]);
                let c = p[0] * q[1] - q[0] * p[1];
                a2 += c;
                cx += (p[0] + q[0]) * c;
                cy += (p[1] + q[1]) * c;
            }
            if a2.abs() > 1e-6 {
                let c = [cx / (3.0 * a2), cy / (3.0 * a2)];
                if point_in_polygon(c, pts) {
                    return Some(c);
                }
            }
            let tri_area = |t: &[u32]| {
                let [a, b, c] = [pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]];
                ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs()
            };
            tri.chunks_exact(3)
                .filter(|t| t.iter().all(|i| (*i as usize) < n))
                .max_by(|x, y| tri_area(x).total_cmp(&tri_area(y)))
                .map(|t| {
                    let [a, b, c] = [pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]];
                    [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0]
                })
                .or_else(|| pts.first().copied())
        }
    }
}

impl App {
    /// Lab: ask for the user's name once; on the server machine: switch the dataset folder;
    /// banner for images someone else is editing.
    fn lab_windows(&mut self, ctx: &egui::Context) {
        if !self.lab() {
            return;
        }
        if self.user.is_empty() {
            egui::Window::new("Welcome to the lab annotator")
                .collapsible(false)
                .resizable(false)
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.label("Your name (shown to the others, saved with your labels):");
                    let r = ui.add(egui::TextEdit::singleline(&mut self.user_edit).desired_width(220.0));
                    self.typing |= r.has_focus();
                    r.request_focus();
                    let enter = r.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
                    let name = self.user_edit.trim().to_string();
                    if (ui.add_enabled(!name.is_empty(), egui::Button::new("Start")).clicked() || enter) && !name.is_empty() {
                        self.user = name.clone();
                        self.store.hello(&name);
                    }
                });
        }
        if self.admin_dialog {
            let mut open = true;
            egui::Window::new("Dataset folder (server)")
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("Folder on this server machine that everyone works on:");
                    let r = ui.add(egui::TextEdit::singleline(&mut self.admin_path).desired_width(360.0).hint_text(r"D:\datasets\my_dataset"));
                    self.typing |= r.has_focus();
                    ui.weak("Only possible from a browser on the server itself.");
                    if ui.button("Open for everyone").clicked() && !self.admin_path.trim().is_empty() {
                        let p = self.admin_path.trim().to_string();
                        self.store.open(&p);
                        self.admin_dialog = false;
                    }
                });
            self.admin_dialog &= open;
        }
        if let Some(u) = self.locked_by_other() {
            let msg = format!("View only: {u} is editing this image");
            egui::Area::new(egui::Id::new("lock_banner")).fade_in(false).order(egui::Order::Foreground)
                .anchor(Align2::CENTER_TOP, Vec2::new(0.0, 56.0))
                .interactable(false)
                .show(ctx, |ui| {
                    egui::Frame::popup(ui.style()).fill(Color32::from_rgb(120, 60, 20)).show(ui, |ui| {
                        ui.label(egui::RichText::new(msg).color(Color32::WHITE).strong());
                    });
                });
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Only text fields may keep keyboard focus. A focused button (e.g. after Tab) would
        // swallow every shortcut and be "clicked" by Space / Enter.
        if !self.typing {
            if let Some(id) = ctx.memory(|m| m.focused()) {
                ctx.memory_mut(|m| m.surrender_focus(id));
            }
        }
        self.typing = false; // set again by the text fields drawn this frame

        if !self.lab() {
            let dropped: Option<String> = ctx.input(|i| {
                i.raw.dropped_files.iter().find_map(|f| f.path.as_ref().map(|p| p.display().to_string()))
            });
            if let Some(p) = dropped {
                self.open_path(ctx, &p);
            }
        }

        self.pump(ctx);
        // lab: an image someone else is editing is view-only. Any change made to it this frame
        // (by whatever control) is undone again below.
        let guard = self
            .locked_by_other()
            .map(|_| self.cur_key().map(|k| (k.clone(), self.project.images.get(&k).cloned())))
            .flatten();
        self.handle_keys(ctx);
        self.top_bar(ctx);
        self.status_bar(ctx);
        self.file_list(ctx);
        self.right_panel(ctx);
        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| self.canvas(ui));
        self.export_window(ctx);
        self.about_window(ctx);
        self.lab_windows(ctx);
        if let Some((k, before)) = guard {
            let now = self.project.images.get(&k);
            let changed = match (&before, now) {
                (None, None) => false,
                (Some(a), Some(b)) => serde_json::to_string(a).ok() != serde_json::to_string(b).ok(),
                _ => true,
            };
            if changed && self.cur_key().as_ref() == Some(&k) {
                match before {
                    Some(a) => self.project.images.insert(k, a),
                    None => self.project.images.remove(&k),
                };
                self.label_dirty = false;
                self.undo.clear();
                self.selected = None;
                if let Some(u) = self.locked_by_other() {
                    self.status = format!("View only: {u} is editing this image.");
                }
            }
        }

        // the current image's YOLO label file is written as soon as an edit is finished
        if self.label_dirty && matches!(self.drag, Drag::None) && self.drawing.is_empty() {
            self.write_current_label();
        }
        // the project file (annotations.json) a while after the last change
        let now = ctx.input(|i| i.time);
        if self.dirty && matches!(self.drag, Drag::None) && now - self.last_save > 30.0 {
            self.save_current(false);
            self.last_save = now;
        }
        if self.dirty {
            ctx.request_repaint_after(std::time::Duration::from_secs(2));
        }
        if self.lab() {
            // messages from the server arrive at any time
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }
        if ctx.input(|i| i.viewport().close_requested()) {
            if !self.drawing.is_empty() {
                self.finish_drawing();
            }
            self.accept_magic();
            self.save_current(false);
        }
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, "user", &self.user);
    }
}

#[cfg(test)]
mod tri_tests {
    use super::*;

    #[test]
    fn triangulates_polygon_with_collinear_points() {
        let p = |v: &[[f32; 2]]| v.iter().map(|q| Pos2::new(q[0], q[1])).collect::<Vec<_>>();
        for poly in [
            p(&[[0., 478.], [300., 350.], [590., 226.], [633., 218.], [510., 470.], [388., 720.], [0., 720.]]),
            p(&[[633., 218.], [697., 222.], [990., 348.], [1280., 472.], [1280., 720.], [388., 720.], [510., 470.]]),
        ] {
            let t = triangulate(&poly);
            assert_eq!(t.len(), (poly.len() - 2) * 3, "triangles: {t:?}");
        }
    }
}

#[cfg(test)]
mod label_tests {
    use super::*;

    #[test]
    fn tags_sit_on_the_shape() {
        let line = Annot { class: 0, kind: Kind::Polyline, pts: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]] };
        assert_eq!(label_anchor(&line, &[]), Some([10.0, 0.0]));
        // concave "U": the centroid is in the notch, so the tag goes into a triangle instead
        let u = Annot {
            class: 0,
            kind: Kind::Polygon,
            pts: vec![[0.0, 0.0], [3.0, 0.0], [3.0, 8.0], [7.0, 8.0], [7.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        };
        let ip: Vec<Pos2> = u.pts.iter().map(|q| Pos2::new(q[0], q[1])).collect();
        let a = label_anchor(&u, &triangulate(&ip)).unwrap();
        assert!(point_in_polygon(a, &u.pts), "{a:?}");
        let b = Annot { class: 0, kind: Kind::BBox, pts: vec![[0.0, 0.0], [5.0, 5.0]] };
        assert_eq!(label_anchor(&b, &[]), None);
    }
}
