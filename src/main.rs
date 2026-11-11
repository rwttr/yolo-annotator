#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod ai;
mod icon;
mod mask;
mod model;

use eframe::egui::{
    self, Align2, Color32, CursorIcon, FontId, Key, Modifiers, PointerButton, Pos2, Rect, Sense,
    Stroke, Vec2,
};
use image::ImageDecoder;
use model::*;
use std::fs;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

const HIT_TOL: f32 = 8.0; // screen pixels
const CLOSE_TOL: f32 = 10.0; // screen pixels
const MAX_TEX: u32 = 8192;

/// `yolo-annotator --ai-selftest <image> [yolo.onnx] [auto|cuda|directml|cpu]`
/// Runs SAM (centre click) and optionally YOLO on one image; writes ai_selftest.txt.
fn ai_selftest(args: &[String]) {
    use std::io::Write;
    let mut log = fs::File::create("ai_selftest.txt").ok();
    let mut out = |s: String| {
        println!("{s}");
        if let Some(f) = &mut log {
            let _ = writeln!(f, "{s}");
        }
    };
    let Some(path) = args.first() else {
        out("usage: --ai-selftest <image> [yolo.onnx] [auto|cuda|directml|cpu]".into());
        return;
    };
    let mut settings = ai::AiSettings::load();
    settings.yolo = args.get(1).cloned().filter(|s| s.ends_with(".onnx")).unwrap_or_default();
    if let Some(e) = args.iter().find_map(|a| match a.to_lowercase().as_str() {
        "auto" => Some(ai::Engine::Auto),
        "cuda" => Some(ai::Engine::Cuda),
        "directml" => Some(ai::Engine::DirectML),
        "cpu" => Some(ai::Engine::Cpu),
        _ => None,
    }) {
        settings.engine = e;
    }
    if let Some(m) = args.iter().find(|a| a.starts_with("sam2")) {
        settings.sam = m.clone();
    }
    let img = match App::decode_ai(Path::new(path)) {
        Ok(i) => i,
        Err(e) => return out(format!("decode error: {e}")),
    };
    out(format!("image {}x{} (AI copy {}x{})", img.w, img.h, img.rgb.width(), img.rgb.height()));
    let t0 = std::time::Instant::now();
    let h = ai::start(settings.clone(), || {});
    let wait = |out: &mut dyn FnMut(String)| -> Option<ai::Resp> {
        loop {
            match h.rx.recv_timeout(std::time::Duration::from_secs(120)) {
                Ok(ai::Resp::Ready(s)) => out(format!("[{:>6} ms] ready: {s}", t0.elapsed().as_millis())),
                Ok(ai::Resp::Info(s)) => out(format!("info: {s}")),
                Ok(r) => return Some(r),
                Err(_) => return None,
            }
        }
    };
    let mut overlays: Vec<(Vec<[f32; 2]>, image::Rgb<u8>)> = vec![];
    for round in 0..2 {
        let key = format!("selftest{round}");
        let _ = h.tx.send(ai::Req::Encode { key: key.clone(), img: img.clone() });
        match wait(&mut out) {
            Some(ai::Resp::Encoded { ms, .. }) => out(format!("[{:>6} ms] SAM encode #{round}: {ms} ms", t0.elapsed().as_millis())),
            Some(ai::Resp::Error(e)) => return out(format!("error: {e}")),
            _ => return out("no answer from the AI worker".into()),
        }
        let c = [img.w as f32 / 2.0, img.h as f32 / 2.0];
        for (i, poly) in [true, false].into_iter().enumerate() {
            let t = std::time::Instant::now();
            let _ = h.tx.send(ai::Req::Segment { key: key.clone(), seq: i as u64, points: vec![(c, true)], bbox: None, polygon: poly });
            match wait(&mut out) {
                Some(ai::Resp::Mask { shape, .. }) => {
                    if poly && round == 0 {
                        if let Some(s) = &shape {
                            overlays.push((s.clone(), image::Rgb([0u8, 255, 0])));
                        }
                    }
                    out(format!(
                    "  click centre -> {} in {} ms: {:?}",
                    if poly { "polygon" } else { "box" },
                    t.elapsed().as_millis(),
                    shape.map(|s| if s.len() > 4 { format!("{} points, first {:?}", s.len(), s[0]) } else { format!("{s:?}") })
                    ));
                }
                Some(ai::Resp::Error(e)) => out(format!("  segment error: {e}")),
                _ => out("  no answer".into()),
            }
        }
        if round == 0 {
            // polygon size for a few Detail settings
            let mut line = "  detail ->".to_string();
            for d in [0.5f32, 1.0, 2.5, 4.0] {
                let _ = h.tx.send(ai::Req::Settings(ai::AiSettings { detail: d, ..settings.clone() }));
                let _ = h.tx.send(ai::Req::Segment { key: key.clone(), seq: 5, points: vec![(c, true)], bbox: None, polygon: true });
                if let Some(ai::Resp::Mask { shape, .. }) = wait(&mut out) {
                    line.push_str(&format!("  {d}: {} pts", shape.map_or(0, |s| s.len())));
                }
            }
            let _ = h.tx.send(ai::Req::Settings(settings.clone()));
            out(line);
        }
        // box prompt over the middle half
        let b = [img.w as f32 * 0.25, img.h as f32 * 0.25, img.w as f32 * 0.75, img.h as f32 * 0.75];
        let t = std::time::Instant::now();
        let _ = h.tx.send(ai::Req::Segment { key: key.clone(), seq: 9, points: vec![], bbox: Some(b), polygon: true });
        match wait(&mut out) {
            Some(ai::Resp::Mask { shape, .. }) => out(format!("  box prompt -> {:?} points in {} ms", shape.map(|s| s.len()), t.elapsed().as_millis())),
            Some(ai::Resp::Error(e)) => out(format!("  box prompt error: {e}")),
            _ => out("  no answer".into()),
        }
    }
    if !settings.yolo.is_empty() {
        for round in 0..2 {
            let t = std::time::Instant::now();
            let _ = h.tx.send(ai::Req::Detect { key: "x".into(), img: img.clone() });
            match wait(&mut out) {
                Some(ai::Resp::Detected { dets, .. }) => {
                    out(format!("[{:>6} ms] YOLO #{round}: {} detections in {} ms", t0.elapsed().as_millis(), dets.len(), t.elapsed().as_millis()));
                    if round == 0 {
                        for d in &dets {
                            let b = d.bbox;
                            overlays.push((vec![[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], image::Rgb([255, 200, 0])));
                            if let Some(p) = &d.polygon {
                                overlays.push((p.clone(), image::Rgb([255, 0, 255])));
                            }
                            out(format!(
                                "  {} ({}) {:.2} box [{:.0} {:.0} {:.0} {:.0}] poly {:?}",
                                d.class_name.clone().unwrap_or_default(),
                                d.class_id,
                                d.score,
                                d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3],
                                d.polygon.as_ref().map(|p| p.len())
                            ));
                        }
                    }
                }
                Some(ai::Resp::Error(e)) => out(format!("YOLO error: {e}")),
                _ => out("no answer".into()),
            }
        }
    }
    // overlay picture of the results, for a visual check
    if let Ok(src) = App::decode(Path::new(path)) {
        let mut im = src.to_rgb8();
        for (poly, col) in &overlays {
            for i in 0..poly.len() {
                let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
                let n = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()) as usize).max(1);
                for k in 0..=n {
                    let t = k as f32 / n as f32;
                    let (x, y) = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t);
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)] {
                        let (px, py) = (x as i64 + dx, y as i64 + dy);
                        if px >= 0 && py >= 0 && (px as u32) < im.width() && (py as u32) < im.height() {
                            im.put_pixel(px as u32, py as u32, *col);
                        }
                    }
                }
            }
            for p in poly {
                for dy in -3..=3i64 {
                    for dx in -3..=3i64 {
                        let (px, py) = (p[0] as i64 + dx, p[1] as i64 + dy);
                        if px >= 0 && py >= 0 && (px as u32) < im.width() && (py as u32) < im.height() {
                            im.put_pixel(px as u32, py as u32, image::Rgb([255, 255, 255]));
                        }
                    }
                }
            }
        }
        let _ = im.save("ai_selftest.png");
        out("overlay: ai_selftest.png".into());
    }
}

fn main() -> eframe::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--ai-selftest") {
        ai_selftest(&args[1..]);
        return Ok(());
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([800.0, 500.0])
            .with_title("YOLO Annotator")
            .with_icon(egui::IconData { rgba: icon::rgba(64), width: 64, height: 64 })
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "YOLO Annotator",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
    Select,
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

/// "Auto-label all unlabeled" running in the AI worker.
struct BatchJob {
    id: u64,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    done: Arc<AtomicUsize>,
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

struct Loaded {
    name: String,
    tex: egui::TextureHandle,
    w: u32,
    h: u32,
    /// downscaled copy for SAM / YOLO
    ai: ai::AiImage,
}

struct ExportDlg {
    open: bool,
    out: String,
    val_ratio: f32,
    include_unlabeled: bool,
    keep_split: bool,
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

#[derive(Default)]
struct ImportStats {
    new: usize,
    refreshed: usize,
    box_lines: usize,
    seg_lines: usize,
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

type Decoded = Result<(egui::ColorImage, u32, u32, ai::AiImage), String>;

/// A dataset export running on a worker thread.
struct ExportJob {
    done: Arc<AtomicUsize>,
    total: Arc<AtomicUsize>,
    rx: mpsc::Receiver<Result<ExportStats, String>>,
    out: PathBuf,
}

/// Background decoding of the image the user will most likely open next.
struct Prefetch {
    name: String,
    rx: mpsc::Receiver<Decoded>,
}

struct App {
    dir: Option<PathBuf>,
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
    prefetch: Option<Prefetch>,
    nav_dir: i64,
    last_draw_tool: Tool,
    about_open: bool,
    /// a text field had keyboard focus last frame (shortcuts are off while typing)
    typing: bool,
    /// polygon triangulations, keyed by a hash of the points (zoom/pan independent)
    tri_cache: RefCell<HashMap<u64, Vec<u32>>>,
    export_job: Option<ExportJob>,
    // ---- AI assist
    ai: Option<ai::AiHandle>,
    ai_settings: ai::AiSettings,
    /// engine / model in use, or why AI is unavailable
    ai_status: String,
    ai_models: Vec<String>,
    /// AI keys of images whose SAM embedding is ready in the worker
    encoded: HashSet<String>,
    magic: Magic,
    batch: Option<BatchJob>,
    last_encode_ms: u128,
    /// id of the newest auto-label batch (results of older ones are ignored)
    batch_seq: u64,
    /// (AI key, reason) when SAM could not prepare that image
    encode_error: Option<(String, String)>,
    ctx: egui::Context,
}

fn class_color(i: usize) -> Color32 {
    let h = (i as f32 * 0.618_034 + 0.05) % 1.0;
    egui::ecolor::Hsva::new(h, 0.8, 1.0, 1.0).into()
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let mut app = App {
            dir: None,
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
            status: "Open an image folder to start.".into(),
            export: ExportDlg {
                open: false,
                out: String::new(),
                val_ratio: 0.2,
                include_unlabeled: false,
                keep_split: true,
            },
            cursor_img: None,
            scroll_list: false,
            show_labels: true,
            tree: DirNode::default(),
            filter: String::new(),
            unlabeled_only: false,
            label_dirty: false,
            expanded: HashSet::new(),
            prefetch: None,
            nav_dir: 1,
            last_draw_tool: Tool::BBox,
            about_open: false,
            typing: false,
            tri_cache: RefCell::new(HashMap::new()),
            stale: HashSet::new(),
            export_job: None,
            ai: None,
            ai_settings: ai::AiSettings::load(),
            ai_status: "not started (select the Magic tool or Auto-label)".into(),
            ai_models: ai::sam_models(),
            encoded: HashSet::new(),
            magic: Magic::default(),
            batch: None,
            last_encode_ms: 0,
            batch_seq: 0,
            encode_error: None,
            ctx: cc.egui_ctx.clone(),
        };
        if let Some(arg) = std::env::args().nth(1) {
            app.open_path(&cc.egui_ctx, PathBuf::from(arg));
        }
        app
    }

    // ------------------------------------------------------------ files

    fn open_path(&mut self, ctx: &egui::Context, path: PathBuf) {
        if path.is_dir() {
            self.open_folder(ctx, path);
        } else if let (Some(parent), Some(_)) = (path.parent(), path.file_name()) {
            // a file inside the already-open dataset: just jump to it
            let rel_in = |root: &Path| {
                path.strip_prefix(root)
                    .ok()
                    .map(|r| r.to_string_lossy().replace('\\', "/"))
            };
            if let Some(rel) = self.dir.as_deref().and_then(rel_in) {
                if let Some(i) = self.files.iter().position(|f| *f == rel) {
                    self.load_image(ctx, i);
                    return;
                }
            }
            // inside the open dataset but not listed yet (added later): rescan the same root
            let target = match &self.dir {
                Some(root) if path.starts_with(root) => root.clone(),
                _ => parent.to_path_buf(),
            };
            self.open_folder(ctx, target);
            if let Some(rel) = self.dir.as_deref().and_then(rel_in) {
                if let Some(i) = self.files.iter().position(|f| *f == rel) {
                    self.load_image(ctx, i);
                }
            }
        }
    }

    /// Open a dataset folder: all images below it (any depth) are listed.
    fn open_folder(&mut self, ctx: &egui::Context, dir: PathBuf) {
        self.accept_magic();
        self.save_current(false);
        if let Some(b) = &self.batch {
            b.cancel.store(true, Ordering::Relaxed);
        }
        self.batch = None;
        self.magic = Magic { seq: self.magic.seq + 1, ..Default::default() };
        let files = match scan_images(&dir) {
            Ok(f) => f,
            Err(e) => {
                self.status = e;
                return;
            }
        };
        let mut warn = None;
        let mut fresh = false;
        let project = match Project::load(&dir) {
            Ok(Some(p)) => p,
            Ok(None) => {
                fresh = true;
                let mut p = Project::default();
                if let Some(c) = find_classes(&dir) {
                    p.classes = c;
                }
                p
            }
            Err(LoadError::Unreadable(e)) => {
                self.status = e;
                return;
            }
            Err(LoadError::Corrupt(e)) => {
                warn = Some(e);
                fresh = true;
                let mut p = Project::default();
                if let Some(c) = find_classes(&dir) {
                    p.classes = c;
                }
                p
            }
        };
        let project_mtime = fs::metadata(dir.join(PROJECT_FILE)).and_then(|m| m.modified()).ok();
        self.project = project;
        self.files = files;
        self.tree = DirNode::build(&self.files);
        // start with the top-level folders expanded
        self.expanded = self.tree.dirs.iter().map(|d| d.path.clone()).collect();
        self.prefetch = None;
        self.dir = Some(dir.clone());
        self.loaded = None;
        self.undo.clear();
        self.stale.clear();
        self.selected = None;
        self.drag = Drag::None;
        self.class = 0;
        self.dirty = false;
        self.label_dirty = false;
        self.labels_stale = false;
        let imp = self.import_existing_labels(&dir, project_mtime);
        let imported = imp.new + imp.refreshed;
        if imported > 0 {
            self.dirty = true; // record them in annotations.json
        }
        // an existing detection-only dataset stays in detection format
        if fresh && imp.box_lines > 0 && imp.seg_lines == 0 {
            self.project.export_mode = ExportMode::Detect;
        }
        let labeled = self
            .files
            .iter()
            .filter(|f| self.project.images.contains_key(*f))
            .count();
        self.status = warn.unwrap_or_else(|| {
            format!(
                "Opened {} images in {} ({} labeled{}). Label files are created next to each image folder (images → labels).",
                self.files.len(),
                dir.display(),
                labeled,
                match (imp.new, imp.refreshed) {
                    (0, 0) => String::new(),
                    (n, 0) => format!(", {n} imported from existing YOLO .txt files"),
                    (n, r) => format!(
                        ", {n} imported, {r} re-read from .txt files newer than {PROJECT_FILE}"
                    ),
                }
            )
        });
        let clashes = label_collisions(&dir, &self.files);
        if let Some((a, b)) = clashes.first() {
            self.status = format!(
                "WARNING: {} image pair(s) share a label file name, e.g. '{a}' and '{b}'. Rename one of them, or their labels overwrite each other. {}",
                clashes.len(),
                self.status
            );
        }
        if !self.files.is_empty() {
            // resume at the first image that has no annotation yet
            let start = self
                .files
                .iter()
                .position(|f| !self.project.images.contains_key(f))
                .unwrap_or(0);
            self.load_image(ctx, start);
        }
    }

    /// Read existing YOLO label files for every image not in the project yet, so the
    /// whole dataset (not only visited images) is known and exported.
    /// Label files written after annotations.json (e.g. the app was killed before it could
    /// save the project file, or the .txt was edited elsewhere) are re-read as well.
    fn import_existing_labels(&mut self, root: &Path, project_mtime: Option<std::time::SystemTime>) -> ImportStats {
        let mut st = ImportStats::default();
        let grace = std::time::Duration::from_secs(2);
        for rel in &self.files {
            let lp = label_path(root, rel);
            let existing = self.project.images.get(rel).map(|a| (a.width, a.height));
            if existing.is_some() {
                let newer = match (project_mtime, fs::metadata(&lp).and_then(|m| m.modified())) {
                    (Some(pt), Ok(lt)) => lt > pt + grace,
                    _ => false,
                };
                if !newer {
                    continue;
                }
            }
            let Ok(text) = fs::read_to_string(&lp) else { continue };
            let (w, h) = match existing {
                Some(wh) => wh,
                // header-only read; the exact (EXIF-rotated) size is fixed when the image is opened
                None => match image::image_dimensions(root.join(rel)) {
                    Ok(wh) => wh,
                    Err(_) => continue,
                },
            };
            for line in text.lines() {
                match line.split_whitespace().count() {
                    0 => {}
                    5 => st.box_lines += 1,
                    _ => st.seg_lines += 1,
                }
            }
            let shapes = parse_yolo(&text, w, h);
            self.project.images.insert(rel.clone(), ImageAnn { width: w, height: h, shapes });
            if existing.is_some() {
                st.refreshed += 1;
            } else {
                st.new += 1;
            }
        }
        self.ensure_classes();
        st
    }

    /// Make sure every class index used by a shape has a name.
    fn ensure_classes(&mut self) {
        let maxc = self
            .project
            .images
            .values()
            .flat_map(|a| a.shapes.iter().map(|s| s.class))
            .max();
        if let Some(maxc) = maxc {
            while self.project.classes.len() <= maxc {
                let n = self.project.classes.len();
                self.project.classes.push(format!("class_{n}"));
            }
        }
    }

    /// Decode + downscale + convert, ready for a texture. Runs on a worker thread for prefetch.
    /// A decoder panic on a corrupt file becomes an error instead of killing the app.
    fn decode_for_display(path: &Path) -> Decoded {
        std::panic::catch_unwind(|| Self::decode_for_display_inner(path))
            .unwrap_or_else(|_| Err("the image decoder crashed on this file (corrupt image?)".into()))
    }

    fn decode_for_display_inner(path: &Path) -> Decoded {
        let img = Self::decode(path)?;
        let (w, h) = (img.width(), img.height());
        let disp = if w.max(h) > MAX_TEX {
            img.resize(MAX_TEX, MAX_TEX, image::imageops::FilterType::Triangle)
        } else {
            img
        };
        let rgba = disp.to_rgba8();
        let ci = egui::ColorImage::from_rgba_unmultiplied(
            [rgba.width() as usize, rgba.height() as usize],
            rgba.as_raw(),
        );
        Ok((ci, w, h, Self::ai_image(&disp, w, h)))
    }

    /// Downscaled RGB copy for the AI models (they work at 640-1024 px anyway).
    fn ai_image(img: &image::DynamicImage, w: u32, h: u32) -> ai::AiImage {
        let rgb = if img.width().max(img.height()) > ai::AI_MAX_SIDE {
            img.thumbnail(ai::AI_MAX_SIDE, ai::AI_MAX_SIDE).to_rgb8()
        } else {
            img.to_rgb8()
        };
        ai::AiImage { rgb: Arc::new(rgb), w, h }
    }

    /// Decode for the AI only (batch auto-labelling).
    fn decode_ai(path: &Path) -> Result<ai::AiImage, String> {
        std::panic::catch_unwind(|| {
            let img = Self::decode(path)?;
            Ok(Self::ai_image(&img, img.width(), img.height()))
        })
        .unwrap_or_else(|_| Err("the image decoder crashed on this file (corrupt image?)".into()))
    }

    fn start_prefetch(&mut self, idx: usize) {
        let (Some(dir), Some(name)) = (&self.dir, self.files.get(idx)) else {
            self.prefetch = None;
            return;
        };
        let path = dir.join(name);
        // with the Magic tool, SAM prepares the next image as soon as it is decoded
        let ai = self
            .ai
            .as_ref()
            .filter(|_| self.tool == Tool::Magic)
            .map(|a| (a.tx.clone(), self.ai_key(name)));
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let d = Self::decode_for_display(&path);
            if let (Ok((_, _, _, img)), Some((atx, key))) = (&d, ai) {
                let _ = atx.send(ai::Req::Encode { key, img: img.clone() });
            }
            let _ = tx.send(d);
        });
        self.prefetch = Some(Prefetch { name: name.clone(), rx });
    }

    fn decode(path: &Path) -> Result<image::DynamicImage, String> {
        let reader = image::ImageReader::open(path)
            .map_err(|e| e.to_string())?
            .with_guessed_format()
            .map_err(|e| e.to_string())?;
        let mut dec = reader.into_decoder().map_err(|e| e.to_string())?;
        // honour EXIF rotation, the same way OpenCV / Ultralytics load the image
        let orient = dec
            .orientation()
            .unwrap_or(image::metadata::Orientation::NoTransforms);
        let mut img = image::DynamicImage::from_decoder(dec).map_err(|e| e.to_string())?;
        img.apply_orientation(orient);
        Ok(img)
    }

    fn load_image(&mut self, ctx: &egui::Context, idx: usize) {
        if idx >= self.files.len() {
            return;
        }
        // keep a polygon / polyline that is still being drawn, and a Magic result
        if !self.drawing.is_empty() {
            self.finish_drawing();
        }
        self.accept_magic();
        self.end_drag();
        self.save_current(false);
        let Some(dir) = self.dir.clone() else { return };
        let name = self.files[idx].clone();
        self.cur = idx;
        self.selected = None;
        self.drawing.clear();
        self.undo.clear();
        self.tri_cache.borrow_mut().clear();
        self.drag = Drag::None;
        self.fit_pending = true;
        self.scroll_list = true;
        for a in ancestors(&name) {
            self.expanded.insert(a.to_string());
        }
        // use the background-decoded image if it is the one we prefetched
        let decoded = match self.prefetch.take() {
            Some(pf) if pf.name == name => pf
                .rx
                .recv()
                .unwrap_or_else(|_| Self::decode_for_display(&dir.join(&name))),
            _ => Self::decode_for_display(&dir.join(&name)),
        };
        let next = idx as i64 + self.nav_dir;
        if next >= 0 && (next as usize) < self.files.len() {
            self.start_prefetch(next as usize);
        }
        match decoded {
            Ok((ci, w, h, ai_img)) => {
                let tex = ctx.load_texture(&name, ci, egui::TextureOptions::LINEAR);
                // (normalized positions are unchanged, so the label file needs no rewrite)
                if let Some(a) = self.project.images.get_mut(&name) {
                    if (a.width, a.height) != (w, h) {
                        rescale(a, w, h);
                        self.dirty = true;
                    }
                } else if let Ok(s) = fs::read_to_string(label_path(&dir, &name)) {
                    // a YOLO label file that appeared since the dataset was opened
                    let shapes = parse_yolo(&s, w, h);
                    self.project.images.insert(name.clone(), ImageAnn { width: w, height: h, shapes });
                    self.ensure_classes();
                    self.dirty = true;
                }
                self.loaded = Some(Loaded { name, tex, w, h, ai: ai_img });
                if self.tool == Tool::Magic {
                    self.request_encode();
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
        if !self.dirty && !self.labels_stale && self.stale.is_empty() && !force {
            return;
        }
        let Some(dir) = self.dir.clone() else { return };
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
            if let Err(e) = write_label_file(&dir, &name, a, &self.project) {
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
        if let Err(e) = write_classes(&dir, &self.project) {
            errs.push(e);
        }
        if let Err(e) = self.project.save(&dir) {
            errs.push(e);
        }
        self.dirty = false;
        if !errs.is_empty() {
            self.status = format!("Save error: {}", errs.join("; "));
        } else if force {
            self.status = format!("Saved. Project file: {}", dir.join(PROJECT_FILE).display());
        }
    }

    /// Write the YOLO .txt of the current image right away.
    fn write_current_label(&mut self) {
        self.label_dirty = false;
        let (Some(dir), Some(l)) = (&self.dir, &self.loaded) else { return };
        let Some(a) = self.project.images.get(&l.name) else { return };
        self.status = match write_label_file(dir, &l.name, a, &self.project) {
            Ok(p) => {
                let shown = p.strip_prefix(dir).unwrap_or(&p).display().to_string();
                format!("✔ {} objects → {shown}", a.shapes.len())
            }
            Err(e) => format!("Save error: {e}"),
        };
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
                    if let (Some(dir), Some(l)) = (&self.dir, &self.loaded) {
                        self.project.images.remove(&l.name);
                        let _ = fs::remove_file(label_path(dir, &l.name));
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
            self.project.classes.push("object".into());
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
        if t != Tool::Select {
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
        if self.ai.is_none() {
            let ctx = self.ctx.clone();
            self.ai = Some(ai::start(self.ai_settings.clone(), move || ctx.request_repaint()));
            self.ai_status = "starting...".into();
        }
    }

    /// Worker cache key: dataset folder + image (two datasets may share relative names).
    fn ai_key(&self, name: &str) -> String {
        format!("{}|{name}", self.dir.as_deref().map(|d| d.display().to_string()).unwrap_or_default())
    }

    fn cur_key(&self) -> Option<String> {
        self.loaded.as_ref().map(|l| self.ai_key(&l.name))
    }

    /// Ask SAM to prepare the current image (no-op if it already is).
    fn request_encode(&mut self) {
        let (Some(a), Some(l)) = (&self.ai, &self.loaded) else { return };
        let key = self.ai_key(&l.name);
        if !self.encoded.contains(&key) {
            let _ = a.tx.send(ai::Req::Encode { key, img: l.ai.clone() });
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
        if self.ai.is_none() {
            return;
        }
        if !self.encoded.contains(&key) {
            self.magic.unsent = true;
            self.request_encode(); // (again, in case an earlier request was dropped)
            return;
        }
        let Some(a) = &self.ai else { return };
        self.magic.seq += 1;
        self.magic.unsent = false;
        let polygon = self.project.export_mode == ExportMode::Segment;
        let _ = a.tx.send(ai::Req::Segment {
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

    fn auto_label(&mut self) {
        if self.ai_settings.yolo.trim().is_empty() {
            self.status = "Auto-label needs your YOLO model: choose a .onnx file under 'AI assist' (right panel).".into();
            return;
        }
        self.ensure_ai();
        let (Some(a), Some(l)) = (&self.ai, &self.loaded) else { return };
        let _ = a.tx.send(ai::Req::Detect { key: self.ai_key(&l.name), img: l.ai.clone() });
        self.status = "Auto-label: running the YOLO model...".into();
    }

    fn auto_label_all(&mut self) {
        if self.ai_settings.yolo.trim().is_empty() {
            self.status = "Auto-label needs your YOLO model: choose a .onnx file under 'AI assist' (right panel).".into();
            return;
        }
        let Some(dir) = self.dir.clone() else { return };
        let items: Vec<(String, PathBuf)> = self
            .files
            .iter()
            .filter(|f| !self.project.images.contains_key(*f))
            .map(|f| (f.clone(), dir.join(f)))
            .collect();
        if items.is_empty() {
            self.status = "Every image is already labeled.".into();
            return;
        }
        self.ensure_ai();
        let Some(a) = &self.ai else { return };
        self.batch_seq += 1;
        let job = BatchJob {
            id: self.batch_seq,
            cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            done: Arc::new(AtomicUsize::new(0)),
            total: items.len(),
            added: 0,
        };
        let _ = a.tx.send(ai::Req::Batch {
            id: job.id,
            items,
            decode: Self::decode_ai,
            cancel: job.cancel.clone(),
            done: job.done.clone(),
        });
        self.batch = Some(job);
    }

    /// YOLO detections -> shapes in the project's classes (unknown names are added).
    fn detections_to_shapes(&mut self, dets: &[ai::Detection], w: u32, h: u32) -> Vec<Annot> {
        let seg = self.project.export_mode == ExportMode::Segment;
        // a new dataset only has the placeholder class "object": use the model's classes instead
        let placeholder_only = self.project.classes.len() == 1
            && self.project.classes[0] == "object"
            && self.project.images.values().all(|a| a.shapes.is_empty());
        if placeholder_only && !dets.is_empty() && dets.iter().all(|d| d.class_name.is_some()) {
            self.project.classes.clear();
        }
        let mut out = vec![];
        for d in dets {
            let class = match &d.class_name {
                Some(n) => match self.project.classes.iter().position(|c| c.eq_ignore_ascii_case(n)) {
                    Some(i) => i,
                    None => {
                        self.project.classes.push(n.clone());
                        self.project.classes.len() - 1
                    }
                },
                None => {
                    while self.project.classes.len() <= d.class_id {
                        let i = self.project.classes.len();
                        self.project.classes.push(format!("class_{i}"));
                    }
                    d.class_id
                }
            };
            let cl = |p: [f32; 2]| [p[0].clamp(0.0, w as f32), p[1].clamp(0.0, h as f32)];
            let a = match (&d.polygon, seg) {
                (Some(p), true) if p.len() >= 3 => Annot { class, kind: Kind::Polygon, pts: p.iter().map(|q| cl(*q)).collect() },
                _ => Annot {
                    class,
                    kind: Kind::BBox,
                    pts: vec![cl([d.bbox[0], d.bbox[1]]), cl([d.bbox[2], d.bbox[3]])],
                },
            };
            out.push(a);
        }
        out
    }

    fn poll_ai(&mut self) {
        let Some(a) = &self.ai else { return };
        let msgs: Vec<ai::Resp> = a.rx.try_iter().collect();
        for m in msgs {
            match m {
                ai::Resp::Ready(s) => self.ai_status = s,
                ai::Resp::Unavailable(e) => {
                    self.ai_status = format!("unavailable: {e}");
                    self.status = format!("AI: {e}");
                }
                ai::Resp::Crashed => {
                    self.ai_status = "unavailable: the AI worker crashed (restart the app)".into();
                    self.status = "AI: the AI worker crashed; restart the app to use AI again.".into();
                    self.batch = None;
                    self.encoded.clear();
                }
                ai::Resp::EncodeFailed { key, err } => {
                    if Some(&key) == self.cur_key().as_ref() {
                        self.status = format!("AI could not prepare this image: {err}");
                    }
                    self.encode_error = Some((key, err));
                }
                ai::Resp::Info(s) => self.status = s,
                ai::Resp::Error(e) => {
                    self.status = format!("AI: {e}");
                    if self.ai_status.starts_with("starting") {
                        self.ai_status = format!("unavailable: {e}");
                    }
                }
                ai::Resp::Encoded { key, ms } => {
                    if ms > 0 {
                        self.last_encode_ms = ms;
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
                ai::Resp::NeedEncode(key) => {
                    self.encoded.remove(&key);
                    if Some(&key) == self.cur_key().as_ref() && self.tool == Tool::Magic {
                        self.magic.unsent = !self.magic.is_empty();
                        self.request_encode();
                    }
                }
                ai::Resp::Mask { key, seq, shape } => {
                    if Some(&key) == self.cur_key().as_ref() && seq == self.magic.seq {
                        if shape.is_none() {
                            self.status = "Magic: no object found there. Try another point or drag a box.".into();
                        }
                        self.magic.shape_is_box = shape.as_ref().is_some_and(|s| s.len() == 2);
                        self.magic.shape = shape;
                    }
                }
                ai::Resp::Detected { key, batch, dets, w, h } => {
                    if let Some(id) = batch {
                        // a late result of a cancelled batch (maybe for another dataset) is dropped
                        let current = self.batch.as_ref().is_some_and(|b| b.id == id);
                        if !current || dets.is_empty() || self.project.images.contains_key(&key) {
                            continue;
                        }
                        let shapes = self.detections_to_shapes(&dets, w, h);
                        self.project.images.insert(key.clone(), ImageAnn { width: w, height: h, shapes });
                        self.stale.insert(key);
                        self.dirty = true;
                        if let Some(b) = &mut self.batch {
                            b.added += 1;
                        }
                    } else if Some(&key) != self.cur_key().as_ref() {
                        self.status = "Auto-label result discarded: the image was changed meanwhile.".into();
                    } else {
                        let shapes = self.detections_to_shapes(&dets, w, h);
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
                }
                ai::Resp::BatchDone { id, labeled, errors, cancelled } => {
                    if self.batch.as_ref().map_or(true, |b| b.id != id) {
                        continue; // an older, replaced batch
                    }
                    let added = self.batch.take().map_or(labeled, |b| b.added);
                    self.save_current(false);
                    self.status = format!(
                        "Auto-label {}: {added} images labeled{}",
                        if cancelled { "cancelled" } else { "finished" },
                        if errors.is_empty() { String::new() } else { format!("; errors: {}", errors.join("; ")) }
                    );
                }
            }
        }
    }

    fn ai_settings_changed(&mut self, sam_changed: bool) {
        self.ai_settings.save();
        if let Some(a) = &self.ai {
            let _ = a.tx.send(ai::Req::Settings(self.ai_settings.clone()));
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

    /// Finish an edit drag: tidy the box corners; a box squashed to nothing is undone.
    fn end_drag(&mut self) {
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
        let hit = if self.tool == Tool::Select {
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
                self.drag = d;
                true
            }
            None => false,
        }
    }

    /// What a drag at `p` would do, as a cursor icon (None = nothing to edit there).
    fn edit_cursor(&self, p: [f32; 2], tol: f32) -> Option<CursorIcon> {
        let shapes = self.shapes();
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
                ],
                NUMS.iter().position(|k| p(*k)),
            )
        });
        let [v, b, pk, l, prev, next, del, back, esc, enter, f, q, mg] = k;
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
        if back && !self.drawing.is_empty() {
            self.drawing.pop();
        } else if del {
            if let Some(s) = self.selected {
                self.delete_shape(s);
            }
        }
        if esc {
            if !self.drawing.is_empty() {
                self.drawing.clear();
            } else if matches!(self.drag, Drag::NewBox { .. }) {
                self.drag = Drag::None;
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
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "Open an image folder (Open button or Ctrl+O), or drag a folder onto this window",
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
                        Tool::Select => self.drag = Drag::Pan,
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
                        self.selected = pos_img.and_then(|p| pick(self, p));
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
                    self.selected = pos_img.and_then(|p| pick(self, p));
                }
            }
            Tool::Polygon | Tool::Polyline => {
                // right-click while not drawing selects / deselects
                if rclick && self.drawing.is_empty() {
                    self.selected = pos_img.and_then(|p| pick(self, p));
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
                if clicked {
                    self.selected = pos_img.and_then(|p| pick(self, p));
                }
                // double-click on an edge of the selected polygon/polyline: insert a vertex
                if dbl {
                    if let (Some(si), Some(p)) = (self.selected, pos_img) {
                        let hit = self.shapes().get(si).and_then(|s| {
                            if s.kind == Kind::BBox {
                                return None;
                            }
                            edges(s)
                                .into_iter()
                                .map(|e| (e.0, dist_to_segment(p, e.1, e.2)))
                                .filter(|e| e.1 <= tol)
                                .min_by(|a, b| a.1.total_cmp(&b.1))
                        });
                        if let Some((ei, _)) = hit {
                            self.push_undo();
                            if let Some(s) = self.shape_mut(si) {
                                s.pts.insert(ei + 1, clampi(p));
                            }
                            self.touch();
                        }
                    }
                }
                // right-click on a vertex: remove it
                if rclick {
                    if let Some(p) = pos_img {
                        if let Some((si, vi)) = self.hit_vertex(p, tol) {
                            let s = &self.shapes()[si];
                            let min = match s.kind {
                                Kind::Polygon => 3,
                                Kind::Polyline => 2,
                                Kind::BBox => usize::MAX,
                            };
                            if s.pts.len() > min {
                                self.push_undo();
                                if let Some(s) = self.shape_mut(si) {
                                    s.pts.remove(vi);
                                }
                                self.selected = Some(si);
                                self.touch();
                            }
                        }
                    }
                }
            }
        }

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
                edit.unwrap_or(if self.tool == Tool::Select {
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
        if self.tool != Tool::Select && !space {
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
        let msg = if self.ai.is_none() || self.ai_status.starts_with("unavailable") {
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
        if let Some(d) = rfd::FileDialog::new().pick_folder() {
            self.open_folder(ctx, d);
        }
    }

    fn open_export(&mut self) {
        let Some(d) = &self.dir else { return };
        if self.export.out.is_empty() {
            let name = d
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "images".into());
            let base = d.parent().unwrap_or(d);
            self.export.out = base.join(format!("{name}_yolo_dataset")).display().to_string();
        }
        self.export.open = true;
    }

    fn top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.add_space(3.0);
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 3.0;
                let has = self.dir.is_some();
                if act_button(ui, Act::Open, "Open", true)
                    .on_hover_text("Open a dataset folder (Ctrl+O)\nAll sub-folders are scanned. You can also drop a folder or image on the window.")
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
                if act_button(ui, Act::Prev, "", has).on_hover_text("Previous image (A or ←)").clicked() {
                    self.goto(ctx, -1);
                }
                if act_button(ui, Act::Next, "", has).on_hover_text("Next image (D or →)").clicked() {
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
                ui.label("Format:");
                let before = self.project.export_mode;
                egui::ComboBox::from_id_salt("mode")
                    .selected_text(match self.project.export_mode {
                        ExportMode::Segment => "Segmentation",
                        ExportMode::Detect => "Detection",
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
                    });
                if before != self.project.export_mode {
                    self.labels_stale = true; // every label file changes format
                    self.touch();
                }
                ui.label("Line width:")
                    .on_hover_text("Polylines are exported as a thin polygon band of this width (image pixels). 0 = zero-width.");
                if ui
                    .add(
                        egui::DragValue::new(&mut self.project.polyline_width)
                            .range(0.0..=500.0)
                            .speed(0.5)
                            .suffix(" px"),
                    )
                    .changed()
                {
                    // only images with polylines are affected
                    for (name, a) in &self.project.images {
                        if a.shapes.iter().any(|s| s.kind == Kind::Polyline) {
                            self.stale.insert(name.clone());
                        }
                    }
                    self.touch();
                }
                ui.separator();
                if act_button(ui, Act::About, "", true).on_hover_text("About YOLO Annotator").clicked() {
                    self.about_open = true;
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
            let img = egui::ColorImage::from_rgba_unmultiplied([96, 96], &icon::rgba(96));
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
                        ui.add(egui::Label::new(egui::RichText::new(include_str!("../LICENSE")).small().monospace()).wrap());
                    });
                });
                ui.label(
                    egui::RichText::new("Built with Rust, egui/eframe, image and rfd (MIT / Apache-2.0).")
                        .small()
                        .weak(),
                );
            });
    }

    fn status_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
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
                let hint = match self.tool {
                    Tool::Select => "Click: select · drag point/shape: edit · dbl-click edge: add point · right-click point: remove · Del: delete · drag empty: pan",
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
                let Some(root) = self.dir.clone() else {
                    ui.weak("No folder open.\nUse Open (Ctrl+O) or drag a folder here.");
                    return;
                };
                let root_name = root
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| root.display().to_string());
                ui.label(egui::RichText::new(format!("Folder: {root_name}")).strong())
                    .on_hover_text(root.display().to_string());
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
                                        file_row(ui, i, files, project, cur, false, &mut goto);
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
                            file_row(ui, idx[k], files, project, cur, true, &mut goto);
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
                            self.project.classes.push(self.new_class.trim().to_string());
                            self.class = self.project.classes.len() - 1;
                            self.new_class.clear();
                            self.touch();
                        }
                    });
                    if self.class < self.project.classes.len() {
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
                    self.ai_panel(ui);

                    ui.separator();
                    egui::CollapsingHeader::new("Help").default_open(false).show(ui, |ui| {
                        ui.label(
                            "Mouse\n\
                             • Wheel: zoom   • Middle-drag / Space+drag: pan\n\
                             • Box: drag a rectangle\n\
                             • Polygon: click points, click the first point (or double/right-click) to close\n\
                             • Polyline: click points, double/right-click to finish\n\
                             • Select: drag points or whole shapes; double-click an edge to add a point; right-click a point to remove it\n\
                             • In the drawing tools the shape you just drew stays selected: drag its handles (box corners or edge midpoints) or its centre grip to adjust. Box tool: click a shape to select it; Polygon / Polyline: right-click a shape. Del deletes the selection, Esc deselects\n\n\
                             Keys\n\
                             V/B/P/L tools · Q toggle Selector ↔ last annotation tool · A/D or ←/→ prev/next image · 1-9 class · Del delete · Ctrl+Z undo · Ctrl+O open · Ctrl+S save · Ctrl+E export · F fit · Esc cancel\n\n\
                             Saving\n\
                             Everything autosaves to annotations.json in the image folder, and YOLO .txt files are written to the 'labels' folder next to it (images → labels). Use 'Export...' to build a train/val split with data.yaml.",
                        );
                    });
                });
            });
    }

    fn ai_panel(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("AI assist").default_open(true).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("Status:");
                let mut s = self.ai_status.clone();
                if self.last_encode_ms > 0 && self.ai.is_some() {
                    s.push_str(&format!(" · {} ms/image", self.last_encode_ms));
                }
                ui.weak(s);
            });
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
                ui.label("Engine:");
                egui::ComboBox::from_id_salt("ai_engine")
                    .selected_text(self.ai_settings.engine.label())
                    .show_ui(ui, |ui| {
                        for e in [ai::Engine::Auto, ai::Engine::Cuda, ai::Engine::DirectML, ai::Engine::Cpu] {
                            ui.selectable_value(&mut self.ai_settings.engine, e, e.label());
                        }
                    })
                    .response
                    .on_hover_text("Auto: CUDA if an NVIDIA GPU and the CUDA pack are present, else DirectML (any GPU), else CPU.\nA change takes effect after restarting the app.");
                ui.end_row();
                ui.label("Detail:");
                ui.add(egui::Slider::new(&mut self.ai_settings.detail, 0.3..=4.0).text("px"))
                    .on_hover_text("Polygon simplification: smaller = more points, follows the edge closer");
                ui.end_row();
            });
            if self.ai_models.is_empty() {
                ui.colored_label(Color32::from_rgb(230, 160, 60), "No SAM models found (models folder next to the exe).");
            }

            ui.add_space(4.0);
            ui.label(egui::RichText::new("Auto-label · your YOLO model").strong());
            ui.horizontal(|ui| {
                let name = Path::new(&self.ai_settings.yolo)
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "none".into());
                ui.label(name).on_hover_text(&self.ai_settings.yolo);
                if ui.small_button("Browse...").on_hover_text("An Ultralytics model exported with: yolo export model=best.pt format=onnx").clicked() {
                    if let Some(f) = rfd::FileDialog::new().add_filter("ONNX model", &["onnx"]).pick_file() {
                        self.ai_settings.yolo = f.display().to_string();
                    }
                }
                if !self.ai_settings.yolo.is_empty() && ui.small_button("✖").clicked() {
                    self.ai_settings.yolo.clear();
                }
            });
            ui.add(egui::Slider::new(&mut self.ai_settings.conf, 0.05..=0.95).text("confidence"));
            if self.ai_settings != before {
                let sam_changed = self.ai_settings.sam != before.sam;
                self.ai_settings_changed(sam_changed);
            }
            let has_model = !self.ai_settings.yolo.is_empty();
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(has_model && self.loaded.is_some(), egui::Button::new("Auto-label image"))
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
                let d = b.done.load(Ordering::Relaxed);
                ui.add(egui::ProgressBar::new(d as f32 / b.total.max(1) as f32).text(format!("{d} / {} · {} labeled", b.total, b.added)));
                if ui.button("Cancel").clicked() {
                    b.cancel.store(true, Ordering::Relaxed);
                }
                self.ctx.request_repaint_after(std::time::Duration::from_millis(200));
            }
        });
    }

    fn export_window(&mut self, ctx: &egui::Context) {
        // collect the result of a running export (even if its window was closed)
        if let Some(job) = &self.export_job {
            match job.rx.try_recv() {
                Ok(res) => {
                    self.status = match res {
                        Ok(st) => {
                            self.export.open = false;
                            format!(
                                "Exported {} train + {} val + {} test images to {}",
                                st.train,
                                st.val,
                                st.test,
                                job.out.display()
                            )
                        }
                        Err(e) => format!("Export failed: {e}"),
                    };
                    self.export_job = None;
                }
                Err(mpsc::TryRecvError::Empty) => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.status = "Export failed: the export thread stopped unexpectedly.".into();
                    self.export_job = None;
                }
            }
        }
        if !self.export.open {
            return;
        }
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
                    ui.label("Output folder:");
                    let r = ui.add(egui::TextEdit::singleline(&mut self.export.out).desired_width(320.0));
                    self.typing |= r.has_focus();
                    if ui.button("Browse...").clicked() {
                        if let Some(d) = rfd::FileDialog::new().pick_folder() {
                            self.export.out = d.display().to_string();
                        }
                    }
                });
                let has_split = self.files.iter().any(|f| folder_split(f).is_some());
                ui.add_enabled(
                    has_split,
                    egui::Checkbox::new(
                        &mut self.export.keep_split,
                        "Keep existing train / val / test folders",
                    ),
                )
                .on_hover_text("Images inside folders named train, val/valid or test keep that split. Other images are split randomly.");
                ui.add(
                    egui::Slider::new(&mut self.export.val_ratio, 0.0..=0.5)
                        .text("validation fraction (random split)"),
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
                if let Some(job) = &self.export_job {
                    let (d, t) = (job.done.load(Ordering::Relaxed), job.total.load(Ordering::Relaxed));
                    let frac = if t > 0 { d as f32 / t as f32 } else { 0.0 };
                    ui.add(egui::ProgressBar::new(frac).text(format!("copying {d} / {t}")));
                } else if ui.button("Export").clicked() {
                    run = true;
                }
            });
        if run && self.export_job.is_none() {
            self.save_current(true);
            if let Some(dir) = self.dir.clone() {
                let out = PathBuf::from(self.export.out.trim());
                // copying thousands of images takes a while: do it off the UI thread
                let (project, files) = (self.project.clone(), self.files.clone());
                let (vr, inc, keep) = (
                    self.export.val_ratio,
                    self.export.include_unlabeled,
                    self.export.keep_split,
                );
                let done = Arc::new(AtomicUsize::new(0));
                let total = Arc::new(AtomicUsize::new(0));
                let (d2, t2) = (done.clone(), total.clone());
                let (tx, rx) = mpsc::channel();
                let out2 = out.clone();
                let ctx2 = ctx.clone();
                std::thread::spawn(move || {
                    let res = export_dataset(&project, &dir, &files, &out2, vr, inc, keep, &|k, n| {
                        d2.store(k, Ordering::Relaxed);
                        t2.store(n, Ordering::Relaxed);
                    });
                    let _ = tx.send(res);
                    ctx2.request_repaint();
                });
                self.export_job = Some(ExportJob { done, total, rx, out });
                self.status = "Exporting...".into();
            }
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
    About,
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
        Act::About => {
            p.circle_stroke(r.center(), r.width() * 0.4, st);
            p.circle_filled(pt(0.5, 0.3), 1.8, c);
            line(vec![pt(0.5, 0.44), pt(0.5, 0.74)]);
        }
    }
}

/// One image row in the navigation pane.
fn file_row(
    ui: &mut egui::Ui,
    i: usize,
    files: &[String],
    project: &Project,
    cur: usize,
    full_path: bool,
    goto: &mut Option<usize>,
) {
    let rel = &files[i];
    let name = if full_path { rel.as_str() } else { rel.rsplit('/').next().unwrap_or(rel) };
    let text = match project.images.get(rel) {
        Some(a) if a.shapes.is_empty() => format!("✔ {name}  (bg)"),
        Some(a) => format!("✔ {name}  ({})", a.shapes.len()),
        None => format!("    {name}"),
    };
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
        let b = s.bounds();
        let anchor = to_s([b[0], b[1]]);
        let galley = p.layout_no_wrap(label.to_string(), FontId::proportional(12.0), Color32::BLACK);
        let size = galley.size() + Vec2::new(6.0, 2.0);
        let r = Rect::from_min_size(anchor - Vec2::new(0.0, size.y), size);
        p.rect_filled(r, 2.0, c);
        p.galley(r.min + Vec2::new(3.0, 1.0), galley, Color32::BLACK);
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

        let dropped: Option<PathBuf> =
            ctx.input(|i| i.raw.dropped_files.iter().find_map(|f| f.path.clone()));
        if let Some(p) = dropped {
            self.open_path(ctx, p);
        }

        self.poll_ai();
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

        // the current image's YOLO label file is written as soon as an edit is finished
        if self.label_dirty && matches!(self.drag, Drag::None) && self.drawing.is_empty() {
            self.write_current_label();
        }
        // the project file (annotations.json) a couple of seconds after the last change
        let now = ctx.input(|i| i.time);
        if self.dirty && matches!(self.drag, Drag::None) && now - self.last_save > 30.0 {
            self.save_current(false);
            self.last_save = now;
        }
        if self.dirty {
            ctx.request_repaint_after(std::time::Duration::from_secs(2));
        }
        if ctx.input(|i| i.viewport().close_requested()) {
            if !self.drawing.is_empty() {
                self.finish_drawing();
            }
            self.accept_magic();
            self.save_current(false);
        }
    }
}
