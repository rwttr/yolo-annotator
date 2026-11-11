//! Data model, persistence and Ultralytics YOLO import/export.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const PROJECT_FILE: &str = "annotations.json";
pub const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"];

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    BBox,
    Polygon,
    Polyline,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::BBox => "box",
            Kind::Polygon => "polygon",
            Kind::Polyline => "polyline",
        }
    }
}

/// One annotation. Points are in image pixel coordinates.
/// A box is stored as two points: [x0, y0] (top-left) and [x1, y1] (bottom-right).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Annot {
    pub class: usize,
    pub kind: Kind,
    pub pts: Vec<[f32; 2]>,
}

impl Annot {
    /// [x0, y0, x1, y1]
    pub fn bounds(&self) -> [f32; 4] {
        bounds_of(&self.pts)
    }

    pub fn normalize_box(&mut self) {
        if self.kind == Kind::BBox && self.pts.len() == 2 {
            let b = self.bounds();
            self.pts = vec![[b[0], b[1]], [b[2], b[3]]];
        }
    }

    /// The 4 corners of the bounding box: TL, TR, BR, BL.
    pub fn corners(&self) -> [[f32; 2]; 4] {
        let b = self.bounds();
        [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
    }

    /// Editable handles. Box: corners TL, TR, BR, BL (0-3), then edge midpoints
    /// top, right, bottom, left (4-7). Polygon / polyline: the vertices.
    pub fn handles(&self) -> Vec<[f32; 2]> {
        if self.kind == Kind::BBox {
            let b = self.bounds();
            let (cx, cy) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0);
            let mut h = self.corners().to_vec();
            h.extend([[cx, b[1]], [b[2], cy], [cx, b[3]], [b[0], cy]]);
            h
        } else {
            self.pts.clone()
        }
    }

    /// Centre of the bounding box (the "move" grip).
    pub fn center(&self) -> [f32; 2] {
        let b = self.bounds();
        [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]
    }

    pub fn set_handle(&mut self, idx: usize, p: [f32; 2]) {
        if self.kind == Kind::BBox {
            if self.pts.len() != 2 {
                return;
            }
            // which stored x / y each handle controls (None = unchanged)
            let (ix, iy) = match idx {
                0 => (Some(0), Some(0)),
                1 => (Some(1), Some(0)),
                2 => (Some(1), Some(1)),
                3 => (Some(0), Some(1)),
                4 => (None, Some(0)),
                5 => (Some(1), None),
                6 => (None, Some(1)),
                7 => (Some(0), None),
                _ => (None, None),
            };
            if let Some(i) = ix {
                self.pts[i][0] = p[0];
            }
            if let Some(i) = iy {
                self.pts[i][1] = p[1];
            }
        } else if let Some(q) = self.pts.get_mut(idx) {
            *q = p;
        }
    }
}

pub fn bounds_of(pts: &[[f32; 2]]) -> [f32; 4] {
    let mut b = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
    for p in pts {
        b[0] = b[0].min(p[0]);
        b[1] = b[1].min(p[1]);
        b[2] = b[2].max(p[0]);
        b[3] = b[3].max(p[1]);
    }
    b
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct ImageAnn {
    pub width: u32,
    pub height: u32,
    pub shapes: Vec<Annot>,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExportMode {
    /// Ultralytics segmentation format: `class x1 y1 x2 y2 ...` (boxes become 4-point polygons).
    Segment,
    /// Ultralytics detection format: `class cx cy w h` (polygons become their bounding boxes).
    Detect,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Project {
    pub classes: Vec<String>,
    pub export_mode: ExportMode,
    /// Polylines are exported as a thin polygon of this width (pixels). 0 = zero-width (out and back).
    pub polyline_width: f32,
    pub images: BTreeMap<String, ImageAnn>,
}

impl Default for Project {
    fn default() -> Self {
        Self {
            classes: vec!["object".into()],
            export_mode: ExportMode::Segment,
            polyline_width: 6.0,
            images: BTreeMap::new(),
        }
    }
}

pub enum LoadError {
    /// the file exists but cannot be read (locked by another program, no permission, ...)
    Unreadable(String),
    /// the file was read but is not valid; a backup copy was made
    Corrupt(String),
}

impl Project {
    /// Ok(None) = no project file yet.
    pub fn load(dir: &Path) -> Result<Option<Project>, LoadError> {
        let path = dir.join(PROJECT_FILE);
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                return Err(LoadError::Unreadable(format!(
                    "Cannot read {} ({e}). Is it open in another program or being synced? The dataset was not opened, so nothing gets overwritten.",
                    path.display()
                )))
            }
        };
        match serde_json::from_str(&text) {
            Ok(p) => Ok(Some(p)),
            Err(e) => {
                let backup = dir.join("annotations.corrupt.json");
                let _ = fs::copy(&path, &backup);
                Err(LoadError::Corrupt(format!(
                    "{PROJECT_FILE} is damaged ({e}). A copy was saved as {}; labels were re-read from the YOLO .txt files.",
                    backup.display()
                )))
            }
        }
    }

    pub fn save(&self, dir: &Path) -> Result<(), String> {
        let text = serde_json::to_string(self).map_err(|e| e.to_string())?;
        let tmp = dir.join(format!("{PROJECT_FILE}.tmp"));
        fs::write(&tmp, text).map_err(|e| format!("write {}: {e}", tmp.display()))?;
        fs::rename(&tmp, dir.join(PROJECT_FILE)).map_err(|e| format!("save project: {e}"))
    }
}

pub fn is_image(name: &str) -> bool {
    Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

pub fn stem(name: &str) -> String {
    Path::new(name)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| name.to_string())
}

/// Ultralytics finds labels by replacing the last `images` path component with `labels`
/// (case-sensitive). If the folder has no `images` component, labels go into `<dir>/labels`.
pub fn labels_dir_for(dir: &Path) -> PathBuf {
    let comps: Vec<_> = dir.components().collect();
    match comps
        .iter()
        .rposition(|c| c.as_os_str() == "images")
    {
        Some(idx) => comps
            .iter()
            .enumerate()
            .map(|(i, c)| {
                if i == idx {
                    PathBuf::from("labels")
                } else {
                    PathBuf::from(c.as_os_str())
                }
            })
            .collect(),
        None => dir.join("labels"),
    }
}

pub fn read_classes(labels_dir: &Path) -> Option<Vec<String>> {
    let text = fs::read_to_string(labels_dir.join("classes.txt")).ok()?;
    let v: Vec<String> = text
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    (!v.is_empty()).then_some(v)
}

/// Folders never scanned for images.
fn skip_dir(name: &str) -> bool {
    name.starts_with('.')
        || name.eq_ignore_ascii_case("labels")
        || name.eq_ignore_ascii_case("__pycache__")
        || name.eq_ignore_ascii_case("runs")
}

/// All images under `root` (recursively), as relative paths with `/` separators.
pub fn scan_images(root: &Path) -> Result<Vec<String>, String> {
    fn walk(dir: &Path, rel: &str, out: &mut Vec<String>) -> std::io::Result<()> {
        for e in fs::read_dir(dir)?.flatten() {
            let Ok(name) = e.file_name().into_string() else { continue };
            let Ok(ft) = e.file_type() else { continue };
            let r = if rel.is_empty() { name.clone() } else { format!("{rel}/{name}") };
            if ft.is_dir() {
                if !skip_dir(&name) {
                    let _ = walk(&e.path(), &r, out);
                }
            } else if ft.is_file() && is_image(&name) {
                out.push(r);
            }
        }
        Ok(())
    }
    let mut out = vec![];
    walk(root, "", &mut out).map_err(|e| format!("Cannot read {}: {e}", root.display()))?;
    // sort by folder, then name (case-insensitive)
    out.sort_by_key(|s| s.to_lowercase());
    Ok(out)
}

/// Where the YOLO label file of image `rel` (relative to `root`) lives.
pub fn label_path(root: &Path, rel: &str) -> PathBuf {
    let img = root.join(rel);
    let parent = img.parent().unwrap_or(root);
    labels_dir_for(parent).join(format!("{}.txt", stem(rel)))
}

/// Class names from an Ultralytics data.yaml (`names: [a, b]`, `names: {0: a}`,
/// or a block of `0: a` / `- a` lines).
pub fn parse_yaml_names(text: &str) -> Option<Vec<String>> {
    let unq = |s: &str| {
        let s = s.trim();
        let s = s.split(" #").next().unwrap_or(s).trim();
        s.trim_matches(|c| c == '\'' || c == '"').to_string()
    };
    let lines: Vec<&str> = text.lines().collect();
    let i = lines.iter().position(|l| l.starts_with("names:"))?;
    let rest = lines[i]["names:".len()..].trim();
    let mut names = vec![];
    if rest.starts_with('[') || rest.starts_with('{') {
        // inline list / map, possibly spanning lines
        let mut buf = rest.to_string();
        let close = if rest.starts_with('[') { ']' } else { '}' };
        let mut j = i + 1;
        while !buf.contains(close) && j < lines.len() {
            buf.push_str(lines[j]);
            j += 1;
        }
        let inner = buf.trim_start_matches(['[', '{']).split(close).next().unwrap_or("");
        for item in inner.split(',') {
            let item = item.trim();
            if item.is_empty() {
                continue;
            }
            // "0: name" in a map
            let v = match item.split_once(':') {
                Some((k, v)) if k.trim().parse::<usize>().is_ok() => v,
                _ => item,
            };
            names.push(unq(v));
        }
    } else {
        for l in &lines[i + 1..] {
            if l.trim().is_empty() || l.trim_start().starts_with('#') {
                continue;
            }
            if !l.starts_with(' ') && !l.starts_with('\t') && !l.starts_with('-') {
                break;
            }
            let t = l.trim();
            if let Some(v) = t.strip_prefix('-') {
                names.push(unq(v));
            } else if let Some((k, v)) = t.split_once(':') {
                if k.trim().parse::<usize>().is_ok() {
                    names.push(unq(v));
                }
            }
        }
    }
    (!names.is_empty()).then_some(names)
}

/// Look for class names next to / above a dataset folder: data.yaml (or any *.yaml with
/// `names:`), then labels/classes.txt.
pub fn find_classes(root: &Path) -> Option<Vec<String>> {
    for dir in [Some(root), root.parent()].into_iter().flatten() {
        let Ok(rd) = fs::read_dir(dir) else { continue };
        let mut yamls: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .map(|e| e.eq_ignore_ascii_case("yaml") || e.eq_ignore_ascii_case("yml"))
                    .unwrap_or(false)
            })
            .collect();
        yamls.sort_by_key(|p| !p.ends_with("data.yaml")); // prefer data.yaml
        for y in yamls {
            if let Some(n) = fs::read_to_string(&y).ok().as_deref().and_then(parse_yaml_names) {
                return Some(n);
            }
        }
    }
    read_classes(&labels_dir_for(root))
}

// ---------------------------------------------------------------- YOLO export

fn norm(v: f32, size: f32) -> f32 {
    (v / size).clamp(0.0, 1.0)
}

fn fmt_poly(class: usize, pts: &[[f32; 2]], w: f32, h: f32) -> String {
    let mut s = class.to_string();
    for p in pts {
        s.push_str(&format!(" {:.6} {:.6}", norm(p[0], w), norm(p[1], h)));
    }
    s
}

fn fmt_box(class: usize, b: [f32; 4], w: f32, h: f32) -> String {
    let x0 = b[0].clamp(0.0, w);
    let y0 = b[1].clamp(0.0, h);
    let x1 = b[2].clamp(0.0, w);
    let y1 = b[3].clamp(0.0, h);
    format!(
        "{} {:.6} {:.6} {:.6} {:.6}",
        class,
        (x0 + x1) / 2.0 / w,
        (y0 + y1) / 2.0 / h,
        (x1 - x0) / w,
        (y1 - y0) / h
    )
}

fn unit(v: [f32; 2]) -> [f32; 2] {
    let l = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if l < 1e-6 {
        [0.0, 0.0]
    } else {
        [v[0] / l, v[1] / l]
    }
}

/// Turn a polyline into a closed polygon outline of the given width (pixels).
pub fn buffer_polyline(pts: &[[f32; 2]], width: f32) -> Vec<[f32; 2]> {
    let mut clean: Vec<[f32; 2]> = Vec::with_capacity(pts.len());
    for p in pts {
        if clean.last().map_or(true, |q| dist(*q, *p) > 1e-3) {
            clean.push(*p);
        }
    }
    let pts = &clean[..];
    let n = pts.len();
    if n < 2 {
        return pts.to_vec();
    }
    if width <= 0.0 {
        // zero-width: go forward, come back
        let mut out = pts.to_vec();
        out.extend(pts[1..n - 1].iter().rev());
        return out;
    }
    let hw = width / 2.0;
    let seg = |i: usize| unit([pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]]);
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    for i in 0..n {
        let (dir, scale) = if i == 0 {
            (seg(0), hw)
        } else if i == n - 1 {
            (seg(n - 2), hw)
        } else {
            let a = seg(i - 1);
            let b = seg(i);
            let d = unit([a[0] + b[0], a[1] + b[1]]);
            let d = if d == [0.0, 0.0] { b } else { d };
            // miter: keep the band width constant at corners (limited for sharp turns)
            let cos = (d[0] * a[0] + d[1] * a[1]).max(0.35);
            (d, hw / cos)
        };
        let nrm = [-dir[1], dir[0]];
        let p = pts[i];
        left.push([p[0] + nrm[0] * scale, p[1] + nrm[1] * scale]);
        right.push([p[0] - nrm[0] * scale, p[1] - nrm[1] * scale]);
    }
    left.extend(right.into_iter().rev());
    left
}

/// Contents of the YOLO .txt label file for one image.
pub fn yolo_text(ann: &ImageAnn, mode: ExportMode, polyline_width: f32) -> String {
    let (w, h) = (ann.width as f32, ann.height as f32);
    if w <= 0.0 || h <= 0.0 {
        return String::new();
    }
    let mut lines = Vec::new();
    for s in &ann.shapes {
        let line = match (mode, s.kind) {
            (_, Kind::BBox) if s.pts.len() != 2 => continue,
            (_, Kind::Polygon) if s.pts.len() < 3 => continue,
            (_, Kind::Polyline) if s.pts.len() < 2 => continue,
            (ExportMode::Detect, Kind::Polyline) => {
                fmt_box(s.class, bounds_of(&buffer_polyline(&s.pts, polyline_width)), w, h)
            }
            (ExportMode::Detect, _) => fmt_box(s.class, s.bounds(), w, h),
            (ExportMode::Segment, Kind::BBox) => fmt_poly(s.class, &s.corners(), w, h),
            (ExportMode::Segment, Kind::Polygon) => fmt_poly(s.class, &s.pts, w, h),
            (ExportMode::Segment, Kind::Polyline) => {
                fmt_poly(s.class, &buffer_polyline(&s.pts, polyline_width), w, h)
            }
        };
        lines.push(line);
    }
    if lines.is_empty() {
        String::new()
    } else {
        lines.join("\n") + "\n"
    }
}

/// Parse an existing YOLO label file (detect or segment format) into annotations.
pub fn parse_yolo(text: &str, w: u32, h: u32) -> Vec<Annot> {
    let (w, h) = (w as f32, h as f32);
    let mut out = Vec::new();
    for line in text.lines() {
        let v: Vec<f32> = line
            .split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect();
        if v.is_empty() || v[0] < 0.0 || v.iter().any(|x| !x.is_finite()) {
            continue;
        }
        // slightly out-of-range values are common in third-party labels: keep them in the image
        let cx = |x: f32| x.clamp(0.0, w);
        let cy = |y: f32| y.clamp(0.0, h);
        let class = v[0] as usize;
        if v.len() == 5 {
            let (x, y, bw, bh) = (v[1] * w, v[2] * h, v[3] * w, v[4] * h);
            out.push(Annot {
                class,
                kind: Kind::BBox,
                pts: vec![
                    [cx(x - bw / 2.0), cy(y - bh / 2.0)],
                    [cx(x + bw / 2.0), cy(y + bh / 2.0)],
                ],
            });
        } else if v.len() >= 7 && v.len() % 2 == 1 {
            let pts: Vec<[f32; 2]> = v[1..].chunks(2).map(|c| [cx(c[0] * w), cy(c[1] * h)]).collect();
            // a 4-point axis-aligned rectangle was most likely a box exported in segment mode
            let is_rect = pts.len() == 4 && {
                let b = bounds_of(&pts);
                let eps = 0.75;
                pts.iter().all(|p| {
                    ((p[0] - b[0]).abs() < eps || (p[0] - b[2]).abs() < eps)
                        && ((p[1] - b[1]).abs() < eps || (p[1] - b[3]).abs() < eps)
                })
            };
            if is_rect {
                let b = bounds_of(&pts);
                out.push(Annot {
                    class,
                    kind: Kind::BBox,
                    pts: vec![[b[0], b[1]], [b[2], b[3]]],
                });
            } else {
                out.push(Annot {
                    class,
                    kind: Kind::Polygon,
                    pts,
                });
            }
        }
    }
    out
}

/// Write the YOLO label file of image `rel` (relative to the dataset root).
/// Returns the path written.
pub fn write_label_file(root: &Path, rel: &str, ann: &ImageAnn, p: &Project) -> Result<PathBuf, String> {
    let path = label_path(root, rel);
    if let Some(d) = path.parent() {
        fs::create_dir_all(d).map_err(|e| format!("create {}: {e}", d.display()))?;
    }
    fs::write(&path, yolo_text(ann, p.export_mode, p.polyline_width))
        .map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(path)
}

pub fn write_classes(root: &Path, p: &Project) -> Result<(), String> {
    let ld = labels_dir_for(root);
    fs::create_dir_all(&ld).map_err(|e| e.to_string())?;
    fs::write(ld.join("classes.txt"), p.classes.join("\n") + "\n").map_err(|e| e.to_string())
}

/// Set the image's real size. Existing pixel coordinates are rescaled so their
/// normalized (YOLO) positions are kept (e.g. labels imported with EXIF-unrotated size).
pub fn rescale(ann: &mut ImageAnn, w: u32, h: u32) {
    if ann.width > 0 && ann.height > 0 && (ann.width != w || ann.height != h) {
        let (sx, sy) = (w as f32 / ann.width as f32, h as f32 / ann.height as f32);
        for s in &mut ann.shapes {
            for q in &mut s.pts {
                q[0] *= sx;
                q[1] *= sy;
            }
        }
    }
    ann.width = w;
    ann.height = h;
}

const SPLIT_DIRS: &[(&str, &str)] = &[
    ("train", "train"),
    ("training", "train"),
    ("val", "val"),
    ("valid", "val"),
    ("validation", "val"),
    ("test", "test"),
    ("testing", "test"),
];

/// train / val / test taken from the image's folder names, if any.
pub fn folder_split(rel: &str) -> Option<&'static str> {
    let dirs: Vec<&str> = rel.split('/').collect();
    dirs[..dirs.len().saturating_sub(1)].iter().rev().find_map(|d| {
        SPLIT_DIRS
            .iter()
            .find(|(n, _)| d.eq_ignore_ascii_case(n))
            .map(|(_, s)| *s)
    })
}

/// Relative path inside out/images/<split>/: the original sub-folders minus
/// `images`, `labels` and split-folder names.
fn export_rel(rel: &str) -> String {
    let parts: Vec<&str> = rel.split('/').collect();
    let (file, dirs) = parts.split_last().unwrap();
    let mut keep: Vec<&str> = dirs
        .iter()
        .copied()
        .filter(|d| {
            !d.eq_ignore_ascii_case("images")
                && !d.eq_ignore_ascii_case("labels")
                && !SPLIT_DIRS.iter().any(|(n, _)| d.eq_ignore_ascii_case(n))
        })
        .collect();
    keep.push(file);
    keep.join("/")
}

#[derive(Debug, Default, PartialEq)]
pub struct ExportStats {
    pub train: usize,
    pub val: usize,
    pub test: usize,
}

/// Build a ready-to-train Ultralytics dataset:
/// out/images/{train,val[,test]}, out/labels/{...}, out/data.yaml
/// `keep_split`: images inside train/val/test folders keep that split; the rest are split
/// randomly with `val_ratio`.
pub fn export_dataset(
    p: &Project,
    root: &Path,
    files: &[String],
    out: &Path,
    val_ratio: f32,
    include_unlabeled: bool,
    keep_split: bool,
    progress: &dyn Fn(usize, usize),
) -> Result<ExportStats, String> {
    if out.as_os_str().is_empty() {
        return Err("Choose an output folder.".into());
    }
    if is_inside(out, root) {
        return Err(format!(
            "The output folder must be outside the dataset folder {} (otherwise images would be copied onto themselves or scanned twice).",
            root.display()
        ));
    }
    let items: Vec<&String> = files
        .iter()
        .filter(|f| include_unlabeled || p.images.contains_key(*f))
        .collect();
    if items.is_empty() {
        return Err("Nothing to export: no labeled images.".into());
    }
    prepare_output(out)?;
    fs::create_dir_all(out).map_err(|e| format!("create {}: {e}", out.display()))?;
    let _ = fs::write(out.join(EXPORT_MARKER), "exported by YOLO Annotator
");
    let mut assigned: Vec<(&String, &'static str)> = vec![];
    let mut pool: Vec<&String> = vec![];
    for f in items {
        match folder_split(f).filter(|_| keep_split) {
            Some(s) => assigned.push((f, s)),
            None => pool.push(f),
        }
    }
    // deterministic shuffle (xorshift) so re-exports give the same split
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    for i in (1..pool.len()).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        pool.swap(i, (s % (i as u64 + 1)) as usize);
    }
    let mut n_val = (pool.len() as f32 * val_ratio).round() as usize;
    if val_ratio > 0.0 && n_val == 0 && pool.len() > 1 {
        n_val = 1;
    }
    for (i, f) in pool.into_iter().enumerate() {
        assigned.push((f, if i < n_val { "val" } else { "train" }));
    }

    let mut stats = ExportStats::default();
    let mut used = std::collections::HashSet::new();
    let total = assigned.len();
    for (k, (f, split)) in assigned.into_iter().enumerate() {
        progress(k, total);
        // the label file name must be unique too (a.jpg and a.png would share a.txt)
        let key = |r: &str| format!("{split}/{}", without_ext(r).to_lowercase());
        let mut rel = export_rel(f);
        if !used.insert(key(&rel)) {
            // clash after flattening: build a unique name from the full original path
            let flat = f.replace('/', "_");
            rel = match flat.rsplit_once('.') {
                Some((stem, ext)) => format!("{stem}_{ext}.{ext}"),
                None => flat,
            };
            used.insert(key(&rel));
        }
        let dst = out.join("images").join(split).join(&rel);
        let lp = out
            .join("labels")
            .join(split)
            .join(&rel)
            .with_extension("txt");
        for d in [dst.parent(), lp.parent()].into_iter().flatten() {
            fs::create_dir_all(d).map_err(|e| format!("create {}: {e}", d.display()))?;
        }
        fs::copy(root.join(f), &dst).map_err(|e| format!("copy {f}: {e}"))?;
        let text = p
            .images
            .get(f)
            .map(|a| yolo_text(a, p.export_mode, p.polyline_width))
            .unwrap_or_default();
        fs::write(&lp, text).map_err(|e| format!("write {}: {e}", lp.display()))?;
        match split {
            "train" => stats.train += 1,
            "val" => stats.val += 1,
            _ => stats.test += 1,
        }
    }
    let abs = fs::canonicalize(out).unwrap_or_else(|_| out.to_path_buf());
    let abs = abs.to_string_lossy().to_string();
    // canonicalize gives a \\?\ verbatim path on Windows; data.yaml wants a normal one
    let abs = abs.strip_prefix(r"\\?\").map(str::to_string).unwrap_or(abs);
    let mut yaml = format!(
        "# Generated by YOLO Annotator ({} format)\npath: '{}'\ntrain: images/train\nval: images/{}\n",
        match p.export_mode {
            ExportMode::Segment => "segment",
            ExportMode::Detect => "detect",
        },
        abs.replace('\'', "''"),
        if stats.val > 0 { "val" } else { "train" }
    );
    if stats.test > 0 {
        yaml.push_str("test: images/test\n");
    }
    yaml.push_str("\nnames:\n");
    for (i, n) in p.classes.iter().enumerate() {
        yaml.push_str(&format!("  {i}: '{}'\n", n.replace('\'', "''")));
    }
    fs::write(out.join("data.yaml"), yaml).map_err(|e| format!("write data.yaml: {e}"))?;
    Ok(stats)
}

fn without_ext(rel: &str) -> &str {
    match rel.rfind('.') {
        Some(i) if i > rel.rfind('/').map_or(0, |j| j + 1) => &rel[..i],
        _ => rel,
    }
}

/// The output folder must be empty, or a previous export of this tool (which is then
/// replaced, so images moved between train/val by a re-export do not linger).
fn prepare_output(out: &Path) -> Result<(), String> {
    let Ok(mut rd) = fs::read_dir(out) else { return Ok(()) }; // does not exist yet
    if rd.next().is_none() {
        return Ok(());
    }
    let ours = out.join(EXPORT_MARKER).exists()
        || fs::read_to_string(out.join("data.yaml"))
            .map(|t| t.contains("Generated by YOLO Annotator"))
            .unwrap_or(false);
    if !ours {
        return Err(format!(
            "The output folder {} is not empty. Choose an empty or new folder.",
            out.display()
        ));
    }
    for sub in ["images", "labels"] {
        let d = out.join(sub);
        if d.exists() {
            fs::remove_dir_all(&d).map_err(|e| format!("clear {}: {e}", d.display()))?;
        }
    }
    let _ = fs::remove_file(out.join("data.yaml"));
    Ok(())
}

/// Written first into every export folder, so an interrupted export can be replaced later.
const EXPORT_MARKER: &str = ".yolo_annotator_export";

/// Pairs of images that would share one label file (e.g. `a.jpg` and `a.png`).
pub fn label_collisions(root: &Path, files: &[String]) -> Vec<(String, String)> {
    let mut seen: std::collections::HashMap<String, &String> = std::collections::HashMap::new();
    let mut out = vec![];
    for f in files {
        let k = label_path(root, f).to_string_lossy().to_lowercase();
        if let Some(first) = seen.get(&k) {
            out.push(((*first).clone(), f.clone()));
        } else {
            seen.insert(k, f);
        }
    }
    out
}

/// `a` is `b` or somewhere below it (case-insensitive, after making both absolute).
pub fn is_inside(a: &Path, b: &Path) -> bool {
    let norm = |p: &Path| -> Vec<String> {
        std::path::absolute(p)
            .unwrap_or_else(|_| p.to_path_buf())
            .components()
            .filter(|c| !matches!(c, std::path::Component::CurDir))
            .map(|c| c.as_os_str().to_string_lossy().to_lowercase())
            .collect()
    };
    let (a, b) = (norm(a), norm(b));
    a.len() >= b.len() && a[..b.len()] == b[..]
}

// ---------------------------------------------------------------- geometry helpers

pub fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

pub fn dist_to_segment(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l2 = dx * dx + dy * dy;
    if l2 < 1e-9 {
        return dist(p, a);
    }
    let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0);
    dist(p, [a[0] + t * dx, a[1] + t * dy])
}

pub fn point_in_polygon(p: [f32; 2], poly: &[[f32; 2]]) -> bool {
    let mut inside = false;
    let n = poly.len();
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        let (a, b) = (poly[i], poly[j]);
        if (a[1] > p[1]) != (b[1] > p[1])
            && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Segments of a shape's outline (closed for polygons).
pub fn edges(s: &Annot) -> Vec<(usize, [f32; 2], [f32; 2])> {
    let pts = &s.pts;
    let n = pts.len();
    let mut e: Vec<_> = (0..n.saturating_sub(1)).map(|i| (i, pts[i], pts[i + 1])).collect();
    if s.kind == Kind::Polygon && n >= 3 {
        e.push((n - 1, pts[n - 1], pts[0]));
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_box_and_polygon() {
        let ann = ImageAnn {
            width: 200,
            height: 100,
            shapes: vec![
                Annot { class: 0, kind: Kind::BBox, pts: vec![[20.0, 10.0], [60.0, 50.0]] },
                Annot { class: 1, kind: Kind::Polygon, pts: vec![[10.0, 10.0], [100.0, 20.0], [50.0, 90.0]] },
            ],
        };
        for mode in [ExportMode::Segment, ExportMode::Detect] {
            let t = yolo_text(&ann, mode, 4.0);
            let back = parse_yolo(&t, 200, 100);
            assert_eq!(back.len(), 2);
            assert_eq!(back[0].kind, Kind::BBox);
            let b = back[0].bounds();
            assert!((b[0] - 20.0).abs() < 0.01 && (b[3] - 50.0).abs() < 0.01);
        }
        let t = yolo_text(&ann, ExportMode::Detect, 0.0);
        assert!(t.starts_with("0 0.200000 0.300000 0.200000 0.400000"));
    }

    #[test]
    fn box_edge_handles() {
        let mut a = Annot { class: 0, kind: Kind::BBox, pts: vec![[10.0, 10.0], [50.0, 30.0]] };
        assert_eq!(a.handles().len(), 8);
        a.set_handle(5, [80.0, 999.0]); // right edge: only x changes
        a.set_handle(4, [999.0, 5.0]); // top edge: only y changes
        assert_eq!(a.bounds(), [10.0, 5.0, 80.0, 30.0]);
        a.set_handle(7, [90.0, 0.0]); // drag left edge past the right one
        a.normalize_box();
        assert_eq!(a.bounds(), [80.0, 5.0, 90.0, 30.0]);
        assert_eq!(a.pts[0], [80.0, 5.0]);
    }

    #[test]
    fn polyline_buffer() {
        // repeated points must not produce NaN / collapsed corners
        let d = buffer_polyline(&[[0.0, 0.0], [5.0, 0.0], [5.0, 0.0], [10.0, 0.0]], 4.0);
        assert!(d.iter().all(|p| p[0].is_finite() && p[1].is_finite()));
        assert!(d.iter().all(|p| (p[1].abs() - 2.0).abs() < 1e-3));
        let b = buffer_polyline(&[[0.0, 0.0], [10.0, 0.0]], 4.0);
        assert_eq!(b.len(), 4);
        assert!((b[0][1] - 2.0).abs() < 1e-4 && (b[3][1] + 2.0).abs() < 1e-4);
    }

    #[test]
    fn export_split() {
        let root = std::env::temp_dir().join(format!("yolo_annot_test_{}", std::process::id()));
        let img_dir = root.join("images");
        fs::create_dir_all(&img_dir).unwrap();
        let mut p = Project::default();
        let mut files = vec![];
        for i in 0..5 {
            let f = format!("im{i}.jpg");
            fs::write(img_dir.join(&f), b"x").unwrap();
            p.images.insert(f.clone(), ImageAnn { width: 10, height: 10, shapes: vec![
                Annot { class: 0, kind: Kind::BBox, pts: vec![[1.0, 1.0], [5.0, 5.0]] }] });
            files.push(f);
        }
        files.push("unlabeled.jpg".into());
        let out = root.join("out");
        let st = export_dataset(&p, &img_dir, &files, &out, 0.2, false, true, &|_, _| {}).unwrap();
        assert_eq!((st.train, st.val), (4, 1));
        assert_eq!(fs::read_dir(out.join("labels/train")).unwrap().count(), 4);
        let yaml = fs::read_to_string(out.join("data.yaml")).unwrap();
        assert!(yaml.contains("val: images/val") && yaml.contains("0: 'object'"));
        assert!(!yaml.contains(r"\\?\"), "verbatim prefix leaked into data.yaml: {yaml}");
        // refuse an output folder inside the dataset
        assert!(export_dataset(&p, &img_dir, &files, &img_dir.join("x"), 0.2, false, true, &|_, _| {}).is_err());
        assert!(is_inside(Path::new(r"C:\A\b\c"), Path::new(r"c:\a\B")));
        assert!(!is_inside(Path::new(r"C:\A\bc"), Path::new(r"C:\A\b")));
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn dataset_dir() {
        let root = std::env::temp_dir().join(format!("yolo_annot_ds_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        for d in ["images/train", "images/val/sub", "labels/train", ".hidden"] {
            fs::create_dir_all(root.join(d)).unwrap();
        }
        for f in ["images/train/a.jpg", "images/train/b.png", "images/val/sub/a.jpg",
                  "labels/train/x.jpg", ".hidden/h.jpg", "notes.txt"] {
            fs::write(root.join(f), b"x").unwrap();
        }
        let files = scan_images(&root).unwrap();
        assert_eq!(files, vec!["images/train/a.jpg", "images/train/b.png", "images/val/sub/a.jpg"]);
        assert_eq!(label_path(&root, "images/val/sub/a.jpg"), root.join("labels/val/sub/a.txt"));
        assert_eq!(folder_split("images/val/sub/a.jpg"), Some("val"));
        assert_eq!(folder_split("cats/a.jpg"), None);

        let mut p = Project::default();
        for f in &files {
            p.images.insert(f.clone(), ImageAnn { width: 10, height: 10, shapes: vec![] });
        }
        let out = std::env::temp_dir().join(format!("yolo_annot_ds_out_{}", std::process::id()));
        let _ = fs::remove_dir_all(&out);
        let st = export_dataset(&p, &root, &files, &out, 0.5, false, true, &|_, _| {}).unwrap();
        assert_eq!(st, ExportStats { train: 2, val: 1, test: 0 });
        assert!(out.join("images/val/sub/a.jpg").exists());
        assert!(out.join("labels/val/sub/a.txt").exists());
        assert!(out.join("images/train/b.png").exists() && out.join("labels/train/b.txt").exists());
        fs::remove_dir_all(&out).unwrap();
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn robust_import_and_export() {
        // out-of-range and NaN values
        let a = parse_yolo("0 0.5 0.5 1.2 1.2\n1 nan 0.1 0.2 0.3\n2 -0.1 0 1.1 0 1.1 1 -0.1 1 0.5 1.05\n", 100, 50);
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].bounds(), [0.0, 0.0, 100.0, 50.0]);
        assert!(a[1].pts.iter().all(|p| (0.0..=100.0).contains(&p[0]) && (0.0..=50.0).contains(&p[1])));

        let root = std::env::temp_dir().join(format!("yolo_annot_rx_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("ds/images")).unwrap();
        for f in ["a.jpg", "a.png"] {
            fs::write(root.join("ds/images").join(f), b"x").unwrap();
        }
        let ds = root.join("ds");
        let files = scan_images(&ds).unwrap();
        assert_eq!(label_collisions(&ds, &files).len(), 1);
        let mut p = Project::default();
        for f in &files {
            p.images.insert(f.clone(), ImageAnn { width: 10, height: 10, shapes: vec![] });
        }
        let out = root.join("out");
        export_dataset(&p, &ds, &files, &out, 0.0, false, true, &|_, _| {}).unwrap();
        let n = fs::read_dir(out.join("labels/train")).unwrap().count();
        assert_eq!(n, 2, "a.jpg and a.png must get separate label files");
        // re-export replaces the previous export
        fs::write(out.join("images/train/stale.jpg"), b"x").unwrap();
        export_dataset(&p, &ds, &files, &out, 0.0, false, true, &|_, _| {}).unwrap();
        assert!(!out.join("images/train/stale.jpg").exists());
        // a foreign non-empty folder is refused
        let other = root.join("other");
        fs::create_dir_all(&other).unwrap();
        fs::write(other.join("keep.txt"), b"x").unwrap();
        assert!(export_dataset(&p, &ds, &files, &other, 0.0, false, true, &|_, _| {}).is_err());
        assert!(other.join("keep.txt").exists());
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn yaml_names() {
        let a = "path: x
names:
  0: person
  1: 'traffic light'  # c
nc: 2
";
        assert_eq!(parse_yaml_names(a).unwrap(), vec!["person", "traffic light"]);
        let b = "nc: 2
names: ['cat', \"dog\"]
";
        assert_eq!(parse_yaml_names(b).unwrap(), vec!["cat", "dog"]);
        let c = "names:
- lane
- arrow
train: a
";
        assert_eq!(parse_yaml_names(c).unwrap(), vec!["lane", "arrow"]);
        let d = "names: {0: a, 1: b}
";
        assert_eq!(parse_yaml_names(d).unwrap(), vec!["a", "b"]);
    }

    #[test]
    fn labels_dir() {
        assert_eq!(
            labels_dir_for(Path::new(r"C:\data\images\train")),
            PathBuf::from(r"C:\data\labels\train")
        );
        assert_eq!(labels_dir_for(Path::new(r"C:\pics")), PathBuf::from(r"C:\pics\labels"));
    }
}
