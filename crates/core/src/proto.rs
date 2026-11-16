//! Messages and settings shared by the UI, the desktop store, the lab server and the web client.
//! Everything here is plain serde data, so it travels over the WebSocket as JSON unchanged.

use crate::model::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

// ------------------------------------------------------------------ AI settings / results

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug, Default)]
pub enum Engine {
    #[default]
    Auto,
    Cuda,
    DirectML,
    Cpu,
}

impl Engine {
    pub fn label(self) -> &'static str {
        match self {
            Engine::Auto => "Auto (best available)",
            Engine::Cuda => "CUDA (NVIDIA)",
            Engine::DirectML => "DirectML (any GPU)",
            Engine::Cpu => "CPU",
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
#[serde(default)]
pub struct AiSettings {
    pub engine: Engine,
    /// "auto" or a folder name under models\ (e.g. "sam2.1-small")
    pub sam: String,
    /// YOLO .onnx model for auto-labelling ("" = none). Desktop: a path; lab server: a file
    /// name in the server's models folder.
    pub yolo: String,
    pub conf: f32,
    pub iou: f32,
    /// polygon simplification tolerance (pixels of the AI image)
    pub detail: f32,
}

impl Default for AiSettings {
    fn default() -> Self {
        AiSettings { engine: Engine::Auto, sam: "auto".into(), yolo: String::new(), conf: 0.35, iou: 0.6, detail: 2.5 }
    }
}

impl AiSettings {
    fn file() -> Option<PathBuf> {
        std::env::var_os("APPDATA").map(|a| PathBuf::from(a).join("YOLO Annotator").join("settings.json"))
    }
    /// Desktop: settings of this Windows user (no-op elsewhere).
    pub fn load() -> AiSettings {
        Self::file()
            .and_then(|f| std::fs::read_to_string(f).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }
    pub fn save(&self) {
        if let Some(f) = Self::file() {
            let _ = std::fs::create_dir_all(f.parent().unwrap());
            let _ = std::fs::write(f, serde_json::to_string_pretty(self).unwrap_or_default());
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Detection {
    pub class_id: usize,
    pub class_name: Option<String>,
    pub score: f32,
    pub bbox: [f32; 4],
    pub polygon: Option<Vec<[f32; 2]>>,
}

/// AI request from the UI. Images are named by their key (the image's dataset-relative path);
/// the store (desktop) or the server decodes the image itself.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum AiReq {
    Settings(AiSettings),
    /// prepare SAM for this image (no-op if it already is)
    Encode { key: String },
    Segment { key: String, seq: u64, points: Vec<([f32; 2], bool)>, bbox: Option<[f32; 4]>, polygon: bool },
    /// run the YOLO model on one image
    Detect { key: String },
    /// run the YOLO model on these images (unlabeled ones)
    Batch { id: u64, keys: Vec<String> },
    CancelBatch { id: u64 },
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum AiResp {
    /// engine / model description for the status bar
    Ready(String),
    /// SAM cannot be used (model missing / failed to load)
    Unavailable(String),
    /// the AI worker crashed; nothing more will be answered
    Crashed,
    Info(String),
    Error(String),
    Encoded { key: String, ms: u64 },
    NeedEncode(String),
    EncodeFailed { key: String, err: String },
    Mask { key: String, seq: u64, shape: Option<Vec<[f32; 2]>> },
    /// `batch`: id of the batch (desktop only: the lab server applies batch results itself)
    Detected { key: String, batch: Option<u64>, dets: Vec<Detection>, w: u32, h: u32 },
    BatchProgress { id: u64, done: usize, total: usize },
    BatchDone { id: u64, labeled: usize, errors: Vec<String>, cancelled: bool },
    /// models available: SAM folders, YOLO .onnx files (lab server: names in its models folder)
    Models { sam: Vec<String>, yolo: Vec<String> },
}

// ------------------------------------------------------------------ class / project operations

/// A change to the class list or project settings. In the lab version these are applied by
/// the server in one order and sent to every client, so all copies stay identical.
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub enum ClassOp {
    Add(String),
    Rename(usize, String),
    /// only allowed while no shape uses the class; higher class ids move down by one
    Delete(usize),
    Settings { export_mode: ExportMode, polyline_width: f32, new_frames: NewFrames },
}

impl Project {
    /// Apply `op`. Returns the images whose label files change.
    pub fn apply_op(&mut self, op: &ClassOp) -> Result<Vec<String>, String> {
        match op {
            ClassOp::Add(name) => {
                let name = name.trim();
                if name.is_empty() {
                    return Err("empty class name".into());
                }
                if !self.classes.iter().any(|c| c == name) {
                    self.classes.push(name.to_string());
                }
                Ok(vec![])
            }
            ClassOp::Rename(i, name) => {
                let Some(c) = self.classes.get_mut(*i) else { return Err(format!("no class {i}")) };
                let name = name.trim();
                *c = if name.is_empty() { format!("class_{i}") } else { name.to_string() };
                Ok(vec![])
            }
            ClassOp::Delete(c) => {
                let c = *c;
                if c >= self.classes.len() {
                    return Err(format!("no class {c}"));
                }
                let used: usize =
                    self.images.values().map(|a| a.shapes.iter().filter(|s| s.class == c).count()).sum();
                if used > 0 {
                    return Err(format!(
                        "Class '{}' is used by {used} objects; reassign or delete them first.",
                        self.classes[c]
                    ));
                }
                self.classes.remove(c);
                let mut changed = vec![];
                for (name, a) in self.images.iter_mut() {
                    let mut ch = false;
                    for s in &mut a.shapes {
                        if s.class > c {
                            s.class -= 1;
                            ch = true;
                        }
                    }
                    if ch {
                        changed.push(name.clone());
                    }
                }
                Ok(changed)
            }
            ClassOp::Settings { export_mode, polyline_width, new_frames } => {
                let mut changed = vec![];
                if *export_mode != self.export_mode {
                    changed = self.images.keys().cloned().collect();
                } else if (*polyline_width - self.polyline_width).abs() > f32::EPSILON {
                    changed = self
                        .images
                        .iter()
                        .filter(|(_, a)| a.shapes.iter().any(|s| s.kind == Kind::Polyline))
                        .map(|(n, _)| n.clone())
                        .collect();
                }
                self.export_mode = *export_mode;
                self.polyline_width = *polyline_width;
                self.new_frames = *new_frames;
                Ok(changed)
            }
        }
    }

    /// Make sure every class index used by a shape has a name.
    pub fn ensure_classes(&mut self) {
        let maxc = self.images.values().flat_map(|a| a.shapes.iter().map(|s| s.class)).max();
        if let Some(maxc) = maxc {
            while self.classes.len() <= maxc {
                let n = self.classes.len();
                self.classes.push(format!("class_{n}"));
            }
        }
    }

    /// True while the dataset only has the placeholder class "object" and no shapes: then a
    /// YOLO model's own class names replace it.
    fn placeholder_only(&self) -> bool {
        self.classes.len() == 1
            && self.classes[0] == "object"
            && self.images.values().all(|a| a.shapes.is_empty())
    }

    /// Class names of `dets` that are not in the project yet (in order, no duplicates).
    pub fn missing_classes(&self, dets: &[Detection]) -> Vec<String> {
        let mut out: Vec<String> = vec![];
        for n in dets.iter().filter_map(|d| d.class_name.as_ref()) {
            let known = self.classes.iter().any(|c| c.eq_ignore_ascii_case(n))
                || out.iter().any(|c| c.eq_ignore_ascii_case(n));
            if !known {
                out.push(n.clone());
            }
        }
        out
    }

    /// YOLO detections -> shapes in the project's classes (unknown names are added).
    pub fn detections_to_shapes(&mut self, dets: &[Detection], w: u32, h: u32) -> Vec<Annot> {
        let seg = self.export_mode == ExportMode::Segment;
        if self.placeholder_only() && !dets.is_empty() && dets.iter().all(|d| d.class_name.is_some()) {
            self.classes.clear();
        }
        let mut out = vec![];
        for d in dets {
            let class = match &d.class_name {
                Some(n) => match self.classes.iter().position(|c| c.eq_ignore_ascii_case(n)) {
                    Some(i) => i,
                    None => {
                        self.classes.push(n.clone());
                        self.classes.len() - 1
                    }
                },
                None => {
                    while self.classes.len() <= d.class_id {
                        let i = self.classes.len();
                        self.classes.push(format!("class_{i}"));
                    }
                    d.class_id
                }
            };
            let cl = |p: [f32; 2]| [p[0].clamp(0.0, w as f32), p[1].clamp(0.0, h as f32)];
            out.push(match (&d.polygon, seg) {
                (Some(p), true) if p.len() >= 3 => {
                    Annot { class, kind: Kind::Polygon, pts: p.iter().map(|q| cl(*q)).collect() }
                }
                _ => Annot { class, kind: Kind::BBox, pts: vec![cl([d.bbox[0], d.bbox[1]]), cl([d.bbox[2], d.bbox[3]])] },
            });
        }
        out
    }
}

// ------------------------------------------------------------------ export

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ExportOpts {
    /// desktop: output folder; lab server: a plain name (no path), exported into the server's
    /// exports folder
    pub out: String,
    pub val_ratio: f32,
    pub include_unlabeled: bool,
    pub keep_split: bool,
    pub by_folder: bool,
}

// ------------------------------------------------------------------ lab server <-> web client

/// A dataset as opened: everything the UI needs to show it.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct OpenedData {
    /// shown in the UI (desktop: folder path; lab: dataset name)
    pub label: String,
    pub files: Vec<String>,
    pub project: Project,
    /// summary for the status bar
    pub status: String,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum ClientMsg {
    /// first message: who is this (any name; no passwords on the lab LAN)
    Hello { user: String },
    /// switch the server to another dataset folder. Only accepted from a browser running on
    /// the server machine itself (loopback address); lab members cannot change folders.
    AdminOpen { path: String },
    /// this client now shows `name` and wants to edit it (granted if nobody else is)
    View { name: String },
    /// label of `name` changed (None: back to never-labeled)
    PutImage { name: String, ann: Option<ImageAnn> },
    Op(ClassOp),
    Ai(AiReq),
    Export(ExportOpts),
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum ServerMsg {
    /// `admin`: this client runs on the server machine and may switch the dataset folder
    /// `user`: this client's name (made unique if it is already in use)
    Welcome { version: String, admin: bool, user: String },
    Opened(Result<OpenedData, String>),
    /// image -> user editing it, for the whole dataset
    Locks(BTreeMap<String, String>),
    /// someone (or the server: auto-label) changed an image
    ImageChanged { name: String, ann: Option<ImageAnn>, by: String },
    /// a class / settings change, applied in this order by everyone (including its sender)
    Op { op: ClassOp, by: String },
    OpFailed(String),
    /// result of a PutImage of this client
    Saved { name: String, result: Result<String, String> },
    Ai(AiResp),
    ExportProgress { done: usize, total: usize },
    ExportDone(Result<String, String>),
    /// users connected to this dataset
    Users(Vec<String>),
    Status(String),
}
