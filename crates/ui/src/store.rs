//! Where the UI's data lives: local folders (desktop app) or the lab server (web client).
//! The UI never touches files or the network itself; it calls a `Store` and reacts to the
//! events it returns from `poll`.

use annot_core::model::*;
use annot_core::proto::*;
use std::collections::BTreeMap;

/// A decoded image, ready for a texture.
pub struct ImageData {
    /// RGBA pixels of size `size` (at most MAX_TEX per side)
    pub rgba: Vec<u8>,
    pub size: [usize; 2],
    /// full image size (annotation coordinates)
    pub w: u32,
    pub h: u32,
    /// desktop: the image's YOLO label file, if it appeared after the dataset was opened
    pub label_text: Option<String>,
}

pub enum StoreEvent {
    /// a dataset was opened (`dirty`: labels were imported, save the project soon)
    Opened(Result<OpenedData, String>, bool),
    Image(String, Result<ImageData, String>),
    Ai(AiResp),
    ExportProgress { done: usize, total: usize },
    /// Ok: where the export went
    ExportDone(Result<String, String>),
    // ---- lab server only
    Connected(bool),
    /// `user`: the name the server gave us (made unique if two people use the same one)
    Welcome { admin: bool, user: String },
    /// image -> user editing it
    Locks(BTreeMap<String, String>),
    ImageChanged { name: String, ann: Option<ImageAnn>, by: String },
    /// class / settings change, to apply to the local copy of the project
    Op { op: ClassOp, by: String },
    OpFailed(String),
    /// the server stored (or failed to store) a label of ours
    Saved { name: String, result: Result<String, String> },
    Users(Vec<String>),
    Status(String),
}

pub trait Store {
    /// true for the lab server (shared data, locks, server-side files)
    fn is_lab(&self) -> bool;
    /// desktop: open a folder. Lab: switch the server's dataset (only from the server machine).
    fn open(&mut self, path: &str);
    /// Load image `name`; `next` is the image the user will probably open next (prefetch).
    fn load_image(&mut self, name: &str, next: Option<&str>);
    /// Write (or with None: remove) the YOLO label of `name`. Desktop: returns where it went.
    /// Lab: sent to the server (Ok right away; failures come back as `Saved`).
    fn write_label(&mut self, name: &str, ann: Option<&ImageAnn>, project: &Project) -> Result<String, String>;
    /// Desktop: write annotations.json + classes.txt. Lab: nothing (the server keeps them).
    fn save_project(&mut self, project: &Project) -> Result<String, String>;
    /// Lab: send a class / settings change; it is applied when the server echoes it.
    fn op(&mut self, _op: ClassOp) {}
    fn ai(&mut self, req: AiReq);
    fn export(&mut self, opts: ExportOpts);
    fn poll(&mut self) -> Vec<StoreEvent>;
    /// Lab: the user's name, sent to the server.
    fn hello(&mut self, _user: &str) {}
}
