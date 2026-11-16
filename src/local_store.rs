//! The desktop app's store: datasets are local folders, the AI runs in this process.

use annot_ai as ai;
use annot_core::dataset::open_dataset;
use annot_core::imageio::{decode_path, display_image, DisplayImage};
use annot_core::model::*;
use annot_core::proto::*;
use annot_ui::store::{ImageData, Store, StoreEvent};
use eframe::egui;
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

type Decoded = Result<(DisplayImage, ai::AiImage), String>;

/// Background decoding of the image the user will most likely open next.
struct Prefetch {
    name: String,
    rx: mpsc::Receiver<Decoded>,
}

struct Batch {
    id: u64,
    cancel: Arc<AtomicBool>,
    done: Arc<AtomicUsize>,
    total: usize,
    reported: usize,
}

struct Export {
    done: Arc<AtomicUsize>,
    total: Arc<AtomicUsize>,
    rx: mpsc::Receiver<Result<ExportStats, String>>,
    out: PathBuf,
}

pub struct LocalStore {
    root: Option<PathBuf>,
    events: VecDeque<StoreEvent>,
    ctx: egui::Context,
    prefetch: Option<Prefetch>,
    ai: Option<ai::AiHandle>,
    /// AI copies of recently shown images (for SAM / YOLO requests)
    ai_images: VecDeque<(String, ai::AiImage)>,
    batch: Option<Batch>,
    export: Option<Export>,
}

fn decode(path: &Path) -> Decoded {
    let img = decode_path(path)?;
    Ok((display_image(&img), ai::ai_image(&img)))
}

impl LocalStore {
    pub fn new(ctx: egui::Context) -> Self {
        LocalStore {
            root: None,
            events: VecDeque::new(),
            ctx,
            prefetch: None,
            ai: None,
            ai_images: VecDeque::new(),
            batch: None,
            export: None,
        }
    }

    /// Worker cache key: dataset folder + image (two datasets may share relative names).
    fn key(&self, name: &str) -> String {
        format!("{}|{name}", self.root.as_deref().map(|d| d.display().to_string()).unwrap_or_default())
    }

    /// The image name of a worker key of the open dataset (None: another dataset's).
    fn name_of(&self, key: &str) -> Option<String> {
        let prefix = self.key("");
        key.strip_prefix(&prefix).map(str::to_string)
    }

    fn ai_image(&self, name: &str) -> Option<ai::AiImage> {
        self.ai_images.iter().find(|(n, _)| n == name).map(|(_, i)| i.clone())
    }

    fn remember_ai_image(&mut self, name: &str, img: ai::AiImage) {
        self.ai_images.retain(|(n, _)| n != name);
        self.ai_images.push_back((name.to_string(), img));
        while self.ai_images.len() > 4 {
            self.ai_images.pop_front();
        }
    }

    /// Send a request that needs the image; decode it in the background if not at hand.
    fn with_image(&mut self, name: &str, make: impl FnOnce(String, ai::AiImage) -> ai::Req + Send + 'static) {
        let (Some(a), Some(root)) = (&self.ai, &self.root) else { return };
        let key = self.key(name);
        match self.ai_image(name) {
            Some(img) => {
                let _ = a.tx.send(make(key, img));
            }
            None => {
                let (tx, path) = (a.tx.clone(), root.join(name));
                std::thread::spawn(move || {
                    if let Ok(img) = ai::decode_ai(&path) {
                        let _ = tx.send(make(key, img));
                    }
                });
            }
        }
    }

    fn start_prefetch(&mut self, name: &str) {
        let Some(root) = &self.root else { return };
        let path = root.join(name);
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(decode(&path));
        });
        self.prefetch = Some(Prefetch { name: name.to_string(), rx });
    }

    fn map_resp(&self, r: AiResp) -> Option<AiResp> {
        let n = |k: &str| self.name_of(k);
        Some(match r {
            AiResp::Encoded { key, ms } => AiResp::Encoded { key: n(&key)?, ms },
            AiResp::NeedEncode(key) => AiResp::NeedEncode(n(&key)?),
            AiResp::EncodeFailed { key, err } => AiResp::EncodeFailed { key: n(&key)?, err },
            AiResp::Mask { key, seq, shape } => AiResp::Mask { key: n(&key)?, seq, shape },
            AiResp::Detected { key, batch, dets, w, h } => AiResp::Detected { key: n(&key)?, batch, dets, w, h },
            other => other,
        })
    }
}

impl Store for LocalStore {
    fn is_lab(&self) -> bool {
        false
    }

    fn open(&mut self, path: &str) {
        let p = PathBuf::from(path);
        let dir = if p.is_dir() { p } else { p.parent().map(Path::to_path_buf).unwrap_or(p) };
        if let Some(b) = self.batch.take() {
            b.cancel.store(true, Ordering::Relaxed);
        }
        let res = open_dataset(&dir);
        if res.is_ok() {
            self.root = Some(dir);
            self.prefetch = None;
            self.ai_images.clear();
        }
        self.events.push_back(match res {
            Ok(l) => StoreEvent::Opened(Ok(l.data), l.dirty),
            Err(e) => StoreEvent::Opened(Err(e), false),
        });
    }

    fn load_image(&mut self, name: &str, next: Option<&str>) {
        let Some(root) = self.root.clone() else { return };
        // use the background-decoded image if it is the one we prefetched
        let decoded = match self.prefetch.take() {
            Some(pf) if pf.name == name => pf.rx.recv().unwrap_or_else(|_| decode(&root.join(name))),
            _ => decode(&root.join(name)),
        };
        if let Some(n) = next {
            self.start_prefetch(n);
        }
        let res = decoded.map(|(d, a)| {
            self.remember_ai_image(name, a);
            ImageData {
                rgba: d.rgba,
                size: d.size,
                w: d.w,
                h: d.h,
                label_text: fs::read_to_string(label_path(&root, name)).ok(),
            }
        });
        self.events.push_back(StoreEvent::Image(name.to_string(), res));
    }

    fn write_label(&mut self, name: &str, ann: Option<&ImageAnn>, project: &Project) -> Result<String, String> {
        let root = self.root.as_ref().ok_or("no dataset open")?;
        match ann {
            Some(a) => {
                let p = write_label_file(root, name, a, project)?;
                Ok(p.strip_prefix(root).unwrap_or(&p).display().to_string())
            }
            None => {
                let p = label_path(root, name);
                match fs::remove_file(&p) {
                    Ok(()) => Ok("label file removed".into()),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok("no label file".into()),
                    Err(e) => Err(format!("remove {}: {e}", p.display())),
                }
            }
        }
    }

    fn save_project(&mut self, project: &Project) -> Result<String, String> {
        let root = self.root.as_ref().ok_or("no dataset open")?;
        let mut errs = vec![];
        if let Err(e) = write_classes(root, project) {
            errs.push(e);
        }
        if let Err(e) = project.save(root) {
            errs.push(e);
        }
        if errs.is_empty() {
            Ok(format!("Project file: {}", root.join(PROJECT_FILE).display()))
        } else {
            Err(errs.join("; "))
        }
    }

    fn ai(&mut self, req: AiReq) {
        if self.ai.is_none() {
            let settings = match &req {
                AiReq::Settings(s) => s.clone(),
                _ => AiSettings::load(),
            };
            let ctx = self.ctx.clone();
            self.ai = Some(ai::start(settings, move || ctx.request_repaint()));
            self.events.push_back(StoreEvent::Ai(AiResp::Models { sam: ai::sam_models(), yolo: vec![] }));
            if matches!(req, AiReq::Settings(_)) {
                return;
            }
        }
        match req {
            AiReq::Settings(s) => {
                if let Some(a) = &self.ai {
                    let _ = a.tx.send(ai::Req::Settings(s));
                }
            }
            AiReq::Encode { key } => self.with_image(&key, |key, img| ai::Req::Encode { key, img }),
            AiReq::Detect { key } => self.with_image(&key, |key, img| ai::Req::Detect { key, img }),
            AiReq::Segment { key, seq, points, bbox, polygon } => {
                let key = self.key(&key);
                if let Some(a) = &self.ai {
                    let _ = a.tx.send(ai::Req::Segment { key, seq, points, bbox, polygon });
                }
            }
            AiReq::Batch { id, keys } => {
                let (Some(a), Some(root)) = (&self.ai, &self.root) else { return };
                let items: Vec<(String, PathBuf)> = keys.iter().map(|k| (self.key(k), root.join(k))).collect();
                let b = Batch {
                    id,
                    cancel: Arc::new(AtomicBool::new(false)),
                    done: Arc::new(AtomicUsize::new(0)),
                    total: items.len(),
                    reported: 0,
                };
                let _ = a.tx.send(ai::Req::Batch {
                    id,
                    items,
                    decode: ai::decode_ai,
                    cancel: b.cancel.clone(),
                    done: b.done.clone(),
                });
                self.batch = Some(b);
            }
            AiReq::CancelBatch { id } => {
                if let Some(b) = self.batch.as_ref().filter(|b| b.id == id) {
                    b.cancel.store(true, Ordering::Relaxed);
                }
            }
        }
    }

    fn export(&mut self, o: ExportOpts) {
        let Some(root) = self.root.clone() else { return };
        // the UI's current project is what gets exported: it was just saved to disk
        let project = match Project::load(&root) {
            Ok(Some(p)) => p,
            _ => {
                self.events.push_back(StoreEvent::ExportDone(Err("could not read the project file".into())));
                return;
            }
        };
        let files = match scan_images(&root) {
            Ok(f) => f,
            Err(e) => {
                self.events.push_back(StoreEvent::ExportDone(Err(e)));
                return;
            }
        };
        let out = PathBuf::from(o.out.trim());
        let done = Arc::new(AtomicUsize::new(0));
        let total = Arc::new(AtomicUsize::new(0));
        let (d2, t2) = (done.clone(), total.clone());
        let (tx, rx) = mpsc::channel();
        let (out2, ctx) = (out.clone(), self.ctx.clone());
        // copying thousands of images takes a while: do it off the UI thread
        std::thread::spawn(move || {
            let res = export_dataset(&project, &root, &files, &out2, o.val_ratio, o.include_unlabeled, o.keep_split, o.by_folder, &|k, n| {
                d2.store(k, Ordering::Relaxed);
                t2.store(n, Ordering::Relaxed);
            });
            let _ = tx.send(res);
            ctx.request_repaint();
        });
        self.export = Some(Export { done, total, rx, out });
    }

    fn poll(&mut self) -> Vec<StoreEvent> {
        let mut out: Vec<StoreEvent> = self.events.drain(..).collect();
        if let Some(a) = &self.ai {
            let msgs: Vec<AiResp> = a.rx.try_iter().collect();
            for m in msgs {
                if let AiResp::BatchDone { id, .. } = &m {
                    if self.batch.as_ref().is_some_and(|b| b.id == *id) {
                        self.batch = None;
                    }
                }
                if let Some(m) = self.map_resp(m) {
                    out.push(StoreEvent::Ai(m));
                }
            }
        }
        if let Some(b) = &mut self.batch {
            let d = b.done.load(Ordering::Relaxed);
            if d != b.reported {
                b.reported = d;
                out.push(StoreEvent::Ai(AiResp::BatchProgress { id: b.id, done: d, total: b.total }));
            }
        }
        if let Some(e) = &self.export {
            match e.rx.try_recv() {
                Ok(res) => {
                    let msg = res.map(|st| {
                        format!("{} train + {} val + {} test images to {}", st.train, st.val, st.test, e.out.display())
                    });
                    out.push(StoreEvent::ExportDone(msg));
                    self.export = None;
                }
                Err(mpsc::TryRecvError::Empty) => {
                    out.push(StoreEvent::ExportProgress {
                        done: e.done.load(Ordering::Relaxed),
                        total: e.total.load(Ordering::Relaxed),
                    });
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    out.push(StoreEvent::ExportDone(Err("the export thread stopped unexpectedly".into())));
                    self.export = None;
                }
            }
        }
        out
    }
}
