//! YOLO Annotator lab server.
//!
//!   yolo-annotator-server.exe [DATASET_FOLDER] [--port 8080] [--listen 0.0.0.0] [--no-admin] [--web DIR]
//!
//! --listen   address to listen on (default all network interfaces; 127.0.0.1 = this PC only)
//! --no-admin nobody can switch the dataset from a browser, not even on the server machine
//!
//! Serves the web annotator to the LAN. The dataset folder is chosen only here, on the
//! server machine (argument, last used folder, or a folder dialog); browsers cannot pick or
//! read other folders. Labels are written exactly like the desktop app writes them.

use annot_ai as ai;
use annot_core::dataset::open_dataset;
use annot_core::model::*;
use annot_core::proto::*;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{ConnectInfo, Path as UrlPath, State as AxState};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The open dataset.
struct Ds {
    root: PathBuf,
    data: OpenedData,
    file_set: HashSet<String>,
    /// annotations.json / classes.txt need writing
    dirty: bool,
}

struct Client {
    user: Option<String>,
    admin: bool,
    tx: mpsc::UnboundedSender<ServerMsg>,
}

struct Batch {
    id: u64,
    owner: u64,
    cancel: Arc<AtomicBool>,
    done: Arc<AtomicUsize>,
    total: usize,
    labeled: usize,
    reported: usize,
}

#[derive(Default)]
struct Inner {
    ds: Option<Ds>,
    clients: HashMap<u64, Client>,
    next_id: u64,
    /// image -> client editing it
    locks: BTreeMap<String, u64>,
    ai: Option<ai::AiHandle>,
    settings: AiSettings,
    /// (key, seq) of a Magic request -> client that asked
    segment_owner: HashMap<(String, u64), u64>,
    /// key -> clients waiting for a single-image YOLO result
    detect_waiters: HashMap<String, Vec<u64>>,
    batch: Option<Batch>,
    export_running: bool,
    /// last "AI ready" message, for clients that connect later
    ai_status: Option<String>,
}

struct Server {
    inner: Mutex<Inner>,
    web_dir: PathBuf,
    exports_dir: PathBuf,
    yolo_dir: PathBuf,
    /// nobody may switch the dataset from a browser
    no_admin: bool,
}

type Shared = Arc<Server>;

// ------------------------------------------------------------------ helpers

impl Inner {
    fn user_of(&self, id: u64) -> String {
        self.clients.get(&id).and_then(|c| c.user.clone()).unwrap_or_else(|| "?".into())
    }

    fn send(&self, id: u64, m: ServerMsg) {
        if let Some(c) = self.clients.get(&id) {
            let _ = c.tx.send(m);
        }
    }

    /// To every client that said hello (has a name), except `skip`.
    fn broadcast(&self, m: &ServerMsg, skip: Option<u64>) {
        for (id, c) in &self.clients {
            if c.user.is_some() && Some(*id) != skip {
                let _ = c.tx.send(m.clone());
            }
        }
    }

    fn locks_msg(&self) -> ServerMsg {
        ServerMsg::Locks(self.locks.iter().map(|(img, id)| (img.clone(), self.user_of(*id))).collect())
    }

    fn users_msg(&self) -> ServerMsg {
        let mut u: Vec<String> = self.clients.values().filter_map(|c| c.user.clone()).collect();
        u.sort();
        ServerMsg::Users(u)
    }

    /// Worker key of an image of the open dataset.
    fn key(&self, name: &str) -> String {
        let root = self.ds.as_ref().map(|d| d.root.display().to_string()).unwrap_or_default();
        format!("{root}|{name}")
    }

    fn name_of(&self, key: &str) -> Option<String> {
        key.strip_prefix(&self.key("")).map(str::to_string)
    }

    /// Write annotations.json + classes.txt if needed.
    fn flush(&mut self) {
        let Some(ds) = &mut self.ds else { return };
        if !ds.dirty {
            return;
        }
        let p = &ds.data.project;
        let r1 = write_classes(&ds.root, p);
        let r2 = p.save(&ds.root);
        match r1.and(r2) {
            Ok(()) => ds.dirty = false,
            Err(e) => log(&format!("SAVE ERROR: {e}")),
        }
    }

    /// Apply a class / settings change, rewrite affected label files, tell everyone.
    fn apply_op(&mut self, op: ClassOp, by: &str) -> Result<(), String> {
        let ds = self.ds.as_mut().ok_or("no dataset open")?;
        let changed = ds.data.project.apply_op(&op)?;
        for name in changed {
            if let Some(a) = ds.data.project.images.get(&name) {
                if let Err(e) = write_label_file(&ds.root, &name, a, &ds.data.project) {
                    log(&format!("label write error: {e}"));
                }
            }
        }
        ds.dirty = true;
        self.broadcast(&ServerMsg::Op { op, by: by.to_string() }, None);
        Ok(())
    }

    /// Make sure the classes a YOLO result needs exist (added for everyone, in order).
    fn add_classes_for(&mut self, dets: &[Detection]) {
        let Some(ds) = &self.ds else { return };
        let p = &ds.data.project;
        let mut need = p.missing_classes(dets);
        if let Some(m) = dets.iter().filter(|d| d.class_name.is_none()).map(|d| d.class_id).max() {
            for i in p.classes.len() + need.len()..=m {
                need.push(format!("class_{i}"));
            }
        }
        for n in need {
            let _ = self.apply_op(ClassOp::Add(n), "auto-label");
        }
    }

    fn ensure_ai(&mut self) {
        if self.ai.is_none() {
            self.ai = Some(ai::start(self.settings.clone(), || {}));
        }
    }

    /// The shared AI settings as browsers see them (YOLO model by file name only).
    fn client_settings(&self) -> AiSettings {
        let mut s = self.settings.clone();
        s.yolo = Path::new(&s.yolo).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        s
    }
}

fn log(s: &str) {
    println!("[{}] {s}", chrono_now());
}

fn chrono_now() -> String {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (h, m, s) = ((t / 3600 + 7) % 24, (t / 60) % 60, t % 60); // shown in UTC+7 (lab time)
    format!("{h:02}:{m:02}:{s:02}")
}

/// A plain file name: no folders, no "..".
fn plain_name(s: &str) -> Option<&str> {
    let s = s.trim();
    let bad = s.is_empty() || s.contains(['/', '\\', ':']) || s.starts_with('.');
    (!bad).then_some(s)
}

fn yolo_models(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| n.to_lowercase().ends_with(".onnx"))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

// ------------------------------------------------------------------ dataset

fn open_ds(srv: &Server, dir: &Path) -> Result<(), String> {
    let loaded = open_dataset(dir)?;
    let mut st = srv.inner.lock().unwrap();
    st.flush();
    if let Some(b) = st.batch.take() {
        b.cancel.store(true, Ordering::Relaxed);
    }
    let mut data = loaded.data;
    data.label = dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| dir.display().to_string());
    // browsers see the dataset name only, never the server's folder path
    data.status = data.status.replace(&dir.display().to_string(), &format!("'{}'", data.label));
    log(&format!("dataset: {} ({} images)", dir.display(), data.files.len()));
    st.ds = Some(Ds {
        root: dir.to_path_buf(),
        file_set: data.files.iter().cloned().collect(),
        data,
        dirty: loaded.dirty,
    });
    st.locks.clear();
    st.segment_owner.clear();
    st.detect_waiters.clear();
    let msg = ServerMsg::Opened(Ok(st.ds.as_ref().unwrap().data.clone()));
    st.broadcast(&msg, None);
    let l = st.locks_msg();
    st.broadcast(&l, None);
    remember_dataset(dir);
    Ok(())
}

fn config_file() -> Option<PathBuf> {
    std::env::var_os("APPDATA").map(|a| PathBuf::from(a).join("YOLO Annotator").join("server_dataset.txt"))
}

fn remember_dataset(dir: &Path) {
    if let Some(f) = config_file() {
        let _ = std::fs::create_dir_all(f.parent().unwrap());
        let _ = std::fs::write(f, dir.display().to_string());
    }
}

fn remembered_dataset() -> Option<PathBuf> {
    let p = PathBuf::from(std::fs::read_to_string(config_file()?).ok()?.trim());
    p.is_dir().then_some(p)
}

// ------------------------------------------------------------------ client messages

fn handle(srv: &Shared, id: u64, msg: ClientMsg) {
    let mut st = srv.inner.lock().unwrap();
    match msg {
        ClientMsg::Hello { user } => {
            let base: String = user.trim().chars().take(40).collect();
            let base = if base.is_empty() { "guest".to_string() } else { base };
            let taken = |n: &str, st: &Inner| st.clients.iter().any(|(k, c)| *k != id && c.user.as_deref() == Some(n));
            let mut name = base.clone();
            let mut k = 2;
            while taken(&name, &st) {
                name = format!("{base} ({k})");
                k += 1;
            }
            let admin = st.clients.get(&id).is_some_and(|c| c.admin);
            if let Some(c) = st.clients.get_mut(&id) {
                c.user = Some(name.clone());
            }
            log(&format!("{name} joined{}", if admin { " (server machine)" } else { "" }));
            st.send(id, ServerMsg::Welcome { version: VERSION.into(), admin, user: name });
            let opened = match &st.ds {
                Some(ds) => Ok(ds.data.clone()),
                None => Err("No dataset is open on the lab server yet. Open one on the server machine.".into()),
            };
            st.send(id, ServerMsg::Opened(opened));
            st.send(id, st.locks_msg());
            st.send(id, ServerMsg::Ai(AiResp::Models { sam: ai::sam_models(), yolo: yolo_models(&srv.yolo_dir) }));
            st.send(id, ServerMsg::Ai(AiResp::Settings(st.client_settings())));
            if let Some(s) = st.ai_status.clone() {
                st.send(id, ServerMsg::Ai(AiResp::Ready(s)));
            }
            let u = st.users_msg();
            st.broadcast(&u, None);
        }
        ClientMsg::AdminOpen { path } => {
            if !st.clients.get(&id).is_some_and(|c| c.admin) {
                st.send(id, ServerMsg::Status("The dataset folder can only be changed on the server machine.".into()));
                return;
            }
            drop(st);
            if let Err(e) = open_ds(srv, Path::new(path.trim())) {
                srv.inner.lock().unwrap().send(id, ServerMsg::Status(format!("Cannot open {path}: {e}")));
            }
        }
        ClientMsg::View { name } => {
            let Some(ds) = &st.ds else { return };
            if !ds.file_set.contains(&name) {
                return;
            }
            st.locks.retain(|img, c| *c != id || *img == name);
            st.locks.entry(name).or_insert(id);
            let l = st.locks_msg();
            st.broadcast(&l, None);
        }
        ClientMsg::PutImage { name, ann } => {
            let user = st.user_of(id);
            if let Some(owner) = st.locks.get(&name).copied().filter(|o| *o != id) {
                let who = st.user_of(owner);
                st.send(id, ServerMsg::Saved { name, result: Err(format!("{who} is editing this image")) });
                return;
            }
            let Some(ds) = st.ds.as_mut() else { return };
            if !ds.file_set.contains(&name) {
                return;
            }
            let result = match &ann {
                Some(a) => {
                    ds.data.project.images.insert(name.clone(), a.clone());
                    write_label_file(&ds.root, &name, a, &ds.data.project)
                        .map(|p| p.strip_prefix(&ds.root).unwrap_or(&p).display().to_string())
                }
                None => {
                    ds.data.project.images.remove(&name);
                    let p = label_path(&ds.root, &name);
                    match std::fs::remove_file(&p) {
                        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(format!("remove label: {e}")),
                        _ => Ok("label file removed".into()),
                    }
                }
            };
            ds.dirty = true;
            st.locks.entry(name.clone()).or_insert(id);
            st.send(id, ServerMsg::Saved { name: name.clone(), result });
            st.broadcast(&ServerMsg::ImageChanged { name, ann, by: user }, Some(id));
        }
        ClientMsg::Op(op) => {
            let user = st.user_of(id);
            log(&format!("{user}: {op:?}"));
            if let Err(e) = st.apply_op(op, &user) {
                st.send(id, ServerMsg::OpFailed(e));
            }
        }
        ClientMsg::Ai(req) => handle_ai(srv, &mut st, id, req),
        ClientMsg::Export(o) => {
            let Some(name) = plain_name(&o.out).map(str::to_string) else {
                st.send(id, ServerMsg::ExportDone(Err("Use a plain name (no folders) for the export.".into())));
                return;
            };
            if st.export_running {
                st.send(id, ServerMsg::ExportDone(Err("Another export is running; try again in a moment.".into())));
                return;
            }
            st.flush();
            let Some(ds) = &st.ds else { return };
            let (project, root, files) = (ds.data.project.clone(), ds.root.clone(), ds.data.files.clone());
            let out = srv.exports_dir.join(&name);
            st.export_running = true;
            log(&format!("{} exports to {}", st.user_of(id), out.display()));
            drop(st);
            let srv2 = srv.clone();
            std::thread::spawn(move || {
                let _ = std::fs::create_dir_all(&srv2.exports_dir);
                let last = AtomicUsize::new(usize::MAX);
                let res = export_dataset(&project, &root, &files, &out, o.val_ratio, o.include_unlabeled, o.keep_split, o.by_folder, &|k, n| {
                    // progress about every 2 %
                    let step = (n / 50).max(1);
                    if k / step != last.swap(k / step, Ordering::Relaxed) {
                        srv2.inner.lock().unwrap().send(id, ServerMsg::ExportProgress { done: k, total: n });
                    }
                });
                let mut st = srv2.inner.lock().unwrap();
                st.export_running = false;
                let msg = res.map(|s| {
                    format!("{} train + {} val + {} test images to the server folder {}", s.train, s.val, s.test, out.display())
                });
                st.send(id, ServerMsg::ExportDone(msg));
            });
        }
    }
}

fn handle_ai(srv: &Shared, st: &mut Inner, id: u64, req: AiReq) {
    st.ensure_ai();
    let tx = st.ai.as_ref().unwrap().tx.clone();
    let root = st.ds.as_ref().map(|d| d.root.clone());
    // decode on a worker thread (never with the state lock held)
    let with_image = |name: String, key: String, make: fn(String, ai::AiImage) -> ai::Req| {
        if let Some(root) = &root {
            let path = root.join(&name);
            let tx = tx.clone();
            std::thread::spawn(move || match ai::decode_ai(&path) {
                Ok(img) => {
                    let _ = tx.send(make(key, img));
                }
                Err(e) => log(&format!("AI decode {name}: {e}")),
            });
        }
    };
    let known = |st: &Inner, k: &str| st.ds.as_ref().is_some_and(|d| d.file_set.contains(k));
    match req {
        AiReq::Settings(mut s) => {
            // models are names in the server's models folder; never paths from a browser
            let yolo_name = plain_name(&s.yolo).filter(|n| srv.yolo_dir.join(n).is_file()).unwrap_or("").to_string();
            s.yolo = if yolo_name.is_empty() { String::new() } else { srv.yolo_dir.join(&yolo_name).display().to_string() };
            if st.settings.sam != s.sam || st.settings.yolo != s.yolo {
                log(&format!("{}: AI settings sam={} yolo={}", st.user_of(id), s.sam, if yolo_name.is_empty() { "none" } else { &yolo_name }));
            }
            s.engine = st.settings.engine; // the engine is the server's choice
            st.settings = s.clone();
            let _ = tx.send(ai::Req::Settings(s));
            let cs = st.client_settings();
            st.broadcast(&ServerMsg::Ai(AiResp::Settings(cs)), Some(id));
        }
        AiReq::Encode { key } if known(st, &key) => {
            let k = st.key(&key);
            with_image(key, k, |key, img| ai::Req::Encode { key, img });
        }
        AiReq::Segment { key, seq, points, bbox, polygon } if known(st, &key) => {
            let k = st.key(&key);
            st.segment_owner.insert((key, seq), id);
            let _ = tx.send(ai::Req::Segment { key: k, seq, points, bbox, polygon });
        }
        AiReq::Detect { key } if known(st, &key) => {
            let k = st.key(&key);
            st.detect_waiters.entry(key.clone()).or_default().push(id);
            with_image(key, k, |key, img| ai::Req::Detect { key, img });
        }
        AiReq::Batch { id: bid, keys } => {
            if st.batch.is_some() {
                st.send(id, ServerMsg::Ai(AiResp::BatchDone { id: bid, labeled: 0, errors: vec!["another auto-label run is going on".into()], cancelled: true }));
                return;
            }
            let Some(root) = root.clone() else { return };
            let items: Vec<(String, PathBuf)> = keys
                .into_iter()
                .filter(|k| known(st, k))
                .map(|k| (st.key(&k), root.join(&k)))
                .collect();
            let b = Batch {
                id: bid,
                owner: id,
                cancel: Arc::new(AtomicBool::new(false)),
                done: Arc::new(AtomicUsize::new(0)),
                total: items.len(),
                labeled: 0,
                reported: 0,
            };
            log(&format!("{} auto-labels {} images", st.user_of(id), b.total));
            let _ = tx.send(ai::Req::Batch { id: bid, items, decode: ai::decode_ai, cancel: b.cancel.clone(), done: b.done.clone() });
            st.batch = Some(b);
        }
        AiReq::CancelBatch { id: bid } => {
            if let Some(b) = st.batch.as_ref().filter(|b| b.id == bid) {
                b.cancel.store(true, Ordering::Relaxed);
            }
        }
        _ => {} // an image that is not in the dataset
    }
}

/// Route the AI worker's answers (runs on its own thread).
fn ai_router(srv: Shared) {
    loop {
        let msg = {
            let st = srv.inner.lock().unwrap();
            match &st.ai {
                Some(a) => a.rx.recv_timeout(std::time::Duration::from_millis(0)).ok(),
                None => None,
            }
        };
        let Some(m) = msg else {
            // batch progress, then wait a little
            {
                let mut st = srv.inner.lock().unwrap();
                if let Some(b) = &mut st.batch {
                    let d = b.done.load(Ordering::Relaxed);
                    if d != b.reported {
                        b.reported = d;
                        let (owner, bid, total) = (b.owner, b.id, b.total);
                        st.send(owner, ServerMsg::Ai(AiResp::BatchProgress { id: bid, done: d, total }));
                    }
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(40));
            continue;
        };
        let mut st = srv.inner.lock().unwrap();
        match m {
            AiResp::Encoded { key, ms } => {
                if let Some(n) = st.name_of(&key) {
                    st.broadcast(&ServerMsg::Ai(AiResp::Encoded { key: n, ms }), None);
                }
            }
            AiResp::NeedEncode(key) => {
                if let Some(n) = st.name_of(&key) {
                    st.broadcast(&ServerMsg::Ai(AiResp::NeedEncode(n)), None);
                }
            }
            AiResp::EncodeFailed { key, err } => {
                if let Some(n) = st.name_of(&key) {
                    st.broadcast(&ServerMsg::Ai(AiResp::EncodeFailed { key: n, err }), None);
                }
            }
            AiResp::Mask { key, seq, shape } => {
                if let Some(n) = st.name_of(&key) {
                    if let Some(owner) = st.segment_owner.remove(&(n.clone(), seq)) {
                        st.send(owner, ServerMsg::Ai(AiResp::Mask { key: n, seq, shape }));
                    }
                }
                if st.segment_owner.len() > 1000 {
                    st.segment_owner.clear();
                }
            }
            AiResp::Detected { key, batch: None, dets, w, h } => {
                if let Some(n) = st.name_of(&key) {
                    for owner in st.detect_waiters.remove(&n).unwrap_or_default() {
                        st.send(owner, ServerMsg::Ai(AiResp::Detected { key: n.clone(), batch: None, dets: dets.clone(), w, h }));
                    }
                }
            }
            AiResp::Detected { key, batch: Some(bid), dets, w, h } => {
                // auto-label run: the server labels the image itself (if still unlabeled and free)
                let Some(n) = st.name_of(&key) else { continue };
                if !st.batch.as_ref().is_some_and(|b| b.id == bid) || dets.is_empty() || st.locks.contains_key(&n) {
                    continue;
                }
                if st.ds.as_ref().is_none_or(|d| d.data.project.images.contains_key(&n)) {
                    continue;
                }
                st.add_classes_for(&dets);
                let ds = st.ds.as_mut().unwrap();
                let shapes = ds.data.project.detections_to_shapes(&dets, w, h);
                let ann = ImageAnn { width: w, height: h, shapes };
                ds.data.project.images.insert(n.clone(), ann.clone());
                if let Err(e) = write_label_file(&ds.root, &n, &ann, &ds.data.project) {
                    log(&format!("label write error: {e}"));
                }
                ds.dirty = true;
                if let Some(b) = &mut st.batch {
                    b.labeled += 1;
                }
                st.broadcast(&ServerMsg::ImageChanged { name: n, ann: Some(ann), by: "auto-label".into() }, None);
            }
            AiResp::BatchDone { id, errors, cancelled, .. } => {
                if let Some(b) = st.batch.take_if(|b| b.id == id) {
                    log(&format!("auto-label {}: {} labeled", if cancelled { "cancelled" } else { "done" }, b.labeled));
                    st.send(b.owner, ServerMsg::Ai(AiResp::BatchDone { id, labeled: b.labeled, errors, cancelled }));
                    st.flush();
                }
            }
            AiResp::Crashed => {
                log("AI worker crashed");
                st.broadcast(&ServerMsg::Ai(AiResp::Crashed), None);
                st.ai = None;
            }
            other => {
                if let AiResp::Ready(s) | AiResp::Unavailable(s) | AiResp::Error(s) = &other {
                    log(&format!("AI: {s}"));
                }
                if let AiResp::Ready(s) = &other {
                    st.ai_status = Some(s.clone());
                }
                st.broadcast(&ServerMsg::Ai(other), None);
            }
        }
    }
}

// ------------------------------------------------------------------ web

async fn ws_upgrade(
    ws: WebSocketUpgrade,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    AxState(srv): AxState<Shared>,
) -> Response {
    ws.max_message_size(256 << 20).on_upgrade(move |socket| client_session(socket, addr, srv))
}

async fn client_session(socket: WebSocket, addr: SocketAddr, srv: Shared) {
    use futures_util::{SinkExt, StreamExt};
    let (mut sink, mut stream) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<ServerMsg>();
    let id = {
        let mut st = srv.inner.lock().unwrap();
        st.next_id += 1;
        let id = st.next_id;
        st.clients.insert(id, Client { user: None, admin: addr.ip().is_loopback() && !srv.no_admin, tx });
        id
    };
    let writer = tokio::spawn(async move {
        while let Some(m) = rx.recv().await {
            let Ok(text) = serde_json::to_string(&m) else { continue };
            if sink.send(Message::Text(text)).await.is_err() {
                break;
            }
        }
    });
    while let Some(Ok(msg)) = stream.next().await {
        match msg {
            Message::Text(t) => match serde_json::from_str::<ClientMsg>(&t) {
                Ok(m) => {
                    let srv = srv.clone();
                    // handlers may do file IO: keep it off the async workers
                    let _ = tokio::task::spawn_blocking(move || handle(&srv, id, m)).await;
                }
                Err(e) => log(&format!("bad message from {addr}: {e}")),
            },
            Message::Close(_) => break,
            _ => {}
        }
    }
    writer.abort();
    let mut st = srv.inner.lock().unwrap();
    if let Some(c) = st.clients.remove(&id) {
        if let Some(u) = c.user {
            log(&format!("{u} left"));
        }
    }
    st.locks.retain(|_, c| *c != id);
    st.segment_owner.retain(|_, c| *c != id);
    let (l, u) = (st.locks_msg(), st.users_msg());
    st.broadcast(&l, None);
    st.broadcast(&u, None);
}

/// Image bytes; only files of the open dataset's image list.
async fn image(UrlPath(name): UrlPath<String>, AxState(srv): AxState<Shared>) -> Response {
    let path = {
        let st = srv.inner.lock().unwrap();
        match &st.ds {
            Some(ds) if ds.file_set.contains(&name) => ds.root.join(&name),
            _ => return StatusCode::NOT_FOUND.into_response(),
        }
    };
    match tokio::fs::read(&path).await {
        Ok(bytes) => {
            let ext = name.rsplit('.').next().unwrap_or("").to_lowercase();
            let ct = match ext.as_str() {
                "jpg" | "jpeg" => "image/jpeg",
                "png" => "image/png",
                "bmp" => "image/bmp",
                "webp" => "image/webp",
                "tif" | "tiff" => "image/tiff",
                _ => "application/octet-stream",
            };
            ([(header::CONTENT_TYPE, ct), (header::CACHE_CONTROL, "no-cache")], bytes).into_response()
        }
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

fn lan_ip() -> Option<std::net::IpAddr> {
    // no packet is sent: this only asks the OS which interface it would use
    let s = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    s.connect("8.8.8.8:80").ok()?;
    Some(s.local_addr().ok()?.ip())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut port: u16 = 8080;
    let mut web: Option<PathBuf> = None;
    let mut dataset: Option<PathBuf> = None;
    let mut listen: std::net::IpAddr = [0, 0, 0, 0].into();
    let mut no_admin = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                port = args.get(i + 1).and_then(|p| p.parse().ok()).unwrap_or(port);
                i += 1;
            }
            "--listen" => {
                listen = args.get(i + 1).and_then(|p| p.parse().ok()).unwrap_or(listen);
                i += 1;
            }
            "--no-admin" => no_admin = true,
            "--web" => {
                web = args.get(i + 1).map(PathBuf::from);
                i += 1;
            }
            "-h" | "--help" => {
                println!("yolo-annotator-server [DATASET_FOLDER] [--port 8080] [--listen 0.0.0.0] [--no-admin] [--web DIR]");
                return;
            }
            a => dataset = Some(PathBuf::from(a)),
        }
        i += 1;
    }
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_default();
    let assets = ai::assets_dir().unwrap_or_else(|| exe_dir.clone());
    let srv: Shared = Arc::new(Server {
        inner: Mutex::new(Inner { settings: AiSettings::default(), ..Default::default() }),
        web_dir: web.unwrap_or_else(|| exe_dir.join("web")),
        exports_dir: exe_dir.join("exports"),
        yolo_dir: assets.join("models").join("yolo"),
        no_admin,
    });
    println!("YOLO Annotator lab server {VERSION}");
    // the dataset: argument, else the last one, else ask here on the server machine
    let dataset = dataset.or_else(remembered_dataset).or_else(|| {
        println!("Choose the dataset folder in the dialog...");
        rfd::FileDialog::new().set_title("Dataset folder for the lab").pick_folder()
    });
    match dataset {
        Some(d) => {
            if let Err(e) = open_ds(&srv, &d) {
                println!("Cannot open {}: {e}", d.display());
            }
        }
        None => println!("No dataset yet: open http://localhost:{port} on this machine and choose one (Open)."),
    }
    if !srv.web_dir.join("index.html").exists() {
        println!("WARNING: web app not found in {} (index.html)", srv.web_dir.display());
    }
    {
        let srv = srv.clone();
        std::thread::spawn(move || ai_router(srv));
    }
    {
        // annotations.json is saved every few seconds after changes
        let srv = srv.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            srv.inner.lock().unwrap().flush();
        });
    }
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async move {
        let app = Router::new()
            .route("/ws", get(ws_upgrade))
            .route("/api/image/*name", get(image))
            .fallback_service(tower_http::services::ServeDir::new(&srv.web_dir))
            .with_state(srv.clone());
        let addr = SocketAddr::new(listen, port);
        let listener = match tokio::net::TcpListener::bind(addr).await {
            Ok(l) => l,
            Err(e) => {
                println!("Cannot listen on port {port}: {e}");
                return;
            }
        };
        let ip = lan_ip().map(|i| i.to_string()).unwrap_or_else(|| "<this-pc-ip>".into());
        println!("Lab members open:  http://{ip}:{port}");
        println!("On this machine:   http://localhost:{port}   (can switch the dataset)");
        println!("Exports go to:     {}", srv.exports_dir.display());
        println!("YOLO models from:  {}", srv.yolo_dir.display());
        println!("Press Ctrl+C to stop (all labels are saved).");
        let shutdown = {
            let srv = srv.clone();
            async move {
                let _ = tokio::signal::ctrl_c().await;
                srv.inner.lock().unwrap().flush();
                println!("Saved. Bye.");
            }
        };
        let _ = axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>())
            .with_graceful_shutdown(shutdown)
            .await;
    });
}
