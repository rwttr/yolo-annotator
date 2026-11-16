//! YOLO Annotator in the browser: the shared UI (annot-ui) talking to the lab server.
//! Images come over HTTP (/api/image/...), everything else over one WebSocket (/ws).

use annot_core::imageio::{decode_bytes, display_image};
use annot_core::model::*;
use annot_core::proto::*;
use annot_ui::store::{ImageData, Store, StoreEvent};
use eframe::egui;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

type Fetched = Arc<Mutex<Vec<(String, Result<Vec<u8>, String>)>>>;

pub struct WebStore {
    ctx: egui::Context,
    /// "http://host:port"
    base: String,
    ws: Option<(ewebsock::WsSender, ewebsock::WsReceiver)>,
    connected: bool,
    /// when the connection was lost (for the reconnect delay)
    lost_at: Option<f64>,
    events: VecDeque<StoreEvent>,
    fetched: Fetched,
    /// decoded images (the current one being waited for, and the prefetched next one)
    ready: HashMap<String, ImageData>,
    /// image the UI waits for
    want: Option<String>,
    requested: Vec<String>,
}

impl WebStore {
    pub fn new(ctx: egui::Context) -> Self {
        let loc = web_sys::window().map(|w| w.location());
        let base = loc.and_then(|l| l.origin().ok()).unwrap_or_else(|| "http://localhost:8080".into());
        let mut s = WebStore {
            ctx,
            base,
            ws: None,
            connected: false,
            lost_at: None,
            events: VecDeque::new(),
            fetched: Arc::new(Mutex::new(vec![])),
            ready: HashMap::new(),
            want: None,
            requested: vec![],
        };
        s.connect();
        s
    }

    fn connect(&mut self) {
        let url = format!("{}/ws", self.base.replacen("http", "ws", 1));
        let ctx = self.ctx.clone();
        match ewebsock::connect_with_wakeup(url, ewebsock::Options::default(), move || ctx.request_repaint()) {
            Ok(pair) => self.ws = Some(pair),
            Err(e) => self.events.push_back(StoreEvent::Status(format!("Cannot connect to the lab server: {e}"))),
        }
    }

    fn send(&mut self, m: &ClientMsg) {
        if let (Some((tx, _)), true) = (&mut self.ws, self.connected) {
            if let Ok(t) = serde_json::to_string(m) {
                tx.send(ewebsock::WsMessage::Text(t));
            }
        }
    }

    fn fetch(&mut self, name: &str) {
        if self.ready.contains_key(name) || self.requested.iter().any(|n| n == name) {
            return;
        }
        self.requested.push(name.to_string());
        let path: Vec<String> = name.split('/').map(|s| String::from(js_sys::encode_uri_component(s))).collect();
        let url = format!("{}/api/image/{}", self.base, path.join("/"));
        let (inbox, ctx, n) = (self.fetched.clone(), self.ctx.clone(), name.to_string());
        ehttp::fetch(ehttp::Request::get(url), move |r| {
            let res = match r {
                Ok(resp) if resp.ok => Ok(resp.bytes),
                Ok(resp) => Err(format!("HTTP {} {}", resp.status, resp.status_text)),
                Err(e) => Err(e),
            };
            inbox.lock().unwrap().push((n, res));
            ctx.request_repaint();
        });
    }

    fn deliver_if_ready(&mut self) {
        if let Some(w) = self.want.clone() {
            if let Some(img) = self.ready.remove(&w) {
                self.want = None;
                self.events.push_back(StoreEvent::Image(w, Ok(img)));
            }
        }
    }

    fn on_server(&mut self, m: ServerMsg) {
        let e = match m {
            ServerMsg::Welcome { admin, user, .. } => StoreEvent::Welcome { admin, user },
            ServerMsg::Opened(r) => {
                self.ready.clear();
                StoreEvent::Opened(r, false)
            }
            ServerMsg::Locks(l) => StoreEvent::Locks(l),
            ServerMsg::ImageChanged { name, ann, by } => StoreEvent::ImageChanged { name, ann, by },
            ServerMsg::Op { op, by } => StoreEvent::Op { op, by },
            ServerMsg::OpFailed(e) => StoreEvent::OpFailed(e),
            ServerMsg::Saved { name, result } => StoreEvent::Saved { name, result },
            ServerMsg::Ai(a) => StoreEvent::Ai(a),
            ServerMsg::ExportProgress { done, total } => StoreEvent::ExportProgress { done, total },
            ServerMsg::ExportDone(r) => StoreEvent::ExportDone(r),
            ServerMsg::Users(u) => StoreEvent::Users(u),
            ServerMsg::Status(s) => StoreEvent::Status(s),
        };
        self.events.push_back(e);
    }
}

impl Store for WebStore {
    fn is_lab(&self) -> bool {
        true
    }

    fn open(&mut self, path: &str) {
        self.send(&ClientMsg::AdminOpen { path: path.to_string() });
    }

    fn load_image(&mut self, name: &str, next: Option<&str>) {
        // tell the others we are here (and take the edit lock if the image is free)
        self.send(&ClientMsg::View { name: name.to_string() });
        self.want = Some(name.to_string());
        self.ready.retain(|n, _| n == name || Some(n.as_str()) == next);
        self.fetch(name);
        if let Some(n) = next {
            self.fetch(n);
        }
        self.deliver_if_ready();
    }

    fn write_label(&mut self, name: &str, ann: Option<&ImageAnn>, _p: &Project) -> Result<String, String> {
        if !self.connected {
            return Err("not connected to the lab server".into());
        }
        self.send(&ClientMsg::PutImage { name: name.to_string(), ann: ann.cloned() });
        Ok("sent".into())
    }

    fn save_project(&mut self, _p: &Project) -> Result<String, String> {
        Ok("kept by the lab server".into())
    }

    fn op(&mut self, op: ClassOp) {
        self.send(&ClientMsg::Op(op));
    }

    fn ai(&mut self, req: AiReq) {
        self.send(&ClientMsg::Ai(req));
    }

    fn export(&mut self, opts: ExportOpts) {
        self.send(&ClientMsg::Export(opts));
    }

    fn hello(&mut self, user: &str) {
        self.send(&ClientMsg::Hello { user: user.to_string() });
    }

    fn poll(&mut self) -> Vec<StoreEvent> {
        // WebSocket
        let mut incoming = vec![];
        if let Some((_, rx)) = &self.ws {
            while let Some(ev) = rx.try_recv() {
                incoming.push(ev);
            }
        }
        for ev in incoming {
            match ev {
                ewebsock::WsEvent::Opened => {
                    self.connected = true;
                    self.lost_at = None;
                    self.events.push_back(StoreEvent::Connected(true));
                }
                ewebsock::WsEvent::Message(ewebsock::WsMessage::Text(t)) => match serde_json::from_str::<ServerMsg>(&t) {
                    Ok(m) => self.on_server(m),
                    Err(e) => self.events.push_back(StoreEvent::Status(format!("Bad message from the server: {e}"))),
                },
                ewebsock::WsEvent::Message(_) => {}
                ewebsock::WsEvent::Error(_) | ewebsock::WsEvent::Closed => {
                    if self.connected || self.lost_at.is_none() {
                        self.events.push_back(StoreEvent::Connected(false));
                    }
                    self.connected = false;
                    self.ws = None;
                    self.lost_at = Some(self.ctx.input(|i| i.time));
                }
            }
        }
        // reconnect a couple of seconds after the connection was lost
        if self.ws.is_none() {
            let now = self.ctx.input(|i| i.time);
            if self.lost_at.is_none_or(|t| now - t > 2.0) {
                self.lost_at = Some(now);
                self.connect();
            }
            self.ctx.request_repaint_after(std::time::Duration::from_millis(500));
        }
        // images
        let got: Vec<_> = std::mem::take(&mut *self.fetched.lock().unwrap());
        for (name, res) in got {
            self.requested.retain(|n| *n != name);
            match res.and_then(|b| decode_bytes(&b)) {
                Ok(img) => {
                    let d = display_image(&img);
                    let data = ImageData { rgba: d.rgba, size: d.size, w: d.w, h: d.h, label_text: None };
                    self.ready.insert(name, data);
                }
                Err(e) => {
                    if self.want.as_deref() == Some(name.as_str()) {
                        self.want = None;
                        self.events.push_back(StoreEvent::Image(name, Err(e)));
                    }
                }
            }
        }
        self.deliver_if_ready();
        self.events.drain(..).collect()
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    use wasm_bindgen::JsCast;
    // "?user=NAME" in the address sets the name (handy as a personal bookmark)
    if let Some(w) = web_sys::window() {
        let q = w.location().search().unwrap_or_default();
        let user = q.trim_start_matches('?').split('&').find_map(|kv| kv.strip_prefix("user="));
        if let (Some(u), Ok(Some(ls))) = (user, w.local_storage()) {
            let u = js_sys::decode_uri_component(&u.replace('+', " ")).map(String::from).unwrap_or_default();
            if !u.trim().is_empty() {
                // eframe keeps values as RON; a string is stored quoted
                let _ = ls.set_item("user", &format!("{:?}", u.trim()));
            }
        }
    }
    let canvas = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id("annotator"))
        .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok())
        .expect("canvas #annotator");
    wasm_bindgen_futures::spawn_local(async move {
        let r = eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|cc| {
                    let store = Box::new(WebStore::new(cc.egui_ctx.clone()));
                    Ok(Box::new(annot_ui::App::new(cc, store, None)))
                }),
            )
            .await;
        if let Err(e) = r {
            web_sys::console::error_1(&e);
        }
    });
}
