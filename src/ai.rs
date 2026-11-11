//! AI assistance on a background thread:
//! * SAM 2.1 (click / box prompt -> polygon) for the Magic tool
//! * a user-trained Ultralytics YOLO .onnx model for auto-labelling
//!
//! ONNX Runtime is loaded at run time from `runtime\cuda` or `runtime\directml` next to the
//! exe (or `ai_assets\` while developing), so the app still starts without them.

use crate::mask::{mask_to_polygon, nms};
use image::RgbImage;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

/// Images are handed to the AI downscaled to this longest side.
pub const AI_MAX_SIDE: u32 = 1280;
const SAM_SIZE: usize = 1024;
const EMBED_CACHE: usize = 3;

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
    /// path of a YOLO .onnx model for auto-labelling ("" = none)
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

/// An image prepared for the AI: downscaled RGB + the original size.
#[derive(Clone)]
pub struct AiImage {
    pub rgb: Arc<RgbImage>,
    pub w: u32,
    pub h: u32,
}

pub struct Detection {
    pub class_id: usize,
    pub class_name: Option<String>,
    #[allow(dead_code)] // kept for a future score filter in the UI
    pub score: f32,
    pub bbox: [f32; 4],
    pub polygon: Option<Vec<[f32; 2]>>,
}

pub type DecodeFn = fn(&Path) -> Result<AiImage, String>;

pub enum Req {
    Settings(AiSettings),
    Encode { key: String, img: AiImage },
    Segment { key: String, seq: u64, points: Vec<([f32; 2], bool)>, bbox: Option<[f32; 4]>, polygon: bool },
    Detect { key: String, img: AiImage },
    /// `id` comes back in its results, so late results of an old batch can be ignored
    Batch { id: u64, items: Vec<(String, PathBuf)>, decode: DecodeFn, cancel: Arc<AtomicBool>, done: Arc<AtomicUsize> },
}

pub enum Resp {
    /// engine / model description for the status bar
    Ready(String),
    /// SAM cannot be used (model missing / failed to load)
    Unavailable(String),
    /// the AI thread panicked; nothing more will be answered
    Crashed,
    Info(String),
    Error(String),
    Encoded { key: String, ms: u128 },
    NeedEncode(String),
    EncodeFailed { key: String, err: String },
    Mask { key: String, seq: u64, shape: Option<Vec<[f32; 2]>> },
    /// `batch`: id of the batch, None for a single Detect
    Detected { key: String, batch: Option<u64>, dets: Vec<Detection>, w: u32, h: u32 },
    BatchDone { id: u64, labeled: usize, errors: Vec<String>, cancelled: bool },
}

pub struct AiHandle {
    pub tx: mpsc::Sender<Req>,
    pub rx: mpsc::Receiver<Resp>,
}

pub fn start(settings: AiSettings, repaint: impl Fn() + Send + 'static) -> AiHandle {
    let (tx, rx_req) = mpsc::channel();
    let (tx_resp, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("ai".into())
        .spawn(move || {
            let send = |r: Resp| {
                let _ = tx_resp.send(r);
                repaint();
            };
            // a panic in ONNX Runtime glue must not take the UI down
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| worker(settings, rx_req, &send)));
            if res.is_err() {
                send(Resp::Crashed);
            }
        })
        .expect("spawn ai thread");
    AiHandle { tx, rx }
}

// ------------------------------------------------------------------ assets / engine

/// Folder holding `runtime\` and `models\`.
pub fn assets_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let ed = exe.parent()?.to_path_buf();
    let mut c = vec![ed.clone(), ed.join("ai_assets")];
    // development builds: <project>\target\release\yolo-annotator.exe
    let project = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if exe.starts_with(project.join("target")) {
        c.push(project.join("ai_assets"));
    }
    c.into_iter().find(|d| d.join("models").is_dir() || d.join("runtime").is_dir())
}

pub fn sam_models() -> Vec<String> {
    let Some(d) = assets_dir() else { return vec![] };
    let mut v: Vec<String> = std::fs::read_dir(d.join("models"))
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().join("vision_encoder.onnx").exists())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    let rank = |s: &str| ["tiny", "small", "base", "large"].iter().position(|k| s.contains(k)).unwrap_or(9);
    v.sort_by_key(|s| rank(s));
    v
}

fn try_load(lib: &Path) -> Result<(), String> {
    // keep it loaded for the life of the process (also fixes the DLL search for ORT)
    unsafe { libloading::Library::new(lib) }.map(std::mem::forget).map_err(|e| e.to_string())
}

/// Can the CUDA execution provider work here? Loads the CUDA 12 + cuDNN 9 libraries.
fn cuda_ready(cuda_rt: &Path) -> Result<(), String> {
    let drv = unsafe { libloading::Library::new("nvcuda.dll") }.map_err(|_| "no NVIDIA driver".to_string())?;
    let mut ver = 0i32;
    let ok = unsafe {
        drv.get::<unsafe extern "system" fn(*mut i32) -> i32>(b"cuDriverGetVersion")
            .map(|f| f(&mut ver) == 0)
            .unwrap_or(false)
    };
    if !ok || ver < 12000 {
        return Err(format!("NVIDIA driver too old for CUDA 12 (supports CUDA {}.{}): update the driver", ver / 1000, ver % 1000 / 10));
    }
    std::mem::forget(drv);
    // search order: the CUDA pack, then the installed CUDA toolkit
    let mut dirs = vec![cuda_rt.to_path_buf()];
    if let Some(p) = std::env::var_os("CUDA_PATH") {
        dirs.push(PathBuf::from(p).join("bin"));
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    let mut all = dirs.clone();
    all.extend(std::env::split_paths(&path));
    if let Ok(p) = std::env::join_paths(all) {
        std::env::set_var("PATH", p);
    }
    for lib in ["cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll"] {
        let found = dirs.iter().map(|d| d.join(lib)).find(|p| p.exists());
        match found {
            Some(p) => try_load(&p).map_err(|e| format!("{lib}: {e}"))?,
            None => try_load(Path::new(lib)).map_err(|_| format!("{lib} not found"))?,
        }
    }
    Ok(())
}

struct Runtime {
    engine: Engine, // the one actually used (never Auto)
    dll: PathBuf,
    env: ort::environment::Environment,
}

fn init_runtime(want: Engine, assets: &Path) -> Result<(Runtime, Option<String>), String> {
    let cuda_dll = assets.join("runtime").join("cuda").join("onnxruntime.dll");
    let dml_dll = assets.join("runtime").join("directml").join("onnxruntime.dll");
    // The Visual C++ runtime ships in runtime\vc: a clean Windows may not have it (or an old
    // one). Loaded first, so ONNX Runtime and the CUDA libraries all use this copy.
    let vc = assets.join("runtime").join("vc");
    for dep in [
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "msvcp140.dll",
        "msvcp140_1.dll",
        "msvcp140_2.dll",
        "msvcp140_atomic_wait.dll",
        "msvcp140_codecvt_ids.dll",
    ] {
        let p = vc.join(dep);
        if p.exists() {
            try_load(&p).map_err(|e| format!("{dep}: {e}"))?;
        }
    }
    let mut note = None;
    let (engine, dll) = match want {
        Engine::Cuda | Engine::Auto if cuda_dll.exists() => match cuda_ready(cuda_dll.parent().unwrap()) {
            Ok(()) => (Engine::Cuda, cuda_dll.clone()),
            Err(e) => {
                note = Some(format!("CUDA not usable ({e})"));
                if dml_dll.exists() {
                    (Engine::DirectML, dml_dll.clone())
                } else {
                    (Engine::Cpu, cuda_dll.clone())
                }
            }
        },
        Engine::Cuda | Engine::Auto => {
            if want == Engine::Cuda {
                note = Some("CUDA pack not installed (runtime\\cuda)".into());
            }
            (Engine::DirectML, dml_dll.clone())
        }
        Engine::DirectML if !dml_dll.exists() && cuda_dll.exists() => {
            note = Some("DirectML runtime not installed".into());
            (Engine::Cpu, cuda_dll.clone())
        }
        Engine::DirectML => (Engine::DirectML, dml_dll.clone()),
        Engine::Cpu => (Engine::Cpu, if dml_dll.exists() { dml_dll.clone() } else { cuda_dll.clone() }),
    };
    if !dll.exists() {
        return Err(format!(
            "AI runtime not found. Put the 'runtime' and 'models' folders next to the exe (looked in {}).",
            assets.display()
        ));
    }
    // Windows resolves onnxruntime.dll's dependencies from the exe folder / System32, not
    // from the runtime folder. Load ours first so e.g. the old System32 DirectML.dll is not used.
    let dir = dll.parent().unwrap();
    for dep in ["DirectML.dll", "onnxruntime_providers_shared.dll"] {
        let p = dir.join(dep);
        if p.exists() {
            try_load(&p).map_err(|e| format!("{dep}: {e}"))?;
        }
    }
    let env = ort::init_from(&dll)
        .map_err(|e| format!("cannot load ONNX Runtime: {e}"))?
        .with_name("yolo-annotator")
        .build()
        .map_err(|e| e.to_string())?;
    Ok((Runtime { engine, dll, env }, note))
}

/// Session on the runtime's engine; falls back to the CPU (third value = why).
fn session(rt: &Runtime, model: &Path) -> Result<(Session, Engine, Option<String>), String> {
    use ort::ep;
    let build = |engine: Engine| -> Result<Session, String> {
        // no busy-waiting worker threads: the encoder and the per-click decoder take turns,
        // and spinning pools fight each other (and the UI) for the CPU
        let mut b = Session::builder(&rt.env)
            .map_err(|e| e.to_string())?
            .with_config_entry("session.intra_op.allow_spinning", "0")
            .map_err(|e| e.to_string())?;
        b = match engine {
            Engine::Cuda => b
                .with_execution_providers([ep::CUDA::default().build().error_on_failure()])
                .map_err(|e| e.to_string())?,
            Engine::DirectML => b
                .with_memory_pattern(false)
                .map_err(|e| e.to_string())?
                .with_parallel_execution(false)
                .map_err(|e| e.to_string())?
                .with_execution_providers([ep::DirectML::default().build().error_on_failure()])
                .map_err(|e| e.to_string())?,
            _ => b,
        };
        b.commit_from_file(model).map_err(|e| e.to_string())
    };
    match build(rt.engine) {
        Ok(s) => Ok((s, rt.engine, None)),
        Err(e) if rt.engine != Engine::Cpu => {
            // GPU provider refused (old driver, out of memory...): CPU still works
            let s = build(Engine::Cpu).map_err(|e2| format!("{e}; CPU fallback: {e2}"))?;
            let short: String = e.lines().next().unwrap_or("").chars().take(160).collect();
            Ok((s, Engine::Cpu, Some(format!("{} failed: {short}", engine_name(rt.engine)))))
        }
        Err(e) => Err(e),
    }
}

fn engine_name(e: Engine) -> &'static str {
    match e {
        Engine::Cuda => "CUDA",
        Engine::DirectML => "DirectML",
        _ => "CPU",
    }
}

// ------------------------------------------------------------------ SAM

struct Embeds {
    e0: Vec<f32>,
    e1: Vec<f32>,
    e2: Vec<f32>,
    w: u32,
    h: u32,
}

struct Sam {
    name: String,
    enc: Session,
    dec: Session,
    engine: Engine,
    fallback: Option<String>,
}

fn pick_sam(settings: &AiSettings, engine: Engine) -> Option<String> {
    let all = sam_models();
    if settings.sam != "auto" && all.contains(&settings.sam) {
        return Some(settings.sam.clone());
    }
    let pref: &[&str] = match engine {
        Engine::Cuda => &["base", "small", "tiny", "large"],
        Engine::DirectML => &["small", "tiny", "base", "large"],
        _ => &["tiny", "small", "base", "large"],
    };
    pref.iter().find_map(|k| all.iter().find(|m| m.contains(k)).cloned())
}

fn load_sam(rt: &Runtime, assets: &Path, name: &str) -> Result<Sam, String> {
    let d = assets.join("models").join(name);
    let (enc, engine, fallback) = session(rt, &d.join("vision_encoder.onnx")).map_err(|e| format!("SAM encoder: {e}"))?;
    // the decoder is tiny and runs per click: CPU avoids GPU transfer overhead of the big embeddings
    let (dec, _, _) = session(&Runtime { engine: Engine::Cpu, dll: rt.dll.clone(), env: rt.env.clone() }, &d.join("prompt_encoder_mask_decoder.onnx"))
        .map_err(|e| format!("SAM decoder: {e}"))?;
    Ok(Sam { name: name.to_string(), enc, dec, engine, fallback })
}

fn sam_encode(sam: &mut Sam, img: &AiImage) -> Result<Embeds, String> {
    let r = image::imageops::resize(&*img.rgb, SAM_SIZE as u32, SAM_SIZE as u32, image::imageops::FilterType::Triangle);
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let n = SAM_SIZE * SAM_SIZE;
    let mut px = vec![0f32; 3 * n];
    for (i, p) in r.pixels().enumerate() {
        for c in 0..3 {
            px[c * n + i] = (p[c] as f32 / 255.0 - mean[c]) / std[c];
        }
    }
    let t = Tensor::from_array(([1usize, 3, SAM_SIZE, SAM_SIZE], px)).map_err(|e| e.to_string())?;
    let out = sam.enc.run(ort::inputs!["pixel_values" => t]).map_err(|e| e.to_string())?;
    let get = |k: &str| -> Result<Vec<f32>, String> {
        let v = out.get(k).ok_or(format!("SAM encoder has no output {k}"))?;
        Ok(v.try_extract_tensor::<f32>().map_err(|e| e.to_string())?.1.to_vec())
    };
    Ok(Embeds { e0: get("image_embeddings.0")?, e1: get("image_embeddings.1")?, e2: get("image_embeddings.2")?, w: img.w, h: img.h })
}

fn sam_decode(
    sam: &mut Sam,
    em: &Embeds,
    points: &[([f32; 2], bool)],
    bbox: Option<[f32; 4]>,
    detail: f32,
    polygon: bool,
) -> Result<Option<Vec<[f32; 2]>>, String> {
    let (sx, sy) = (SAM_SIZE as f32 / em.w as f32, SAM_SIZE as f32 / em.h as f32);
    let mut pts: Vec<f32> = vec![];
    let mut lab: Vec<i64> = vec![];
    for (p, pos) in points {
        pts.extend([p[0] * sx, p[1] * sy]);
        lab.push(if *pos { 1 } else { 0 });
    }
    if pts.is_empty() {
        // box only: one padding point
        pts.extend([0.0, 0.0]);
        lab.push(-1);
    }
    let np = lab.len();
    let boxes: Vec<f32> = bbox.map(|b| vec![b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]).unwrap_or_default();
    let nb = boxes.len() / 4;
    let t = |e: ort::Error| e.to_string();
    let inputs = ort::inputs![
        "input_points" => Tensor::from_array(([1usize, 1, np, 2], pts)).map_err(t)?,
        "input_labels" => Tensor::from_array(([1usize, 1, np], lab)).map_err(t)?,
        "input_boxes" => Tensor::from_array(([1usize, nb, 4], boxes)).map_err(t)?,
        "image_embeddings.0" => Tensor::from_array(([1usize, 32, 256, 256], em.e0.clone())).map_err(t)?,
        "image_embeddings.1" => Tensor::from_array(([1usize, 64, 128, 128], em.e1.clone())).map_err(t)?,
        "image_embeddings.2" => Tensor::from_array(([1usize, 256, 64, 64], em.e2.clone())).map_err(t)?,
    ];
    let out = sam.dec.run(inputs).map_err(|e| e.to_string())?;
    let (ms, masks) = out["pred_masks"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
    let (_, iou) = out["iou_scores"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
    if ms.len() != 5 || iou.is_empty() {
        return Err(format!("unexpected SAM decoder output shape {:?}", &ms[..]));
    }
    let (nm, mh, mw) = (ms[2] as usize, ms[3] as usize, ms[4] as usize);
    if nm == 0 || mh == 0 || mw == 0 || masks.len() < nm * mh * mw || iou.len() < nm {
        return Err(format!("unexpected SAM decoder output shape {:?}", &ms[..]));
    }
    let best = (0..nm).max_by(|a, b| iou[*a].total_cmp(&iou[*b])).unwrap_or(0);
    let logits = &masks[best * mh * mw..(best + 1) * mh * mw];
    // upsample the low-res logits onto a grid with the image's aspect ratio
    let s = (1024.0 / em.w.max(em.h) as f32).min(1.0);
    let (gw, gh) = (((em.w as f32 * s).round() as usize).max(1), ((em.h as f32 * s).round() as usize).max(1));
    let mut grid = vec![false; gw * gh];
    for gy in 0..gh {
        let my = (gy as f32 + 0.5) * mh as f32 / gh as f32 - 0.5;
        for gx in 0..gw {
            let mx = (gx as f32 + 0.5) * mw as f32 / gw as f32 - 0.5;
            grid[gy * gw + gx] = bilinear(logits, mw, mh, mx, my) > 0.0;
        }
    }
    let scale = [em.w as f32 / gw as f32, em.h as f32 / gh as f32];
    if !polygon {
        return Ok(mask_bbox(&grid, gw, gh).map(|b| vec![[b[0] * scale[0], b[1] * scale[1]], [b[2] * scale[0], b[3] * scale[1]]]));
    }
    Ok(mask_to_polygon(&grid, gw, gh, scale, detail * gw as f32 / 1024.0 + 0.5, 150))
}

fn mask_bbox(m: &[bool], w: usize, h: usize) -> Option<[f32; 4]> {
    let mut b = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
    for y in 0..h {
        for x in 0..w {
            if m[y * w + x] {
                b = [b[0].min(x as f32), b[1].min(y as f32), b[2].max(x as f32 + 1.0), b[3].max(y as f32 + 1.0)];
            }
        }
    }
    (b[0] < b[2]).then_some(b)
}

fn bilinear(v: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let a = v[y0 * w + x0] * (1.0 - fx) + v[y0 * w + x1] * fx;
    let b = v[y1 * w + x0] * (1.0 - fx) + v[y1 * w + x1] * fx;
    a * (1.0 - fy) + b * fy
}

// ------------------------------------------------------------------ YOLO

struct Yolo {
    path: String,
    s: Session,
    input: String,
    /// network input (height, width)
    size: (usize, usize),
    names: Vec<String>,
}

/// "{0: 'person', 1: \"car's\"}" -> ["person", "car's"]
pub fn parse_names(s: &str) -> Vec<String> {
    let mut out: Vec<(usize, String)> = vec![];
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            let st = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            let id: usize = s[st..i].parse().unwrap_or(0);
            while i < b.len() && (b[i] == b':' || b[i] == b' ') {
                i += 1;
            }
            if i < b.len() && (b[i] == b'\'' || b[i] == b'"') {
                let q = b[i];
                i += 1;
                let st = i;
                while i < b.len() && b[i] != q {
                    i += 1;
                }
                out.push((id, s[st..i.min(b.len())].to_string()));
            }
        }
        i += 1;
    }
    out.sort_by_key(|x| x.0);
    let n = out.last().map_or(0, |x| x.0 + 1);
    let mut v: Vec<String> = (0..n).map(|i| format!("class_{i}")).collect();
    for (id, name) in out {
        v[id] = name;
    }
    v
}

fn load_yolo(rt: &Runtime, path: &str) -> Result<Yolo, String> {
    let (s, _, _) = session(rt, Path::new(path))?;
    let input = s.inputs().first().ok_or("model has no input")?.name().to_string();
    let meta = s.metadata().map_err(|e| e.to_string())?;
    let names = meta.custom("names").map(|n| parse_names(&n)).unwrap_or_default();
    let dim = |i: usize| {
        s.inputs()[0].dtype().tensor_shape().and_then(|sh| sh.get(i).copied()).filter(|d| *d > 0).map(|d| d as usize)
    };
    let size = match (dim(2), dim(3)) {
        (Some(h), Some(w)) => (h, w),
        _ => {
            // dynamic input: Ultralytics stores imgsz in the metadata, e.g. "[640, 640]" (h, w)
            let v: Vec<usize> = meta
                .custom("imgsz")
                .map(|v| v.trim_matches(|c| c == '[' || c == ']').split(',').filter_map(|x| x.trim().parse().ok()).collect())
                .unwrap_or_default();
            match v.as_slice() {
                [h, w] => (*h, *w),
                [s] => (*s, *s),
                _ => (640, 640),
            }
        }
    };
    drop(meta);
    Ok(Yolo { path: path.to_string(), s, input, size, names })
}

fn yolo_detect(y: &mut Yolo, img: &AiImage, conf: f32, iou_thr: f32, detail: f32) -> Result<Vec<Detection>, String> {
    let (aw, ah) = (img.rgb.width() as f32, img.rgb.height() as f32);
    let (sh, sw) = y.size;
    // letterbox (centered, gray 114) exactly like Ultralytics
    let r = (sw as f32 / aw).min(sh as f32 / ah);
    let (nw, nh) = (((aw * r).round() as u32).clamp(1, sw as u32), ((ah * r).round() as u32).clamp(1, sh as u32));
    let (px, py) = ((sw as u32 - nw) / 2, (sh as u32 - nh) / 2);
    let rs = image::imageops::resize(&*img.rgb, nw, nh, image::imageops::FilterType::Triangle);
    let n = sh * sw;
    let mut data = vec![114.0 / 255.0; 3 * n];
    for (x, yy, p) in rs.enumerate_pixels() {
        let i = (yy + py) as usize * sw + (x + px) as usize;
        for c in 0..3 {
            data[c * n + i] = p[c] as f32 / 255.0;
        }
    }
    let t = Tensor::from_array(([1usize, 3, sh, sw], data)).map_err(|e| e.to_string())?;
    let input = y.input.clone();
    let out = y.s.run(ort::inputs![input => t]).map_err(|e| e.to_string())?;
    let (s0, o0) = out[0].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
    let protos = if out.len() > 1 {
        let (sp, p) = out[1].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
        (sp.len() == 4).then(|| (sp[1] as usize, sp[2] as usize, sp[3] as usize, p.to_vec()))
    } else {
        None
    };
    let nmc = protos.as_ref().map_or(0, |p| p.0);
    if s0.len() != 3 {
        return Err(format!("unexpected YOLO output shape {:?}", &s0[..]));
    }
    let (d1, d2) = (s0[1] as usize, s0[2] as usize);
    // to AI-image coordinates
    let unbox = |b: [f32; 4]| {
        [
            ((b[0] - px as f32) / r).clamp(0.0, aw),
            ((b[1] - py as f32) / r).clamp(0.0, ah),
            ((b[2] - px as f32) / r).clamp(0.0, aw),
            ((b[3] - py as f32) / r).clamp(0.0, ah),
        ]
    };
    let mut boxes = vec![];
    let mut scores = vec![];
    let mut classes = vec![];
    let mut coefs: Vec<Vec<f32>> = vec![];
    let end2end = d2 == 6 + nmc && d1 > d2;
    if end2end {
        // YOLO26 / YOLOv10 style: [1, max_det, x1 y1 x2 y2 score class (+mask coefs)], NMS done
        for k in 0..d1 {
            let row = &o0[k * d2..(k + 1) * d2];
            if row[4] < conf {
                continue;
            }
            boxes.push([row[0], row[1], row[2], row[3]]);
            scores.push(row[4]);
            classes.push(row[5].max(0.0) as usize);
            coefs.push(row[6..].to_vec());
        }
    } else {
        // YOLOv8 / 11 style: [1, 4 + classes (+mask coefs), anchors], needs NMS
        if d1 < 5 + nmc || d1 > d2 {
            // e.g. YOLOv5 exports ([1, anchors, 85]) have a different layout
            return Err(format!(
                "unsupported YOLO output shape {:?}: export the model with Ultralytics YOLOv8 / YOLO11 / YOLO26 (yolo export format=onnx)",
                &s0[..]
            ));
        }
        let nc = d1 - 4 - nmc;
        let at = |c: usize, a: usize| o0[c * d2 + a];
        for a in 0..d2 {
            let (mut bc, mut bs) = (0, f32::MIN);
            for c in 0..nc {
                let v = at(4 + c, a);
                if v > bs {
                    bs = v;
                    bc = c;
                }
            }
            if bs < conf {
                continue;
            }
            let (cx, cy, w, h) = (at(0, a), at(1, a), at(2, a), at(3, a));
            boxes.push([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]);
            scores.push(bs);
            classes.push(bc);
            coefs.push((0..nmc).map(|k| at(4 + nc + k, a)).collect());
        }
    }
    let keep = if end2end { (0..boxes.len()).collect() } else { nms(&boxes, &scores, &classes, iou_thr, 300) };
    let to_img = [img.w as f32 / aw, img.h as f32 / ah];
    let mut dets = vec![];
    for k in keep {
        let lb = boxes[k]; // letterbox coordinates
        let b = unbox(lb);
        if b[2] - b[0] < 1.0 || b[3] - b[1] < 1.0 {
            continue;
        }
        let polygon = protos.as_ref().and_then(|(nm, ph, pw, p)| {
            yolo_mask(p, *nm, *ph, *pw, &coefs[k], lb, (sh, sw), [px as f32, py as f32], r, [aw, ah], detail)
        });
        let id = classes[k];
        dets.push(Detection {
            class_id: id,
            class_name: y.names.get(id).cloned(),
            score: scores[k],
            bbox: [b[0] * to_img[0], b[1] * to_img[1], b[2] * to_img[0], b[3] * to_img[1]],
            polygon: polygon.map(|p| p.iter().map(|q| [q[0] * to_img[0], q[1] * to_img[1]]).collect()),
        });
    }
    Ok(dets)
}

/// Instance mask of one detection -> polygon in AI-image coordinates.
#[allow(clippy::too_many_arguments)]
fn yolo_mask(
    protos: &[f32],
    nm: usize,
    ph: usize,
    pw: usize,
    coef: &[f32],
    lb: [f32; 4],
    (sh, sw): (usize, usize),
    pad: [f32; 2],
    r: f32,
    ai: [f32; 2],
    detail: f32,
) -> Option<Vec<[f32; 2]>> {
    if coef.len() != nm {
        return None;
    }
    // mask logits at proto resolution, only inside the (slightly grown) box
    let (fx, fy) = (pw as f32 / sw as f32, ph as f32 / sh as f32);
    let x0 = ((lb[0] * fx).floor() as isize - 1).max(0) as usize;
    let y0 = ((lb[1] * fy).floor() as isize - 1).max(0) as usize;
    let x1 = ((lb[2] * fx).ceil() as usize + 1).min(pw);
    let y1 = ((lb[3] * fy).ceil() as usize + 1).min(ph);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let (cw, ch) = (x1 - x0, y1 - y0);
    let mut lg = vec![0f32; cw * ch];
    for yy in 0..ch {
        for xx in 0..cw {
            let pi = (y0 + yy) * pw + (x0 + xx);
            let mut s = 0.0;
            for k in 0..nm {
                s += coef[k] * protos[k * ph * pw + pi];
            }
            lg[yy * cw + xx] = s;
        }
    }
    // sample on a grid over the box in AI-image pixels (at most ~512 px across)
    let bx0 = ((lb[0] - pad[0]) / r).clamp(0.0, ai[0]);
    let by0 = ((lb[1] - pad[1]) / r).clamp(0.0, ai[1]);
    let bx1 = ((lb[2] - pad[0]) / r).clamp(0.0, ai[0]);
    let by1 = ((lb[3] - pad[1]) / r).clamp(0.0, ai[1]);
    let st = ((bx1 - bx0).max(by1 - by0) / 512.0).max(1.0);
    let gw = (((bx1 - bx0) / st).ceil() as usize).max(1);
    let gh = (((by1 - by0) / st).ceil() as usize).max(1);
    let mut grid = vec![false; gw * gh];
    for gy in 0..gh {
        let iy = by0 + (gy as f32 + 0.5) * st; // AI image
        let my = (iy * r + pad[1]) * fy - 0.5 - y0 as f32;
        for gx in 0..gw {
            let ix = bx0 + (gx as f32 + 0.5) * st;
            let mx = (ix * r + pad[0]) * fx - 0.5 - x0 as f32;
            grid[gy * gw + gx] = bilinear(&lg, cw, ch, mx, my) > 0.0;
        }
    }
    let poly = mask_to_polygon(&grid, gw, gh, [st, st], detail / st + 0.5, 150)?;
    Some(poly.iter().map(|p| [p[0] + bx0, p[1] + by0]).collect())
}

// ------------------------------------------------------------------ worker loop

struct Batch {
    id: u64,
    items: VecDeque<(String, PathBuf)>,
    decode: DecodeFn,
    cancel: Arc<AtomicBool>,
    done: Arc<AtomicUsize>,
    labeled: usize,
    errors: Vec<String>,
}

/// AI unusable: keep answering requests that the UI waits for, so it never hangs.
fn refuse(rx: &mpsc::Receiver<Req>, send: &dyn Fn(Resp), why: &str) {
    while let Ok(r) = rx.recv() {
        match r {
            Req::Batch { id, .. } => send(Resp::BatchDone { id, labeled: 0, errors: vec![why.to_string()], cancelled: false }),
            Req::Detect { .. } => send(Resp::Error(why.to_string())),
            _ => {}
        }
    }
}

fn worker(mut settings: AiSettings, rx: mpsc::Receiver<Req>, send: &dyn Fn(Resp)) {
    let Some(assets) = assets_dir() else {
        send(Resp::Error("AI files not found: put the 'runtime' and 'models' folders next to yolo-annotator.exe (see README).".into()));
        refuse(&rx, send, "AI files not found");
        return;
    };
    let (rt, note) = match init_runtime(settings.engine, &assets) {
        Ok(x) => x,
        Err(e) => {
            send(Resp::Error(e.clone()));
            refuse(&rx, send, &e);
            return;
        }
    };
    let mut sam: Option<Sam> = None;
    let mut sam_err: Option<String> = None;
    let mut yolo: Option<Yolo> = None;
    let mut cache: VecDeque<(String, Embeds)> = VecDeque::new();
    let mut pending: VecDeque<Req> = VecDeque::new();
    let mut batch: Option<Batch> = None;

    let describe = |sam: &Option<Sam>, note: &Option<String>| {
        let mut s = match sam {
            Some(m) => format!("{} · {}", m.name.replace("sam2.1-", "SAM 2.1 "), engine_name(m.engine)),
            None => format!("ONNX Runtime · {}", engine_name(rt.engine)),
        };
        if let Some(n) = note {
            s.push_str(&format!(" · {n}"));
        }
        if let Some(f) = sam.as_ref().and_then(|m| m.fallback.as_ref()) {
            s.push_str(&format!(" · {f}"));
        }
        s
    };
    let ensure_sam = |sam: &mut Option<Sam>, sam_err: &mut Option<String>, settings: &AiSettings| {
        if sam.is_some() || sam_err.is_some() {
            return;
        }
        match pick_sam(settings, rt.engine) {
            None => *sam_err = Some("no SAM model found in the models folder".into()),
            Some(name) => match load_sam(&rt, &assets, &name) {
                Ok(m) => {
                    *sam = Some(m);
                    send(Resp::Ready(describe(sam, &note)));
                }
                Err(e) => *sam_err = Some(e),
            },
        }
        if let Some(e) = sam_err {
            send(Resp::Unavailable(e.clone()));
        }
    };
    ensure_sam(&mut sam, &mut sam_err, &settings);

    loop {
        // gather everything queued; block only when idle
        if pending.is_empty() && batch.is_none() {
            match rx.recv() {
                Ok(r) => pending.push_back(r),
                Err(_) => return,
            }
        }
        while let Ok(r) = rx.try_recv() {
            pending.push_back(r);
        }
        // only the newest prompt matters; only the two newest encodes (current + next image)
        if let Some(last_seg) = pending.iter().rposition(|r| matches!(r, Req::Segment { .. })) {
            let mut i = 0;
            pending.retain(|r| {
                let keep = !matches!(r, Req::Segment { .. }) || i == last_seg;
                i += 1;
                keep
            });
        }
        let enc_count = pending.iter().filter(|r| matches!(r, Req::Encode { .. })).count();
        if enc_count > 2 {
            let mut skip = enc_count - 2;
            let mut dropped = vec![];
            pending.retain(|r| match r {
                Req::Encode { key, .. } if skip > 0 => {
                    skip -= 1;
                    dropped.push(key.clone());
                    false
                }
                _ => true,
            });
            // the UI asks again if that image is still the current one
            for k in dropped {
                send(Resp::NeedEncode(k));
            }
        }
        // settings first (a Detail change must apply to the prompt sent right after it),
        // then prompts: they are what the user is waiting for
        let idx = pending
            .iter()
            .position(|r| matches!(r, Req::Settings(_)))
            .or_else(|| pending.iter().position(|r| matches!(r, Req::Segment { .. })))
            .unwrap_or(0);
        let Some(req) = pending.remove(idx) else {
            // idle: one batch item
            if let Some(b) = &mut batch {
                let finished = b.cancel.load(Ordering::Relaxed) || b.items.is_empty();
                if !finished {
                    let (key, path) = b.items.pop_front().unwrap();
                    let res = ensure_yolo(&mut yolo, &rt, &settings).and_then(|_| (b.decode)(&path)).and_then(|img| {
                        let y = yolo.as_mut().ok_or("no YOLO model")?;
                        yolo_detect(y, &img, settings.conf, settings.iou, settings.detail).map(|d| (d, img))
                    });
                    match res {
                        Ok((dets, img)) => {
                            if !dets.is_empty() {
                                b.labeled += 1;
                            }
                            send(Resp::Detected { key, batch: Some(b.id), dets, w: img.w, h: img.h });
                        }
                        Err(e) => {
                            if b.errors.len() < 5 {
                                b.errors.push(format!("{key}: {e}"));
                            }
                        }
                    }
                    b.done.fetch_add(1, Ordering::Relaxed);
                } else {
                    let b = batch.take().unwrap();
                    send(Resp::BatchDone { id: b.id, labeled: b.labeled, errors: b.errors, cancelled: b.cancel.load(Ordering::Relaxed) });
                }
            }
            continue;
        };
        match req {
            Req::Settings(s) => {
                if s.sam != settings.sam {
                    sam = None;
                    sam_err = None;
                    cache.clear();
                }
                if s.yolo != settings.yolo {
                    yolo = None;
                }
                let engine_changed = s.engine != settings.engine;
                settings = s;
                if engine_changed {
                    send(Resp::Info("The new AI engine is used after restarting the app.".into()));
                }
                ensure_sam(&mut sam, &mut sam_err, &settings);
            }
            Req::Encode { key, img } => {
                if cache.iter().any(|c| c.0 == key) {
                    send(Resp::Encoded { key, ms: 0 });
                    continue;
                }
                ensure_sam(&mut sam, &mut sam_err, &settings);
                let Some(m) = sam.as_mut() else {
                    let err = sam_err.clone().unwrap_or_else(|| "SAM is not loaded".into());
                    send(Resp::EncodeFailed { key, err });
                    continue;
                };
                let t = Instant::now();
                match sam_encode(m, &img) {
                    Ok(e) => {
                        cache.push_back((key.clone(), e));
                        while cache.len() > EMBED_CACHE {
                            cache.pop_front();
                        }
                        send(Resp::Encoded { key, ms: t.elapsed().as_millis() });
                    }
                    Err(e) => send(Resp::EncodeFailed { key, err: e }),
                }
            }
            Req::Segment { key, seq, points, bbox, polygon } => {
                let Some(m) = sam.as_mut() else {
                    send(Resp::Mask { key, seq, shape: None });
                    continue;
                };
                let Some(em) = cache.iter().find(|c| c.0 == key).map(|c| &c.1) else {
                    send(Resp::NeedEncode(key));
                    continue;
                };
                match sam_decode(m, em, &points, bbox, settings.detail, polygon) {
                    Ok(shape) => send(Resp::Mask { key, seq, shape }),
                    Err(e) => send(Resp::Error(format!("SAM: {e}"))),
                }
            }
            Req::Detect { key, img } => {
                if let Err(e) = ensure_yolo(&mut yolo, &rt, &settings) {
                    send(Resp::Error(e));
                    continue;
                }
                let y = yolo.as_mut().unwrap();
                match yolo_detect(y, &img, settings.conf, settings.iou, settings.detail) {
                    Ok(dets) => send(Resp::Detected { key, batch: None, dets, w: img.w, h: img.h }),
                    Err(e) => send(Resp::Error(format!("YOLO: {e}"))),
                }
            }
            Req::Batch { id, items, decode, cancel, done } => {
                if let Err(e) = ensure_yolo(&mut yolo, &rt, &settings) {
                    send(Resp::BatchDone { id, labeled: 0, errors: vec![e], cancelled: false });
                    continue;
                }
                if let Some(old) = batch.take() {
                    // replaced by a new batch: tell the UI the old one ended
                    send(Resp::BatchDone { id: old.id, labeled: old.labeled, errors: old.errors, cancelled: true });
                }
                batch = Some(Batch { id, items: items.into(), decode, cancel, done, labeled: 0, errors: vec![] });
            }
        }
    }
}

fn ensure_yolo(yolo: &mut Option<Yolo>, rt: &Runtime, settings: &AiSettings) -> Result<(), String> {
    if settings.yolo.trim().is_empty() {
        return Err("Choose a YOLO model (.onnx) in the AI panel first.".into());
    }
    if yolo.as_ref().map_or(true, |y| y.path != settings.yolo) {
        *yolo = Some(load_yolo(rt, &settings.yolo).map_err(|e| format!("YOLO model: {e}"))?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_metadata() {
        assert_eq!(parse_names("{0: 'person', 1: 'bi cycle', 2: \"car's\"}"), vec!["person", "bi cycle", "car's"]);
        assert_eq!(parse_names("{1: 'b'}"), vec!["class_0", "b"]);
        assert!(parse_names("").is_empty());
    }
}
