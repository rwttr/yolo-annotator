#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
//! YOLO Annotator desktop app: the shared UI (annot-ui) on local folders.

mod local_store;

use annot_ai as ai;
use eframe::egui;
use std::fs;
use std::path::Path;

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
    let img = match ai::decode_ai(Path::new(path)) {
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
    if let Ok(src) = annot_core::imageio::decode_path(Path::new(path)) {
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
            .with_icon(egui::IconData { rgba: annot_ui::icon::rgba(64), width: 64, height: 64 })
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "YOLO Annotator",
        options,
        Box::new(|cc| {
            let store = Box::new(local_store::LocalStore::new(cc.egui_ctx.clone()));
            Ok(Box::new(annot_ui::App::new(cc, store, args.first().cloned())))
        }),
    )
}
