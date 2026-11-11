//! Regenerates assets/icon.ico (16..256 px) and assets/icon.png from src/icon.rs.
//! Run: cargo run --example gen_icon
#[path = "../src/icon.rs"]
mod icon;

use image::codecs::ico::{IcoEncoder, IcoFrame};
use image::ExtendedColorType;

fn main() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
    std::fs::create_dir_all(&dir).unwrap();
    let frames: Vec<IcoFrame> = [16u32, 24, 32, 48, 64, 128, 256]
        .iter()
        .map(|&s| IcoFrame::as_png(&icon::rgba(s), s, s, ExtendedColorType::Rgba8).unwrap())
        .collect();
    let f = std::fs::File::create(dir.join("icon.ico")).unwrap();
    IcoEncoder::new(f).encode_images(&frames).unwrap();
    image::save_buffer(dir.join("icon.png"), &icon::rgba(256), 256, 256, ExtendedColorType::Rgba8)
        .unwrap();
    println!("wrote {}", dir.display());
}
