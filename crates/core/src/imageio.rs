//! Image decoding shared by the desktop app, the lab server and the web client.

use image::{DynamicImage, ImageDecoder};
use std::io::Cursor;
use std::path::Path;

/// Display textures are limited to this size (GPU texture limit).
pub const MAX_TEX: u32 = 8192;

/// Decode image file bytes, honouring the EXIF rotation the same way OpenCV / Ultralytics do.
/// A decoder panic on a corrupt file becomes an error.
pub fn decode_bytes(bytes: &[u8]) -> Result<DynamicImage, String> {
    std::panic::catch_unwind(|| {
        let reader = image::ImageReader::new(Cursor::new(bytes)).with_guessed_format().map_err(|e| e.to_string())?;
        let mut dec = reader.into_decoder().map_err(|e| e.to_string())?;
        let orient = dec.orientation().unwrap_or(image::metadata::Orientation::NoTransforms);
        let mut img = DynamicImage::from_decoder(dec).map_err(|e| e.to_string())?;
        img.apply_orientation(orient);
        Ok(img)
    })
    .unwrap_or_else(|_| Err("the image decoder crashed on this file (corrupt image?)".into()))
}

pub fn decode_path(path: &Path) -> Result<DynamicImage, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    decode_bytes(&bytes)
}

/// An image ready for a texture: RGBA pixels (at most MAX_TEX per side) + the full size.
pub struct DisplayImage {
    pub rgba: Vec<u8>,
    /// size of `rgba`
    pub size: [usize; 2],
    /// size of the original image (annotation coordinates)
    pub w: u32,
    pub h: u32,
}

pub fn display_image(img: &DynamicImage) -> DisplayImage {
    let (w, h) = (img.width(), img.height());
    let rgba = if w.max(h) > MAX_TEX {
        img.resize(MAX_TEX, MAX_TEX, image::imageops::FilterType::Triangle).to_rgba8()
    } else {
        img.to_rgba8()
    };
    DisplayImage { size: [rgba.width() as usize, rgba.height() as usize], rgba: rgba.into_raw(), w, h }
}
