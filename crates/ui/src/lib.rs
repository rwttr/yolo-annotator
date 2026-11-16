//! The YOLO Annotator user interface (egui). The same code runs in the Windows desktop app
//! (local folders) and in the browser (lab server), see [`store::Store`].
pub mod app;
pub mod icon;
pub mod store;

pub use app::App;
