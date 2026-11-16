//! Platform-independent core of YOLO Annotator, shared by the desktop app, the lab server and
//! the web client: annotation model, YOLO label text, dataset export, mask -> polygon, and the
//! messages the parts exchange.
pub mod dataset;
pub mod imageio;
pub mod mask;
pub mod model;
pub mod proto;
