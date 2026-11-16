// Embeds the program icon and version info into the server .exe (also used by its shortcuts).
fn main() {
    println!("cargo:rerun-if-changed=../../assets/icon.ico");
    let icon = std::path::Path::new("../../assets/icon.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") && icon.exists() {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("../../assets/icon.ico")
            .set("ProductName", "YOLO Annotator lab server")
            .set("FileDescription", "YOLO Annotator lab server - team annotation in the browser")
            .set("CompanyName", "DotoRobotics")
            .set("LegalCopyright", "Copyright (c) 2026 DotoRobotics. MIT License.");
        if let Err(e) = res.compile() {
            println!("cargo:warning=could not embed the icon/version resource: {e}");
        }
    }
}
