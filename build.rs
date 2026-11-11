// Embeds the program icon and version info (DotoRobotics, MIT) into the Windows .exe.
fn main() {
    println!("cargo:rerun-if-changed=assets/icon.ico");
    // (missing on the very first build: create it with `cargo run --example gen_icon`)
    let have_icon = std::path::Path::new("assets/icon.ico").exists();
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") && have_icon {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/icon.ico")
            .set("ProductName", "YOLO Annotator")
            .set("FileDescription", "YOLO Annotator - image annotation for Ultralytics YOLO")
            .set("CompanyName", "DotoRobotics")
            .set("LegalCopyright", "Copyright (c) 2026 DotoRobotics. MIT License.");
        if let Err(e) = res.compile() {
            println!("cargo:warning=could not embed the icon/version resource: {e}");
        }
    }
}
