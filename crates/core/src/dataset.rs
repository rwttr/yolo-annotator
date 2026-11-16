//! Opening a dataset folder from disk (desktop app and lab server).

use crate::model::*;
use crate::proto::OpenedData;
use std::fs;
use std::path::Path;

#[derive(Default)]
struct ImportStats {
    new: usize,
    refreshed: usize,
    box_lines: usize,
    seg_lines: usize,
}

/// A dataset read from disk. `dirty`: the project differs from annotations.json (labels were
/// imported) and should be saved.
pub struct Loaded {
    pub data: OpenedData,
    pub dirty: bool,
}

/// Scan `dir` for images, load annotations.json (or start a new project with the classes of
/// data.yaml / classes.txt) and import existing YOLO label files.
pub fn open_dataset(dir: &Path) -> Result<Loaded, String> {
    let files = scan_images(dir)?;
    let mut warn = None;
    let mut fresh = false;
    let new_project = || {
        let mut p = Project::default();
        if let Some(c) = find_classes(dir) {
            p.classes = c;
        }
        p
    };
    let mut project = match Project::load(dir) {
        Ok(Some(p)) => p,
        Ok(None) => {
            fresh = true;
            new_project()
        }
        Err(LoadError::Unreadable(e)) => return Err(e),
        Err(LoadError::Corrupt(e)) => {
            warn = Some(e);
            fresh = true;
            new_project()
        }
    };
    let project_mtime = fs::metadata(dir.join(PROJECT_FILE)).and_then(|m| m.modified()).ok();
    let imp = import_existing_labels(&mut project, dir, &files, project_mtime);
    // an existing detection-only dataset stays in detection format
    if fresh && imp.box_lines > 0 && imp.seg_lines == 0 {
        project.export_mode = ExportMode::Detect;
    }
    let labeled = files.iter().filter(|f| project.images.contains_key(*f)).count();
    let mut status = warn.unwrap_or_else(|| {
        format!(
            "Opened {} images in {} ({} labeled{}). Label files are created next to each image folder (images -> labels).",
            files.len(),
            dir.display(),
            labeled,
            match (imp.new, imp.refreshed) {
                (0, 0) => String::new(),
                (n, 0) => format!(", {n} imported from existing YOLO .txt files"),
                (n, r) => format!(", {n} imported, {r} re-read from .txt files newer than {PROJECT_FILE}"),
            }
        )
    });
    let clashes = label_collisions(dir, &files);
    if let Some((a, b)) = clashes.first() {
        status = format!(
            "WARNING: {} image pair(s) share a label file name, e.g. '{a}' and '{b}'. Rename one of them, or their labels overwrite each other. {status}",
            clashes.len()
        );
    }
    Ok(Loaded {
        dirty: imp.new + imp.refreshed > 0,
        data: OpenedData { label: dir.display().to_string(), files, project, status },
    })
}

/// Read existing YOLO label files for every image not in the project yet, so the whole
/// dataset (not only visited images) is known and exported. Label files written after
/// annotations.json (e.g. the app was killed before it saved the project file, or the .txt
/// was edited elsewhere) are re-read as well.
fn import_existing_labels(
    project: &mut Project,
    root: &Path,
    files: &[String],
    project_mtime: Option<std::time::SystemTime>,
) -> ImportStats {
    let mut st = ImportStats::default();
    let grace = std::time::Duration::from_secs(2);
    for rel in files {
        let lp = label_path(root, rel);
        let existing = project.images.get(rel).map(|a| (a.width, a.height));
        if existing.is_some() {
            let newer = match (project_mtime, fs::metadata(&lp).and_then(|m| m.modified())) {
                (Some(pt), Ok(lt)) => lt > pt + grace,
                _ => false,
            };
            if !newer {
                continue;
            }
        }
        let Ok(text) = fs::read_to_string(&lp) else { continue };
        let (w, h) = match existing {
            Some(wh) => wh,
            // header-only read; the exact (EXIF-rotated) size is fixed when the image is opened
            None => match image_size(&root.join(rel)) {
                Some(wh) => wh,
                None => continue,
            },
        };
        for line in text.lines() {
            match line.split_whitespace().count() {
                0 => {}
                5 => st.box_lines += 1,
                _ => st.seg_lines += 1,
            }
        }
        let shapes = parse_yolo(&text, w, h);
        project.images.insert(rel.clone(), ImageAnn { width: w, height: h, shapes });
        if existing.is_some() {
            st.refreshed += 1;
        } else {
            st.new += 1;
        }
    }
    project.ensure_classes();
    st
}

/// Image size from the file header, without decoding the image.
pub fn image_size(path: &Path) -> Option<(u32, u32)> {
    image::image_dimensions(path).ok()
}
