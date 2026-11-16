//! Binary mask -> polygon: keep the largest region, trace its outer boundary along pixel
//! edges, then simplify (Douglas-Peucker) to an editable number of points.

use std::collections::HashMap;

/// Largest 8-connected region of `m` (w*h, row-major). Returns its pixels as a new mask.
pub fn largest_component(m: &[bool], w: usize, h: usize) -> Option<Vec<bool>> {
    let mut label = vec![0u32; w * h];
    let (mut best, mut best_n, mut next) = (0u32, 0usize, 0u32);
    let mut stack = vec![];
    for start in 0..w * h {
        if !m[start] || label[start] != 0 {
            continue;
        }
        next += 1;
        label[start] = next;
        stack.push(start);
        let mut n = 0;
        while let Some(i) = stack.pop() {
            n += 1;
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            for dy in -1..=1i64 {
                for dx in -1..=1i64 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if m[j] && label[j] == 0 {
                        label[j] = next;
                        stack.push(j);
                    }
                }
            }
        }
        if n > best_n {
            best_n = n;
            best = next;
        }
    }
    (best_n > 0).then(|| label.iter().map(|l| *l == best).collect())
}

/// Outer boundary of a mask (one region) as lattice points (pixel corners), clockwise on
/// screen. Holes are ignored (YOLO polygons cannot have holes).
pub fn outer_boundary(m: &[bool], w: usize, h: usize) -> Option<Vec<[i32; 2]>> {
    let at = |x: i32, y: i32| x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && m[y as usize * w + x as usize];
    // directed boundary edges with the region on their right (y down)
    let mut edges: Vec<([i32; 2], [i32; 2])> = vec![];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            if !at(x, y) {
                continue;
            }
            if !at(x, y - 1) {
                edges.push(([x, y], [x + 1, y]));
            }
            if !at(x + 1, y) {
                edges.push(([x + 1, y], [x + 1, y + 1]));
            }
            if !at(x, y + 1) {
                edges.push(([x + 1, y + 1], [x, y + 1]));
            }
            if !at(x - 1, y) {
                edges.push(([x, y + 1], [x, y]));
            }
        }
    }
    if edges.is_empty() {
        return None;
    }
    let mut from: HashMap<[i32; 2], Vec<usize>> = HashMap::with_capacity(edges.len());
    for (i, e) in edges.iter().enumerate() {
        from.entry(e.0).or_default().push(i);
    }
    let mut used = vec![false; edges.len()];
    let mut best: Option<(f64, Vec<[i32; 2]>)> = None;
    for s in 0..edges.len() {
        if used[s] {
            continue;
        }
        let mut lp = vec![];
        let mut e = s;
        loop {
            used[e] = true;
            let (a, b) = edges[e];
            lp.push(a);
            let din = [b[0] - a[0], b[1] - a[1]];
            // at a pinch point two edges leave the vertex: take the right turn, which keeps
            // diagonal-only neighbours apart and always closes the loop
            let Some(cands) = from.get(&b) else { break };
            let next = cands
                .iter()
                .copied()
                .filter(|c| !used[*c])
                .max_by_key(|c| {
                    let d = [edges[*c].1[0] - b[0], edges[*c].1[1] - b[1]];
                    din[0] * d[1] - din[1] * d[0] // cross > 0: right turn (y down)
                });
            match next {
                Some(n) => e = n,
                None => break,
            }
        }
        let area = shoelace(&lp).abs();
        if best.as_ref().map_or(true, |b| area > b.0) {
            best = Some((area, lp));
        }
    }
    best.map(|b| b.1)
}

fn shoelace(p: &[[i32; 2]]) -> f64 {
    let mut a = 0i64;
    for i in 0..p.len() {
        let (q, r) = (p[i], p[(i + 1) % p.len()]);
        a += q[0] as i64 * r[1] as i64 - r[0] as i64 * q[1] as i64;
    }
    a as f64 / 2.0
}

fn seg_dist(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 { (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
    ((a[0] + t * dx - p[0]).powi(2) + (a[1] + t * dy - p[1]).powi(2)).sqrt()
}

/// Douglas-Peucker on an open chain, endpoints kept; pushes all but the last point.
/// Iterative (explicit stack): ragged masks give chains of 100k+ points, which would
/// overflow the thread stack with recursion.
fn dp(p: &[[f32; 2]], eps: f32, out: &mut Vec<[f32; 2]>) {
    // ranges [lo, hi] still to split, processed left to right
    let mut stack = vec![(0usize, p.len() - 1)];
    while let Some((lo, hi)) = stack.pop() {
        let (a, b) = (p[lo], p[hi]);
        let mut far = (lo, 0.0f32);
        for i in lo + 1..hi {
            let d = seg_dist(p[i], a, b);
            if d > far.1 {
                far = (i, d);
            }
        }
        if far.1 > eps {
            // right half after the left half
            stack.push((far.0, hi));
            stack.push((lo, far.0));
        } else {
            out.push(a);
        }
    }
}

/// Douglas-Peucker for a closed ring.
pub fn simplify_closed(ring: &[[f32; 2]], eps: f32) -> Vec<[f32; 2]> {
    if ring.len() < 4 {
        return ring.to_vec();
    }
    // split at the point farthest from the first one
    let far = (1..ring.len())
        .max_by(|i, j| {
            let d = |k: usize| (ring[k][0] - ring[0][0]).powi(2) + (ring[k][1] - ring[0][1]).powi(2);
            d(*i).total_cmp(&d(*j))
        })
        .unwrap();
    let mut out = vec![];
    dp(&ring[..=far], eps, &mut out);
    let mut tail: Vec<[f32; 2]> = ring[far..].to_vec();
    tail.push(ring[0]);
    dp(&tail, eps, &mut out);
    out
}

/// Mask on a `w`x`h` grid -> polygon in image coordinates (grid point * scale).
/// `eps` is the simplification tolerance in grid pixels; it grows until the polygon has at
/// most `max_pts` points.
pub fn mask_to_polygon(
    m: &[bool],
    w: usize,
    h: usize,
    scale: [f32; 2],
    eps: f32,
    max_pts: usize,
) -> Option<Vec<[f32; 2]>> {
    let comp = largest_component(m, w, h)?;
    let ring = outer_boundary(&comp, w, h)?;
    // drop collinear lattice points first (staircase runs)
    let ring: Vec<[f32; 2]> = ring.iter().map(|p| [p[0] as f32, p[1] as f32]).collect();
    let mut e = eps.max(0.3);
    let mut poly = simplify_closed(&ring, e);
    while poly.len() > max_pts && e < 50.0 {
        e *= 1.4;
        poly = simplify_closed(&ring, e);
    }
    if poly.len() < 3 {
        return None;
    }
    Some(poly.iter().map(|p| [p[0] * scale[0], p[1] * scale[1]]).collect())
}

/// Intersection over union of two [x1, y1, x2, y2] boxes.
pub fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let i = iw * ih;
    let u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
    if u > 0.0 { i / u } else { 0.0 }
}

/// Class-aware non-maximum suppression. Returns kept indices, best score first.
pub fn nms(boxes: &[[f32; 4]], scores: &[f32], classes: &[usize], thr: f32, max_det: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]));
    let mut keep: Vec<usize> = vec![];
    for i in order {
        if keep.len() >= max_det {
            break;
        }
        if keep.iter().all(|k| classes[*k] != classes[i] || iou(boxes[*k], boxes[i]) <= thr) {
            keep.push(i);
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(w: usize, h: usize, f: impl Fn(usize, usize) -> bool) -> Vec<bool> {
        (0..w * h).map(|i| f(i % w, i / w)).collect()
    }

    #[test]
    fn rectangle_becomes_four_points() {
        let m = grid(40, 30, |x, y| (10..30).contains(&x) && (5..20).contains(&y));
        let p = mask_to_polygon(&m, 40, 30, [1.0, 1.0], 0.5, 100).unwrap();
        assert_eq!(p.len(), 4, "{p:?}");
        let xs: Vec<f32> = p.iter().map(|q| q[0]).collect();
        let ys: Vec<f32> = p.iter().map(|q| q[1]).collect();
        assert_eq!(xs.iter().cloned().fold(f32::MAX, f32::min), 10.0);
        assert_eq!(xs.iter().cloned().fold(f32::MIN, f32::max), 30.0);
        assert_eq!(ys.iter().cloned().fold(f32::MAX, f32::min), 5.0);
        assert_eq!(ys.iter().cloned().fold(f32::MIN, f32::max), 20.0);
    }

    #[test]
    fn keeps_largest_region_ignores_holes_and_pinches() {
        // big disc with a hole, a small blob, and a diagonal-only touching pixel
        let m = grid(100, 100, |x, y| {
            let d = ((x as f32 - 50.0).powi(2) + (y as f32 - 50.0).powi(2)).sqrt();
            (d < 30.0 && d > 8.0) || (x < 5 && y < 5) || (x == 81 && y == 50)
        });
        let p = mask_to_polygon(&m, 100, 100, [2.0, 2.0], 1.0, 64).unwrap();
        assert!(p.len() >= 8 && p.len() <= 64, "{}", p.len());
        for q in &p {
            let d = ((q[0] / 2.0 - 50.0).powi(2) + (q[1] / 2.0 - 50.0).powi(2)).sqrt();
            assert!(d > 25.0 && d < 34.0, "point {q:?} off the outer circle (d={d})");
        }
        // point budget is respected
        let p = mask_to_polygon(&m, 100, 100, [1.0, 1.0], 0.3, 12).unwrap();
        assert!(p.len() <= 12);
    }

    #[test]
    fn larger_tolerance_never_adds_points() {
        // wiggly blob, like a real mask edge
        let m = grid(300, 300, |x, y| {
            let (dx, dy) = (x as f32 - 150.0, y as f32 - 150.0);
            let a = dy.atan2(dx);
            (dx * dx + dy * dy).sqrt() < 100.0 + 12.0 * (7.0 * a).sin() + 4.0 * (31.0 * a).cos()
        });
        let mut last = usize::MAX;
        for eps in [0.5f32, 1.0, 1.5, 2.0, 3.0, 5.0] {
            let n = mask_to_polygon(&m, 300, 300, [1.0, 1.0], eps, 10_000).unwrap().len();
            eprintln!("eps {eps}: {n} points");
            assert!(n <= last.saturating_add(2), "eps {eps}: {n} > {last}");
            last = n;
        }
    }

    #[test]
    fn huge_ragged_mask_does_not_overflow() {
        // checkerboard-ish noise inside a big square: very long boundary
        let m = grid(1024, 1024, |x, y| x > 10 && y > 10 && x < 1000 && y < 1000 && ((x / 2 + y / 2) % 3 != 0 || x % 7 == 0));
        let p = std::thread::Builder::new()
            .stack_size(256 * 1024) // much smaller than the AI thread's stack
            .spawn(move || mask_to_polygon(&m, 1024, 1024, [1.0, 1.0], 0.5, 150))
            .unwrap()
            .join()
            .unwrap();
        assert!(p.is_some_and(|p| p.len() <= 150));
    }

    #[test]
    fn empty_mask() {
        assert!(mask_to_polygon(&vec![false; 100], 10, 10, [1.0, 1.0], 1.0, 50).is_none());
    }

    #[test]
    fn nms_per_class() {
        let b = [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]];
        let keep = nms(&b, &[0.9, 0.8, 0.7, 0.6], &[0, 0, 1, 0], 0.5, 100);
        assert_eq!(keep, vec![0, 2, 3]);
    }
}
