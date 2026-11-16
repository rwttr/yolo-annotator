//! Procedurally drawn program icon: a bounding box (amber) around a segmentation
//! polygon (green) on a blue rounded tile. Used for the window icon at runtime and
//! by `examples/gen_icon.rs` to produce `assets/icon.ico` for the .exe.

type C = [f32; 4]; // straight RGBA, 0..1

fn hex(rgb: u32, a: f32) -> C {
    [
        ((rgb >> 16) & 0xff) as f32 / 255.0,
        ((rgb >> 8) & 0xff) as f32 / 255.0,
        (rgb & 0xff) as f32 / 255.0,
        a,
    ]
}

fn over(dst: C, src: C) -> C {
    let a = src[3] + dst[3] * (1.0 - src[3]);
    if a <= 0.0 {
        return [0.0; 4];
    }
    let mut o = [0.0; 4];
    for i in 0..3 {
        o[i] = (src[i] * src[3] + dst[i] * dst[3] * (1.0 - src[3])) / a;
    }
    o[3] = a;
    o
}

fn seg_dist(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
    let (x, y) = (a[0] + t * dx - p[0], a[1] + t * dy - p[1]);
    (x * x + y * y).sqrt()
}

fn inside(p: [f32; 2], poly: &[[f32; 2]]) -> bool {
    let mut c = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[j]);
        if (a[1] > p[1]) != (b[1] > p[1])
            && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]
        {
            c = !c;
        }
        j = i;
    }
    c
}

fn in_round_rect(p: [f32; 2], lo: f32, hi: f32, r: f32) -> bool {
    if p[0] < lo || p[0] > hi || p[1] < lo || p[1] > hi {
        return false;
    }
    let cx = p[0].clamp(lo + r, hi - r);
    let cy = p[1].clamp(lo + r, hi - r);
    (p[0] - cx).powi(2) + (p[1] - cy).powi(2) <= r * r
}

const POLY: [[f32; 2]; 6] = [
    [0.31, 0.60],
    [0.40, 0.37],
    [0.60, 0.31],
    [0.72, 0.49],
    [0.64, 0.71],
    [0.41, 0.73],
];
const BOX: [f32; 4] = [0.21, 0.22, 0.81, 0.82];

fn sample(p: [f32; 2], size: u32) -> C {
    let px = 1.0 / size as f32;
    if !in_round_rect(p, 0.03, 0.97, 0.2) {
        return [0.0; 4];
    }
    // background: vertical blue gradient
    let t = p[1];
    let (top, bot) = (hex(0x2563EB, 1.0), hex(0x1E3A8A, 1.0));
    let mut c = [
        top[0] + (bot[0] - top[0]) * t,
        top[1] + (bot[1] - top[1]) * t,
        top[2] + (bot[2] - top[2]) * t,
        1.0,
    ];
    let w = (0.036_f32).max(1.25 * px); // stroke width, never thinner than ~1px
    // segmentation polygon
    if inside(p, &POLY) {
        c = over(c, hex(0x22C55E, 0.55));
    }
    let edge = (0..POLY.len())
        .map(|i| seg_dist(p, POLY[i], POLY[(i + 1) % POLY.len()]))
        .fold(f32::MAX, f32::min);
    if edge <= w * 0.5 {
        c = over(c, hex(0x4ADE80, 1.0));
    }
    // bounding box
    let [x0, y0, x1, y1] = BOX;
    let corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    let bd = (0..4)
        .map(|i| seg_dist(p, corners[i], corners[(i + 1) % 4]))
        .fold(f32::MAX, f32::min);
    if bd <= w * 0.55 {
        c = over(c, hex(0xFBBF24, 1.0));
    }
    // handles: box corners (squares) and polygon vertices (dots); dropped at tiny sizes
    if size >= 32 {
        let h = w * 1.35;
        if corners.iter().any(|q| (p[0] - q[0]).abs() <= h && (p[1] - q[1]).abs() <= h) {
            c = over(c, hex(0xFFFFFF, 1.0));
            if corners
                .iter()
                .any(|q| (p[0] - q[0]).abs() <= h * 0.55 && (p[1] - q[1]).abs() <= h * 0.55)
            {
                c = over(c, hex(0xF59E0B, 1.0));
            }
        }
        let r = w * 1.05;
        if POLY.iter().any(|q| (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) <= r * r) {
            c = over(c, hex(0xFFFFFF, 1.0));
        }
    }
    c
}

/// Renders the icon as straight-alpha RGBA8, `size` x `size`, 4x4 supersampled.
pub fn rgba(size: u32) -> Vec<u8> {
    const SS: u32 = 4;
    let mut out = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let mut acc = [0.0f32; 4];
            for sy in 0..SS {
                for sx in 0..SS {
                    let p = [
                        (x as f32 + (sx as f32 + 0.5) / SS as f32) / size as f32,
                        (y as f32 + (sy as f32 + 0.5) / SS as f32) / size as f32,
                    ];
                    let c = sample(p, size);
                    // accumulate premultiplied
                    for i in 0..3 {
                        acc[i] += c[i] * c[3];
                    }
                    acc[3] += c[3];
                }
            }
            let n = (SS * SS) as f32;
            let a = acc[3] / n;
            for i in 0..3 {
                let v = if a > 0.0 { acc[i] / n / a } else { 0.0 };
                out.push((v * 255.0).round().clamp(0.0, 255.0) as u8);
            }
            out.push((a * 255.0).round().clamp(0.0, 255.0) as u8);
        }
    }
    out
}
