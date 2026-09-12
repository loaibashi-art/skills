#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تجريد اللوحات | painting-abstraction
------------------------------------
تحويل الصور واللوحات إلى فن تجريدي بأساليب متعددة، مع إخراج PNG أو PDF.

المتطلبات: Pillow و numpy فقط (pip install pillow numpy)

مثال:
    python3 abstract.py input.jpg -o output.png --style cubist --intensity 0.7
    python3 abstract.py input.jpg -o output.pdf --style colorfield --seed 7
"""

import argparse
import math
import random
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("خطأ: مكتبة numpy غير مثبتة. ثبّتها عبر: pip install numpy")

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
except ImportError:
    sys.exit("خطأ: مكتبة Pillow غير مثبتة. ثبّتها عبر: pip install pillow")


# ---------------------------------------------------------------------------
# أدوات مساعدة
# ---------------------------------------------------------------------------

def load_image(path, size):
    """تحميل الصورة وتصغيرها بحيث لا يتجاوز أكبر بُعد `size`."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    if scale < 1:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return img


def dominant_colors(img, k=5, seed=0, max_iter=25, sample=20000):
    """استخراج الألوان السائدة بخوارزمية k-means مبسطة (مرتبة حسب الشيوع)."""
    arr = np.asarray(img, dtype=np.float32).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    if len(arr) > sample:
        arr_s = arr[rng.choice(len(arr), sample, replace=False)]
    else:
        arr_s = arr
    centers = arr_s[rng.choice(len(arr_s), min(k, len(arr_s)), replace=False)].copy()
    for _ in range(max_iter):
        dist = ((arr_s[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        new_centers = np.array([
            arr_s[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
            for i in range(len(centers))
        ])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    counts = np.array([(labels == i).sum() for i in range(len(centers))])
    order = np.argsort(-counts)
    return [tuple(int(c) for c in centers[i]) for i in order]


def make_sampler(img, res=256):
    """مُقتطف ألوان سريع: يعيد دالة تأخذ إحداثيات نسبية (0-1) وتعيد لون RGB."""
    small = np.asarray(img.resize((res, res), Image.BILINEAR), dtype=np.uint8)

    def sample(u, v):
        x = min(res - 1, max(0, int(u * res)))
        y = min(res - 1, max(0, int(v * res)))
        return tuple(int(c) for c in small[y, x])

    return sample


def jitter_color(color, rng, amount=0.08):
    """تباين طفيف في سطوع اللون لإضفاء حيوية."""
    f = 1.0 + rng.uniform(-amount, amount)
    return tuple(min(255, max(0, int(c * f))) for c in color)


def luminance(color):
    r, g, b = color
    return 0.299 * r + 0.587 * g + 0.114 * b


def to_hex(color):
    return "#%02x%02x%02x" % color


# ---------------------------------------------------------------------------
# الأساليب التجريدية السبعة
# ---------------------------------------------------------------------------

def style_geometric(src, palette, intensity, rng):
    """تجريد هندسي: فسيفساء لونية + شبكة جريئة + أشكال مُميَّزة."""
    w, h = src.size
    cells_x = max(6, int(10 + (1.0 - intensity) * 34))
    cells_y = max(6, int(cells_x * h / w))
    small = src.resize((cells_x, cells_y), Image.BILINEAR)
    mosaic = small.resize((w, h), Image.NEAREST)
    mosaic = ImageOps.posterize(mosaic, max(2, int(6 - intensity * 4)))
    draw = ImageDraw.Draw(mosaic, "RGBA")

    # شبكة داكنة بمحاذاة خلايا الفسيفساء
    line_w = max(2, w // 300)
    dark = (15, 15, 15, 235)
    for i in range(cells_x + 1):
        x = int(i * w / cells_x)
        draw.line([(x, 0), (x, h)], fill=dark, width=line_w)
    for j in range(cells_y + 1):
        y = int(j * h / cells_y)
        draw.line([(0, y), (w, y)], fill=dark, width=line_w)

    # أشكال مُميَّزة بألوان اللوحة
    n_shapes = 3 + int(intensity * 4)
    for _ in range(n_shapes):
        c = palette[rng.integers(0, len(palette)).item()] + (200,)
        kind = rng.random()
        if kind < 0.4:  # دائرة
            r = rng.uniform(0.04, 0.12) * min(w, h)
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c, outline=dark, width=line_w)
        elif kind < 0.7:  # قوس سميك
            r = rng.uniform(0.08, 0.2) * min(w, h)
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            draw.arc([cx - r, cy - r, cx + r, cy + r],
                     start=rng.uniform(0, 360), end=rng.uniform(0, 360) + rng.uniform(60, 220),
                     fill=c, width=max(3, int(r * 0.25)))
        else:  # خط قطري جريء
            draw.line([(rng.uniform(0, w), rng.uniform(0, h)),
                       (rng.uniform(0, w), rng.uniform(0, h))],
                      fill=c, width=max(4, w // 90))
    return mosaic.convert("RGB")


def _triangle_grid(w, h, n, rng, jitter):
    """شبكة نقاط مُمَوَّهة للتثليث."""
    xs = np.linspace(0, w, n + 1)
    ys = np.linspace(0, h, n + 1)
    pts = {}
    for i in range(n + 1):
        for j in range(n + 1):
            edge = (i in (0, n)) or (j in (0, n))
            jx = 0 if (edge and rng.random() < 0.7) else rng.uniform(-jitter, jitter) * (w / n)
            jy = 0 if (edge and rng.random() < 0.7) else rng.uniform(-jitter, jitter) * (h / n)
            px = min(w, max(0, xs[i] + jx))
            py = min(h, max(0, ys[j] + jy))
            pts[(i, j)] = (px, py)
    return pts


def style_cubist(src, palette, intensity, rng):
    """تجريد تكعيبي: تفتيت اللوحة إلى مثلثات ومستويات متداخلة."""
    w, h = src.size
    sample = make_sampler(src)
    n = max(5, int(7 + (1.0 - intensity) * 13))
    pts = _triangle_grid(w, h, n, rng, jitter=0.45)

    canvas = Image.new("RGB", (w, h), palette[-1])
    draw = ImageDraw.Draw(canvas)
    dark = (18, 18, 18)
    for i in range(n):
        for j in range(n):
            quads = [
                [pts[(i, j)], pts[(i + 1, j)], pts[(i, j + 1)]],
                [pts[(i + 1, j)], pts[(i + 1, j + 1)], pts[(i, j + 1)]],
            ]
            if rng.random() < 0.5:  # تنويع اتجاه القطر أحيانًا
                quads = [
                    [pts[(i, j)], pts[(i + 1, j)], pts[(i + 1, j + 1)]],
                    [pts[(i, j)], pts[(i + 1, j + 1)], pts[(i, j + 1)]],
                ]
            for tri in quads:
                cu = sum(p[0] for p in tri) / 3 / w
                cv = sum(p[1] for p in tri) / 3 / h
                color = jitter_color(sample(cu, cv), rng, 0.12)
                outline = dark if rng.random() < 0.35 + intensity * 0.4 else None
                draw.polygon(tri, fill=color, outline=outline)

    # مستويات شفافة متداخلة بألوان اللوحة
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    for _ in range(4 + int(intensity * 8)):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        rw, rh = rng.uniform(0.1, 0.35) * w, rng.uniform(0.1, 0.35) * h
        ang = rng.uniform(0, math.pi)
        corners = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            lx, ly = sx * rw / 2, sy * rh / 2
            corners.append((cx + lx * math.cos(ang) - ly * math.sin(ang),
                            cy + lx * math.sin(ang) + ly * math.cos(ang)))
        c = palette[rng.integers(0, len(palette)).item()] + (rng.integers(50, 110).item(),)
        odraw.polygon(corners, fill=c, outline=(15, 15, 15, 160))
    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")


def style_lowpoly(src, palette, intensity, rng):
    """تجريد بالمضلّعات: تثليث نظيف بألوان مأخوذة من اللوحة."""
    w, h = src.size
    sample = make_sampler(src)
    n = max(8, int(12 + (1.0 - intensity) * 26))
    pts = _triangle_grid(w, h, n, rng, jitter=0.35)
    avg = tuple(int(c) for c in np.asarray(src, dtype=np.float32).reshape(-1, 3).mean(axis=0))
    canvas = Image.new("RGB", (w, h), avg)
    draw = ImageDraw.Draw(canvas)
    for i in range(n):
        for j in range(n):
            tris = [
                [pts[(i, j)], pts[(i + 1, j)], pts[(i, j + 1)]],
                [pts[(i + 1, j)], pts[(i + 1, j + 1)], pts[(i, j + 1)]],
            ]
            for tri in tris:
                cu = sum(p[0] for p in tri) / 3 / w
                cv = sum(p[1] for p in tri) / 3 / h
                draw.polygon(tri, fill=jitter_color(sample(cu, cv), rng, 0.05))
    return canvas


def style_expressionist(src, palette, intensity, rng):
    """تجريد تعبيري: ضربات فرشاة عريضة جريئة بألوان مُشبَعة."""
    w, h = src.size
    sample = make_sampler(src)
    base = src.resize((w // 4, h // 4), Image.BILINEAR).filter(
        ImageFilter.GaussianBlur(6)).resize((w, h), Image.BILINEAR)
    base = ImageEnhance.Brightness(base).enhance(0.55)
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    n_strokes = int(120 + intensity * 420)
    phase = rng.uniform(0, 2 * math.pi)
    for _ in range(n_strokes):
        u, v = rng.random(), rng.random()
        x, y = u * w, v * h
        # حقل اتجاه متموّج يعطي الضربات إيقاعًا متماسكًا
        ang = phase + 2.2 * math.sin(u * 5.1 + phase) + 1.7 * math.cos(v * 4.3)
        length = rng.uniform(0.02, 0.02 + intensity * 0.09) * max(w, h)
        width = max(2, int(rng.uniform(0.004, 0.004 + intensity * 0.02) * max(w, h)))
        dx, dy = math.cos(ang) * length / 2, math.sin(ang) * length / 2
        color = sample(u, v) + (rng.integers(150, 235).item(),)
        draw.line([(x - dx, y - dy), (x + dx, y + dy)], fill=color, width=width)
        draw.ellipse([x - dx - width / 2, y - dy - width / 2,
                      x - dx + width / 2, y - dy + width / 2], fill=color)
        draw.ellipse([x + dx - width / 2, y + dy - width / 2,
                      x + dx + width / 2, y + dy + width / 2], fill=color)

    # قطرات لونية منسدلة
    for _ in range(4 + int(intensity * 10)):
        x = rng.uniform(0, w)
        y0 = rng.uniform(0, h * 0.5)
        ln = rng.uniform(0.1, 0.45) * h
        c = palette[rng.integers(0, len(palette)).item()] + (200,)
        draw.line([(x, y0), (x, y0 + ln)], fill=c, width=max(2, w // 220))

    out = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
    out = ImageEnhance.Color(out).enhance(1.25 + intensity * 0.25)
    out = ImageEnhance.Contrast(out).enhance(1.1)
    return out


def style_colorfield(src, palette, intensity, rng):
    """حقول اللون: نطاقات لونية هادئة ضبابية الحواف من ألوان اللوحة."""
    w, h = src.size
    ordered = sorted(palette, key=luminance)
    dark, mid = ordered[0], ordered[len(ordered) // 2]

    # خلفية متدرجة من الألوان الداكنة
    top = np.array(dark, dtype=np.float32)
    bottom = np.array(mid, dtype=np.float32)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    bg = (top[None, None, :] * (1 - t) + bottom[None, None, :] * t)
    bg = np.repeat(bg, w, axis=1).astype(np.uint8)
    canvas = Image.fromarray(bg)

    # نطاقات أفقية ضبابية
    n_bands = 2 + int(intensity * 2.4)
    edges = sorted(rng.uniform(0.06, 0.94, n_bands + 1))
    for i in range(n_bands):
        y0, y1 = int(edges[i] * h), int(edges[i + 1] * h)
        if y1 - y0 < h * 0.04:
            continue
        band = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        bdraw = ImageDraw.Draw(band)
        c = ordered[(i + 1) % len(ordered)] + (rng.integers(150, 220).item(),)
        pad = int(w * rng.uniform(0.04, 0.14))
        bdraw.rounded_rectangle([pad, y0, w - pad, y1],
                                radius=int((y1 - y0) * 0.25), fill=c)
        band = band.filter(ImageFilter.GaussianBlur(max(4, int(h / 60))))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), band)

    canvas = canvas.convert("RGB")
    # حبيبات قماشية خفيفة
    grain = rng.normal(0, 5 + intensity * 5, (h, w, 1))
    arr = np.asarray(canvas, dtype=np.float32) + grain
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def style_flow(src, palette, intensity, rng):
    """تجريد انسيابي: تموّجات ومنحنيات غنائية تذيب معالم اللوحة."""
    w, h = src.size
    arr = np.asarray(src)
    strips = max(8, int(18 + intensity * 60))
    phase = rng.uniform(0, 2 * math.pi)
    freq = rng.uniform(2.0, 4.5)
    amp = w * 0.035 * (0.5 + intensity)
    out = np.empty_like(arr)
    bounds = np.linspace(0, h, strips + 1, dtype=int)
    for i in range(strips):
        y0, y1 = bounds[i], bounds[i + 1]
        offset = int(amp * math.sin(phase + i / strips * freq * math.pi))
        out[y0:y1] = np.roll(arr[y0:y1], offset, axis=1)
    img = Image.fromarray(out).filter(ImageFilter.GaussianBlur(1.2 + intensity * 3.0))

    # منحنيات متدفقة بألوان اللوحة
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    n_curves = 4 + int(intensity * 8)
    for k in range(n_curves):
        c = palette[k % len(palette)] + (rng.integers(60, 120).item(),)
        width = max(3, int(h * rng.uniform(0.008, 0.03)))
        ph = rng.uniform(0, 2 * math.pi)
        fr = rng.uniform(1.5, 3.5)
        yc = rng.uniform(0.15, 0.85) * h
        amp_c = rng.uniform(0.03, 0.12) * h
        pts = [(x, yc + amp_c * math.sin(ph + x / w * fr * math.pi))
               for x in range(0, w + 1, max(2, w // 200))]
        odraw.line(pts, fill=c, width=width, joint="curve")
    out_img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return ImageEnhance.Color(out_img).enhance(1.15 + intensity * 0.2)


def sobel_edges(img, threshold=0.12):
    """قناع حواف (0-1) بخوارزمية Sobel عبر numpy فقط — لإبراز زخارف اللوحة."""
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    p = np.pad(g, 1, mode="edge")
    gx = (-p[:-2, :-2] - 2 * p[1:-1, :-2] - p[2:, :-2]
          + p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:])
    gy = (-p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:]
          + p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
    mag = np.sqrt(gx * gx + gy * gy)
    mag /= (mag.max() + 1e-6)
    return np.clip((mag - threshold) / (1.0 - threshold), 0.0, 1.0)


def style_faceted(src, palette, intensity, rng):
    """تجريد مُسطَّحي: مثلثات كبيرة مع الحفاظ على المعالم وإبراز الزخارف."""
    w, h = src.size
    sample = make_sampler(src)
    # مثلثات كبيرة: الشدة الأعلى تعني مثلثات أكثر وأصغر
    n = max(4, int(5 + intensity * 12))
    pts = _triangle_grid(w, h, n, rng, jitter=0.30)
    avg = tuple(int(c) for c in np.asarray(src, dtype=np.float32).reshape(-1, 3).mean(axis=0))
    canvas = Image.new("RGB", (w, h), avg)
    draw = ImageDraw.Draw(canvas)
    edge = (25, 25, 25)
    for i in range(n):
        for j in range(n):
            # تنويع اتجاه القطر عشوائيًا لكل خلية يعطي إيقاعًا عضويًا
            if rng.random() < 0.5:
                tris = [
                    [pts[(i, j)], pts[(i + 1, j)], pts[(i, j + 1)]],
                    [pts[(i + 1, j)], pts[(i + 1, j + 1)], pts[(i, j + 1)]],
                ]
            else:
                tris = [
                    [pts[(i, j)], pts[(i + 1, j)], pts[(i + 1, j + 1)]],
                    [pts[(i, j)], pts[(i + 1, j + 1)], pts[(i, j + 1)]],
                ]
            for tri in tris:
                cu = sum(p[0] for p in tri) / 3 / w
                cv = sum(p[1] for p in tri) / 3 / h
                # حدود رفيعة لكل مثلث تعطي القراءة البلورية للسطوح
                draw.polygon(tri, fill=jitter_color(sample(cu, cv), rng, 0.04), outline=edge)
    # مزج مع اللوحة الأصلية للحفاظ على المعالم والزخارف الدقيقة
    keep = 0.55 - intensity * 0.30
    blended = Image.blend(canvas, src, keep)
    # إبراز حواف الزخارف الأصلية بخطوط داكنة رفيعة
    mask = sobel_edges(src, threshold=0.15)
    strength = 0.35 + intensity * 0.15
    arr = np.asarray(blended, dtype=np.float32)
    arr *= (1.0 - mask[..., None] * strength)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return ImageEnhance.Color(out).enhance(1.08)


STYLES = {
    "geometric": (style_geometric, "تجريد هندسي — فسيفساء لونية وشبكة جريئة"),
    "cubist": (style_cubist, "تجريد تكعيبي — تفتيت إلى مثلثات ومستويات"),
    "lowpoly": (style_lowpoly, "مضلّعات — تثليث نظيف بألوان اللوحة"),
    "faceted": (style_faceted, "تجريد مُسطَّحي — مثلثات كبيرة مع زخارف واضحة"),
    "expressionist": (style_expressionist, "تجريد تعبيري — ضربات فرشاة جريئة"),
    "colorfield": (style_colorfield, "حقول اللون — نطاقات هادئة ضبابية"),
    "flow": (style_flow, "تجريد انسيابي — تموّجات ومنحنيات غنائية"),
}


# ---------------------------------------------------------------------------
# الواجهة
# ---------------------------------------------------------------------------

def save_output(img, path):
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        img.convert("RGB").save(path, "PDF", resolution=150.0)
    else:
        img.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="تجريد اللوحات: تحويل الصور إلى فن تجريدي (PNG أو PDF)."
    )
    parser.add_argument("input", nargs="?", help="مسار الصورة المدخلة")
    parser.add_argument("-o", "--output", required=False, help="مسار الصورة الناتجة (png أو pdf)")
    parser.add_argument("--style", choices=sorted(STYLES), default="cubist",
                        help="الأسلوب التجريدي (الافتراضي: cubist)")
    parser.add_argument("--intensity", type=float, default=0.65,
                        help="شدة التجريد من 0 إلى 1 (الافتراضي: 0.65)")
    parser.add_argument("--seed", type=int, default=42, help="بذرة العشوائية (الافتراضي: 42)")
    parser.add_argument("--size", type=int, default=1200,
                        help="أكبر بُعد للصورة الناتجة بالبكسل (الافتراضي: 1200)")
    parser.add_argument("--palette-size", type=int, default=5,
                        help="عدد الألوان المستخرجة من اللوحة (الافتراضي: 5)")
    parser.add_argument("--list-styles", action="store_true", help="عرض الأساليب المتاحة")
    args = parser.parse_args(argv)

    if args.list_styles:
        for key in sorted(STYLES):
            print(f"{key}: {STYLES[key][1]}")
        return 0

    if not args.input or not args.output:
        parser.error("يجب تحديد الصورة المدخلة ومسار الإخراج (-o).")

    if not 0.0 <= args.intensity <= 1.0:
        parser.error("شدة التجريد يجب أن تكون بين 0 و 1.")

    in_path = Path(args.input)
    if not in_path.is_file():
        sys.exit(f"خطأ: ملف الإدخال غير موجود: {args.input}")

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    src = load_image(str(in_path), args.size)
    palette = dominant_colors(src, k=max(3, args.palette_size), seed=args.seed)
    func, label = STYLES[args.style]
    result = func(src, palette, args.intensity, rng)
    save_output(result, args.output)

    print(f"تم: {args.output}")
    print(f"الأسلوب: {args.style} ({label})")
    print("لوحة الألوان: " + " ".join(to_hex(c) for c in palette))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
