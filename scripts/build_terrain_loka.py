#!/usr/bin/env python3
"""
Build Loka Terrain (baseHeight, watermask, relief) from landmass SVG.

Input:
  - canon-core/canon/world/loka/loka.json
  - ../canon-assets/world/loka/atlas/Loka_Landmass_canonized.svg

Output (written into canon-assets submodule):
  - ../canon-assets/world/loka/terrain/Loka_Terrain_baseHeight.png   (4096x2048, 16-bit)
  - ../canon-assets/world/loka/terrain/Loka_Terrain_watermask.png   (8-bit 0/255)
  - ../canon-assets/world/loka/terrain/Loka_Terrain_relief.png      (hillshade preview)
"""

import json, os, sys, io, math, pathlib
import numpy as np
from PIL import Image, ImageFilter
import cairosvg
try:
    from noise import pnoise2
except Exception:
    pnoise2 = None

ROOT = pathlib.Path(__file__).resolve().parents[2]  # canon-core root
LOKA_JSON = ROOT / "canon/world/loka/loka.json"

def load_config():
    data = json.loads(LOKA_JSON.read_text(encoding="utf-8"))
    res = data["terrain"].get("resolution", "4096x2048")
    w, h = map(int, res.split("x"))
    sea = int(data["terrain"].get("sea_level", 32768))
    landmass_svg_rel = data["atlas"]["landmass_svg"]  # ../canon-assets/...
    landmass_svg = (ROOT / "canon/world/loka" / "atlas" / "Loka_Landmass_canonized.svg")
    # หากไฟล์ atlas ใน core ไม่มี ให้ใช้ path ที่ชี้ไป submodule
    if not landmass_svg.exists():
        landmass_svg = (ROOT / landmass_svg_rel).resolve()
    out_dir = (ROOT / "../canon-assets/world/loka/terrain").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return dict(W=w, H=h, SEA=sea, SVG=landmass_svg, OUT=out_dir)

def svg_to_mask(svg_path, W, H):
    """Rasterize SVG -> RGBA then threshold (land=1, ocean=0). Expect land as dark/black."""
    png_bytes = cairosvg.svg2png(url=str(svg_path), output_width=W, output_height=H, background_color="white")
    img = Image.open(io.BytesIO(png_bytes)).convert("L")
    # สมมติ land เป็นสีดำ (ค่าต่ำ) และทะเลเป็นขาว → กลับค่าให้ land=1
    arr = np.array(img, dtype=np.float32)
    mask = (arr < 200).astype(np.uint8)  # ปรับ threshold ได้ตาม art
    # ทำความสะอาดขอบเล็กน้อย
    from scipy.ndimage import binary_opening, binary_closing
    try:
        mask = binary_closing(binary_opening(mask, iterations=1), iterations=1).astype(np.uint8)
    except Exception:
        pass
    return mask  # 0/1

def gen_noise_base(W, H, octaves=(1, 2, 4, 8), base_freq=2.0):
    """Perlin multi-octave (ต้องมี noise library), ถ้าไม่มีใช้ random smooth แทน"""
    if pnoise2 is None:
        # สำรอง: random + blur
        rnd = np.random.RandomState(42).rand(H, W).astype(np.float32)
        img = Image.fromarray((rnd*255).astype(np.uint8))
        for _ in range(5):
            img = img.filter(ImageFilter.GaussianBlur(radius=4))
        return np.array(img, dtype=np.float32)/255.0

    scale = 1.0 / base_freq
    out = np.zeros((H, W), dtype=np.float32)
    amp_total = 0.0
    for i, octv in enumerate(octaves):
        freq = base_freq * (2**i)
        amp  = 1.0 / (2**i)
        for y in range(H):
            for x in range(W):
                out[y, x] += amp * pnoise2(x/(W/scale)/freq, y/(H/scale)/freq, repeatx=W, repeaty=H)
        amp_total += amp
    out = (out - out.min())/(out.max()-out.min()+1e-6)
    return out

def shape_mountains(mask, base):
    """เพิ่มสันเขาใกล้ขอบทวีปและกลางทวีปแบบหยาบ"""
    H, W = mask.shape
    # distance from ocean (approx)
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(mask)
    dist = dist / (dist.max()+1e-6)
    ridge = np.clip(dist, 0, 1)
    elev = 0.35*base + 0.65*ridge
    elev *= mask  # ocean -> 0
    # smooth
    img = Image.fromarray((elev*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
    elev = np.array(img, dtype=np.float32)/255.0
    return elev

def to_uint16(elev, sea_level=32768):
    """แปลง 0..1 ให้กลายเป็น 16-bit และแทรกเส้นระดับน้ำทะเล"""
    arr = (elev * 65535.0).astype(np.uint16)
    # ocean = 0.95*sea_level เพื่อหลีกเลี่ยงรอยขอบเวลากะพริบ
    arr[elev <= 1e-6] = max(0, sea_level-1024)
    return arr

def hillshade(elev, azimuth=315.0, altitude=45.0, z=1.0):
    """ทำ hillshade ง่ายๆ เพื่อพรีวิว relief"""
    H, W = elev.shape
    # gradient
    gy, gx = np.gradient(elev)
    slope = np.pi/2.0 - np.arctan(z * np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.deg2rad(azimuth)
    alt = np.deg2rad(altitude)
    shade = np.sin(alt)*np.sin(slope) + np.cos(alt)*np.cos(slope)*np.cos(az - aspect)
    shade = (shade - shade.min())/(shade.max()-shade.min()+1e-6)
    return (shade*255).astype(np.uint8)

def main():
    cfg = load_config()
    W, H, SEA = cfg["W"], cfg["H"], cfg["SEA"]
    svg = cfg["SVG"]
    out = cfg["OUT"]
    print(f"[i] Resolution {W}x{H} sea={SEA} svg={svg}")
    if not svg.exists():
        print(f"[!] Landmass SVG not found: {svg}")
        sys.exit(1)

    # 1) SVG -> mask (0/1)
    mask = svg_to_mask(svg, W, H)

    # 2) base noise + mountain shaping
    base = gen_noise_base(W, H, octaves=(1,2,4,8), base_freq=2.0)
    elev = shape_mountains(mask, base)

    # 3) export baseHeight (16-bit)
    h16 = to_uint16(elev, sea_level=SEA)
    Image.fromarray(h16, mode="I;16").save(out / "Loka_Terrain_baseHeight.png", compress_level=0)

    # 4) export watermask (0/255)
    water = (mask*255).astype(np.uint8)   # land=255, ocean=0
    Image.fromarray(water, mode="L").save(out / "Loka_Terrain_watermask.png")

    # 5) relief preview (8-bit)
    rel = hillshade(elev)
    Image.fromarray(rel, mode="L").save(out / "Loka_Terrain_relief.png")

    print(f"[✓] Wrote: {out}")

if __name__ == "__main__":
    main()