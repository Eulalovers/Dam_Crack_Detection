"""巡检推理：大图切片分割 → 后处理 → 裂缝量化（mm）→ 分级 → 标注图 + Excel 报告。

示例：
  # 已知相机参数和物距（贴近摄影测量航线设计值或激光测距值）
  python infer.py --input G:/photos --weights G:/Dam_crack_data/runs/ozgenel/unet_sp_dsc_combo_s0/best.pt \
      --model unet_sp_dsc --pixel_um 4.4 --focal_mm 35 --distance_m 1.0
  # 或直接给定 GSD（例如从正射影像 / 三维模型标定得到）
  python infer.py --input img.jpg --weights ... --gsd 0.12
"""
import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from crackseg.config import OUTPUT_DIR
from crackseg.datasets import read_image
from crackseg.models import build_model
from crackseg.quantify import gsd_mm, postprocess, quantify_cracks
from crackseg.tiling import predict_tiled

GRADE_COLORS = {"A": (0, 200, 0), "B": (255, 200, 0), "C": (255, 120, 0), "D": (230, 0, 0), "-": (255, 0, 255)}
GRADE_DESC = {"A": "A类 (<0.2mm)", "B": "B类 (0.2~0.3mm)", "C": "C类 (0.3~0.4mm)", "D": "D类 (≥0.4mm)",
              "-": "未分级（缺少GSD）"}


def flag_structural_joint(c, img_shape, straight_tol=0.97):
    """施工缝/伸缩缝/模板缝通常是贯穿整幅、近乎笔直、且沿水平或竖直方向的长线。
    满足这些条件的目标仅标记为"疑似结构缝，需人工复核"，不直接删除（避免漏报真实裂缝）。"""
    x, y, w, h = c.bbox_xywh
    chord = np.hypot(w, h)
    straight = chord / max(c.length_px, 1) > straight_tol
    long_ = c.length_px > 0.6 * max(img_shape)
    axis = min(c.orientation_deg, 90 - c.orientation_deg) < 5
    return bool(straight and long_ and axis)


def draw_overlay(img, mask, cracks):
    vis = img.copy()
    color_mask = np.zeros_like(img)
    lab = cv2.connectedComponentsWithStats(mask, connectivity=8)[1]
    for c in cracks:
        x, y, w, h = c.bbox_xywh
        cx, cy = int(c.centroid_xy[0]), int(c.centroid_xy[1])
        k = lab[cy, cx] if lab[cy, cx] > 0 else np.bincount(lab[y:y + h, x:x + w].ravel())[1:].argmax() + 1
        color_mask[lab == k] = GRADE_COLORS[c.grade]
    sel = color_mask.any(-1)
    vis[sel] = (0.35 * vis[sel] + 0.65 * color_mask[sel]).astype(np.uint8)
    s = max(img.shape[:2]) / 700
    for c in cracks:
        x, y, w, h = c.bbox_xywh
        cv2.rectangle(vis, (x, y), (x + w, y + h), GRADE_COLORS[c.grade], max(1, int(2 * s)))
        label = f"#{c.id} {c.grade}" + (f" {c.p95_width_mm:.2f}mm" if not np.isnan(c.p95_width_mm) else "")
        ty = y - int(8 * s) if y > 30 * s else y + int(30 * s)   # 靠近顶部时画在框内
        for col, th in [((255, 255, 255), max(2, int(3 * s))), (GRADE_COLORS[c.grade], max(1, int(1.5 * s)))]:
            cv2.putText(vis, label, (x + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, col, th, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="图片或文件夹")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", default="unet_dsc")
    ap.add_argument("--out", default=str(OUTPUT_DIR / "inspection"))
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--gsd", type=float, default=None, help="mm/pixel，直接指定")
    ap.add_argument("--pixel_um", type=float, default=None, help="像元尺寸 μm")
    ap.add_argument("--focal_mm", type=float, default=None)
    ap.add_argument("--distance_m", type=float, default=None, help="相机到坝面距离 m")
    a = ap.parse_args()

    gsd = a.gsd
    if gsd is None and None not in (a.pixel_um, a.focal_mm, a.distance_m):
        gsd = gsd_mm(a.pixel_um, a.distance_m, a.focal_mm)
    print(f"GSD = {gsd:.4f} mm/px" if gsd else "未提供 GSD，仅输出像素单位结果")

    inp = Path(a.input)
    files = sorted(p for p in (inp.iterdir() if inp.is_dir() else [inp])
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = build_model(a.model, pretrained=False).cuda().eval()
    model.load_state_dict(torch.load(a.weights, map_location="cuda"))

    rows, summary = [], []
    for f in files:
        img = read_image(f)
        prob = predict_tiled(model, img, tile=a.tile, overlap=a.tile // 4)
        mask = postprocess(prob >= a.thr, min_area_px=64)
        cracks = quantify_cracks(mask, gsd=gsd)
        cv2.imwrite(str(out / f"{f.stem}_mask.png"), mask * 255)
        cv2.imwrite(str(out / f"{f.stem}_overlay.jpg"), cv2.cvtColor(draw_overlay(img, mask, cracks), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        for c in cracks:
            d = c.to_dict()
            rows.append({
                "影像": f.name, "裂缝编号": c.id,
                "长度(px)": round(c.length_px, 1), "最大宽度(px)": round(c.max_width_px, 2),
                "平均宽度(px)": round(c.mean_width_px, 2), "P95宽度(px)": round(c.p95_width_px, 2),
                "长度(mm)": round(d["length_mm"], 1), "最大宽度(mm)": round(d["max_width_mm"], 3),
                "平均宽度(mm)": round(d["mean_width_mm"], 3), "P95宽度(mm)": round(d["p95_width_mm"], 3),
                "走向(°,0=水平)": c.orientation_deg, "初判等级": GRADE_DESC[c.grade],
                "疑似结构缝(需复核)": "是" if flag_structural_joint(c, img.shape[:2]) else "",
                "外接框(x,y,w,h)": str(c.bbox_xywh),
            })
        grades = [c.grade for c in cracks]
        summary.append({"影像": f.name, "尺寸": f"{img.shape[1]}x{img.shape[0]}", "GSD(mm/px)": gsd,
                        "裂缝条数": len(cracks),
                        "总长度(mm)": round(sum(c.length_mm for c in cracks), 1) if gsd else None,
                        **{GRADE_DESC[g]: grades.count(g) for g in "ABCD"},
                        "最大P95宽度(mm)": round(max((c.p95_width_mm for c in cracks), default=0), 3) if gsd else None})
        print(f"{f.name}: {len(cracks)} 条裂缝")

    report = out / "巡检报告.xlsx"
    with pd.ExcelWriter(report) as w:
        pd.DataFrame(summary).to_excel(w, sheet_name="汇总", index=False)
        pd.DataFrame(rows).to_excel(w, sheet_name="裂缝明细", index=False)
        pd.DataFrame({"说明": [
            "宽度测量：骨架结构张量求切线方向 + 法线方向亚像素正交剖面；分级采用 P95 宽度（比最大值抗噪）。",
            "等级为仅依据表面缝宽的初判（A<0.2, B 0.2~0.3, C 0.3~0.4, D≥0.4 mm），缝深需现场检测（如超声）确认。",
            "“疑似结构缝”为贯穿、笔直、沿水平/竖直方向的长线目标，可能是施工缝/伸缩缝，需人工复核。",
            "当裂缝宽度小于约 2 个像素时宽度测量不可靠，应减小物距或换长焦镜头以获得更小 GSD。",
        ]}).to_excel(w, sheet_name="说明", index=False)
    print(f"报告已保存: {report}")


if __name__ == "__main__":
    main()
