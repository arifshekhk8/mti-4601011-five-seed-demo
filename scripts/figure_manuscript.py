"""Wide, three-panel drop-in replacement for the manuscript's fig6_demo.png."""
import json
import zipfile

from figure import blur_faces, load_frame  # also puts caches inside private/
from common import ROOT, THRESHOLD

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from ultralytics import YOLO


def main():
    summary = json.loads((ROOT / "results/analysis.json").read_text())
    panels = summary["panels"]
    assert len(panels) == 3 and all(p["matched_track_id"] is not None for p in panels)
    device = "mps" if __import__("torch").backends.mps.is_available() else "cpu"
    detector = YOLO(str(ROOT / "private/checkpoints/yolo26m.pt"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "font.size": 11})
    fig = plt.figure(figsize=(14, 3.34), facecolor="white")
    gs = fig.add_gridspec(3, 3, left=0.012, right=0.988, top=0.985, bottom=0.028,
                          height_ratios=[0.17, 0.67, 0.16], hspace=0.02, wspace=0.018)
    for col, panel in enumerate(panels):
        header = fig.add_subplot(gs[0, col]); header.axis("off")
        header.text(0.01, 0.57, f"{panel['scene']}   PIE {panel['video']}  |  frame {panel['frame']}",
                    transform=header.transAxes, va="center", ha="left", fontsize=12.5,
                    fontweight="bold", color="#152933")

        frame = blur_faces(load_frame(panel["video"], panel["frame"]), panel, detector, device)
        target = panel["target_gt_box"]
        center = (target[0] + target[2]) / 2
        x0 = max(0, min(1920 - 800, int(center - 400)))
        x1 = x0 + 800
        y0, y1 = 580, 980
        cropped = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(cropped, extent=(x0, x1, y1, y0))
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.7); spine.set_color("#CCD4D7")
        for item in panel["panel_boxes"]:
            bx = item["box"]
            if bx[2] <= x0 or bx[0] >= x1 or bx[3] <= y0 or bx[1] >= y1: continue
            focus = item["track_id"] == panel["matched_track_id"]
            color = "#D35431" if item["flagged"] else "#127B91"
            if focus:
                ax.add_patch(Rectangle((bx[0]-2, bx[1]-2), bx[2]-bx[0]+4, bx[3]-bx[1]+4,
                                       fill=False, linewidth=4.5, edgecolor="white", zorder=4))
            ax.add_patch(Rectangle((bx[0], bx[1]), bx[2]-bx[0], bx[3]-bx[1],
                                   fill=False, linewidth=2.5 if focus else 1.25,
                                   edgecolor=color, zorder=5))

        footer = fig.add_subplot(gs[2, col]); footer.axis("off")
        color = "#D35431" if panel["flagged"] else "#127B91"
        footer.text(0.01, 0.79,
                    f"p = {panel['prob_cross']:.4f}   |   ego {panel['ego_speed_kmh']:.1f} km/h",
                    transform=footer.transAxes, va="top", ha="left", fontsize=13,
                    fontweight="bold", color=color)
        if panel["ground_truth_code"] == 1:
            detail = f"GT crossing  |  {panel['frames_to_first_crossing_state']} frames before onset"
        else:
            detail = "GT non-crossing  |  false alert" if panel["flagged"] else "GT non-crossing  |  not flagged"
        footer.text(0.01, 0.21, detail, transform=footer.transAxes,
                    va="top", ha="left", fontsize=10.5, color="#263D48")
    png = ROOT / "figures/fig6_demo.png"
    pdf = ROOT / "figures/fig6_demo.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, dpi=300, facecolor="white")
    plt.close(fig)
    bundle = ROOT / "figures/fig6_demo_dropin.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo("figures/fig6_demo.png", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, png.read_bytes())
    print(png.name, pdf.name, bundle.name, f"threshold={THRESHOLD:.10f}")


if __name__ == "__main__":
    main()
