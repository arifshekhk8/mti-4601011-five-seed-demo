"""Render the three measured, face-blurred PIE examples as journal PDF and PNG."""
import argparse
import json
import os

from pathlib import Path

from common import CONFIG, ROOT, THRESHOLD

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "private/cache/ultralytics"))
os.environ.setdefault("XDG_CONFIG_HOME", str(ROOT / "private/cache/config"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "private/cache/matplotlib"))
for key in ("YOLO_CONFIG_DIR", "XDG_CONFIG_HOME", "MPLCONFIGDIR"):
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_frame(video, frame):
    cap = cv2.VideoCapture(str(ROOT / "private/videos" / f"{video}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = cap.read()
    cap.release()
    if not ok: raise RuntimeError(f"Cannot read {video} frame {frame}")
    return image


def blur_head(image, box):
    x1, y1, x2, y2 = box
    height = max(0, y2 - y1)
    left = max(0, int(x1 - 0.05 * (x2 - x1)))
    right = min(image.shape[1], int(x2 + 0.05 * (x2 - x1)))
    top = max(0, int(y1 - 0.03 * height))
    bottom = min(image.shape[0], int(y1 + 0.32 * height))
    if right - left < 4 or bottom - top < 4: return
    patch = image[top:bottom, left:right]
    # Large blur kernel and second pass obscure facial detail in the public figure.
    k = max(31, min(91, int(min(patch.shape[:2]) * 0.85) | 1))
    image[top:bottom, left:right] = cv2.GaussianBlur(cv2.GaussianBlur(patch, (k, k), 0), (k, k), 0)


def blur_faces(image, panel, detector, device):
    # PIE annotations cover the ground-truth pedestrians; the extra low-confidence
    # detector pass also catches bystanders that annotation may omit.
    for item in panel["all_annotated_boxes"]:
        blur_head(image, item["box"])
    result = detector.predict(image.copy(), classes=[0], conf=0.05, imgsz=640, device=device, verbose=False)[0]
    for box in result.boxes.xyxy.cpu().numpy():
        blur_head(image, box)
    for item in panel["panel_boxes"]:
        blur_head(image, item["box"])
    return image


def main():
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("auto", "mps", "cpu", "cuda:0"), default="auto")
    args = ap.parse_args()
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    summary = json.loads((ROOT / "results/analysis.json").read_text())
    panels = summary["panels"]
    if any(p["matched_track_id"] is None for p in panels):
        raise ValueError("A selected target lacks a tracked prediction")
    detector = YOLO(str(ROOT / "private/checkpoints/yolo26m.pt"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(8.2, 9.5), facecolor="white")
    grid = fig.add_gridspec(3, 2, left=0.045, right=0.965, top=0.905, bottom=0.085,
                            width_ratios=[2.2, 1.0], hspace=0.22, wspace=0.09)
    headings = {"A": "Crossing: advance prediction", "B": "Non-crossing: stationary", "C": "Non-crossing: moving"}
    for i, panel in enumerate(panels):
        image = load_frame(panel["video"], panel["frame"])
        image = blur_faces(image, panel, detector, device)
        target = panel["target_gt_box"]
        center = (target[0] + target[2]) / 2
        x0 = max(0, min(1920 - 700, int(center - 350)))
        x1 = x0 + 700
        y0, y1 = 560, 1000
        crop = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        ax = fig.add_subplot(grid[i, 0])
        ax.imshow(crop, extent=(x0, x1, y1, y0))
        ax.set_xlim(x0, x1);ax.set_ylim(y1, y0);ax.set_aspect("equal")
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        for item in panel["panel_boxes"]:
            bx = item["box"]
            if bx[2] <= x0 or bx[0] >= x1 or bx[3] <= y0 or bx[1] >= y1: continue
            focus = item["track_id"] == panel["matched_track_id"]
            color = "#C65331" if item["flagged"] else "#187B91"
            if focus:
                ax.add_patch(Rectangle((bx[0]-2, bx[1]-2), bx[2]-bx[0]+4, bx[3]-bx[1]+4,
                                       fill=False, linewidth=4.2, edgecolor="white", zorder=4))
            ax.add_patch(Rectangle((bx[0], bx[1]), bx[2]-bx[0], bx[3]-bx[1],
                                   fill=False, linewidth=2.5 if focus else 1.25,
                                   edgecolor=color, zorder=5))
        ax.text(x0 + 10, y0 + 10, panel["scene"], ha="left", va="top",
                fontsize=11, fontweight="bold", color="#122331",
                bbox={"facecolor":"white", "edgecolor":"none", "pad":3, "alpha":0.92})

        card = fig.add_subplot(grid[i, 1]);card.axis("off")
        card.add_patch(Rectangle((0, 0), 1, 1, transform=card.transAxes,
                                 facecolor="#F5F7F7", edgecolor="#D6DEE1", linewidth=0.8))
        card.text(0.07, 0.92, headings[panel["scene"]], transform=card.transAxes,
                  fontsize=9.3, fontweight="bold", va="top", color="#142A36")
        card.text(0.07, 0.69, f"p = {panel['prob_cross']:.4f}", transform=card.transAxes,
                  fontsize=16, fontweight="bold", va="top",
                  color="#C65331" if panel["flagged"] else "#187B91")
        card.text(0.07, 0.53, f"Ego speed: {panel['ego_speed_kmh']:.1f} km/h", transform=card.transAxes,
                  fontsize=9.3, va="top", color="#263943")
        gt_label = "crossing" if panel["ground_truth_code"] == 1 else "non-crossing"
        card.text(0.07, 0.40, f"PIE label: {gt_label}", transform=card.transAxes,
                  fontsize=9.3, va="top", color="#263943")
        status = "flagged" if panel["flagged"] else "not flagged"
        card.text(0.07, 0.27, f"Prediction: {status}", transform=card.transAxes,
                  fontsize=9.3, va="top", color="#263943")
        if panel["first_crossing_state_frame"] is not None:
            timing = f"{panel['frames_to_first_crossing_state']} frames before onset"
        else:
            timing = f"{panel['video']}, frame {panel['frame']}"
        card.text(0.07, 0.14, timing, transform=card.transAxes,
                  fontsize=8.8, va="top", color="#50646C")
    fig.text(0.045, 0.965, "Offline five-seed BiLSTM demonstration on PIE test video",
             fontsize=13.5, fontweight="bold", color="#142A36", va="top")
    fig.text(0.045, 0.934, "YOLO26m + ByteTrack | event-anchored box and OBD-speed model | seed-mean probability",
             fontsize=8.8, color="#53666E", va="top")
    fig.text(0.045, 0.048, "Thick box: annotation-matched tracked target. Thin boxes: other tracked pedestrians. Faces blurred.",
             fontsize=8.2, color="#465B63")
    fig.text(0.045, 0.026, f"Fixed validation threshold = {THRESHOLD:.4f}. Each probability uses a contiguous 16-frame window ending at the shown frame.",
             fontsize=8.2, color="#465B63")
    pdf = ROOT / "figures/issue5_three_panel.pdf"
    png = ROOT / "figures/issue5_three_panel.png"
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(pdf.name, png.name)


if __name__ == "__main__":
    main()
