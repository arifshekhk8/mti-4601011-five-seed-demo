"""Copy licensed/private inputs into this repository's ignored private/ folder.

No source or destination file is overwritten. All paths supplied by the user stay
out of public manifests and logs.
"""
import argparse
import shutil
from pathlib import Path

from common import ROOT, SEEDS


def copy(src, dst):
    if not src.is_file(): raise FileNotFoundError(src)
    if dst.exists(): raise FileExistsError(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(dst.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation-root", type=Path, required=True,
                    help="local pedestrian-intention-temporal-validity checkout with matched run checkpoints")
    ap.add_argument("--pie-root", type=Path, required=True, help="local PIE dataset root")
    ap.add_argument("--clips-root", type=Path, required=True,
                    help="folder containing set03/video_0012.mp4 and video_0016.mp4")
    ap.add_argument("--detector-weights", type=Path, required=True, help="local YOLO26m weights")
    args = ap.parse_args()
    e, pie, clips = args.evaluation_root, args.pie_root, args.clips_root
    plan = []
    for name in ("X.npy", "y.npy", "meta.pkl"):
        plan.append((e / "data/pie_clean" / name, ROOT / "private/data" / name))
    plan.append((e / "experiments/02_model_comparison/matched_comparison_results.json",
                 ROOT / "private/source/matched_comparison_results.json"))
    for seed in SEEDS:
        for name in ("best.pt", "final.json", "norm_mean.npy", "norm_std.npy"):
            plan.append((e / "runs/matched/BiLSTM" / f"seed{seed}" / name,
                         ROOT / "private/checkpoints" / f"seed{seed}" / name))
    for video in ("video_0012", "video_0016"):
        plan.append((clips / "set03" / f"{video}.mp4", ROOT / "private/videos" / f"{video}.mp4"))
        for sub, suffix in (("annotations_vehicle", "_obd.xml"),
                            ("annotations_attributes", "_attributes.xml"),
                            ("annotations", "_annt.xml")):
            plan.append((pie / sub / "set03" / f"{video}{suffix}",
                         ROOT / "private/annotations" / f"{video}{suffix}"))
    plan.append((args.detector_weights, ROOT / "private/checkpoints/yolo26m.pt"))
    missing = [src for src, _ in plan if not src.is_file()]
    existing = [dst for _, dst in plan if dst.exists()]
    if missing or existing:
        raise SystemExit(f"Preflight failed: {len(missing)} missing sources, {len(existing)} existing destinations. No files copied.")
    for src, dst in plan: copy(src, dst)


if __name__ == "__main__":
    main()
