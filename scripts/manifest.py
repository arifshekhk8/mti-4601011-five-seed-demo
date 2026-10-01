"""Write a public, path-safe manifest for this particular offline run."""
import json
import os
import platform
from pathlib import Path

from common import CONFIG, ROOT, sha256

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "private/cache/ultralytics"))
os.environ.setdefault("XDG_CONFIG_HOME", str(ROOT / "private/cache/config"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "private/cache/matplotlib"))
for key in ("YOLO_CONFIG_DIR", "XDG_CONFIG_HOME", "MPLCONFIGDIR"):
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

import cv2
import matplotlib
import numpy as np
import scipy
import sklearn
import torch
import ultralytics


def digest(relative):
    p = ROOT / relative
    if not p.is_file(): raise FileNotFoundError(relative)
    return sha256(p)


def main():
    verification = json.loads((ROOT / "results/verification.json").read_text())
    if not verification["pass"]: raise RuntimeError("Cannot manifest an unverified model")
    analysis = json.loads((ROOT / "results/analysis.json").read_text())
    run = json.loads((ROOT / "results/demo_run.json").read_text())
    expected = {"A", "B", "C"}
    assert set(run) == expected
    outputs = [".gitignore", "README.md", "config.json", "requirements.txt",
               "scripts/common.py", "scripts/prepare_inputs.py", "scripts/verify.py",
               "scripts/demo.py", "scripts/analyze.py", "scripts/figure.py",
               "scripts/figure_manuscript.py", "scripts/manifest.py",
               "figures/issue5_three_panel.pdf",
               "figures/issue5_three_panel.png", "figures/fig6_demo.png",
               "figures/fig6_demo.pdf", "figures/fig6_demo_dropin.zip",
               "results/verification.json",
               "results/verification_metrics.csv", "results/demo_run.json", "results/predictions_all.csv",
               "results/analysis.json", "results/scene_counts.csv", "results/panel_predictions.csv",
               "results/scene_A_predictions.csv", "results/scene_B_predictions.csv",
               "results/scene_C_predictions.csv", "reports/verification.md",
               "reports/demo_analysis.md", "reports/issue5_replacements.md",
               "reports/source_audit.md", "reports/figure_replacement.md"]
    manifest = {
        "purpose": "MTI 4601011 Issue 5 offline video demonstration",
        "source_repository_commit": CONFIG["source_commit"],
        "secondary_repository_commit": CONFIG["secondary_commit"],
        "seed_order": CONFIG["seeds"], "threshold": CONFIG["threshold"],
        "model_selection": CONFIG["model"]["selection"],
        "features": CONFIG["features"], "input_frames": CONFIG["input_frames"],
        "normalization": CONFIG["normalization"],
        "detector_and_tracker": CONFIG["detector"],
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "ultralytics": ultralytics.__version__, "opencv": cv2.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "verification_pass": verification["pass"],
        "checkpoint_sha256": verification["checkpoint_sha256"],
        "window_data_sha256": verification["data_sha256"],
        "controlled_record_sha256": verification["record_sha256"],
        "detector_weights_sha256": digest("private/checkpoints/yolo26m.pt"),
        "videos_sha256": {v: digest(f"private/videos/{v}.mp4") for v in ("video_0012", "video_0016")},
        "annotations_sha256": {f"{v}{s}": digest(f"private/annotations/{v}{s}")
                                for v in ("video_0012", "video_0016")
                                for s in ("_obd.xml", "_attributes.xml", "_annt.xml")},
        "scene_runs": run,
        "summary_counts": {k: analysis[k] for k in ("frames_processed", "pedestrian_frame_records", "unique_tracker_ids")},
        "public_output_sha256": {p: digest(p) for p in outputs},
        "raw_inputs_published": False,
    }
    (ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("run_manifest.json written")


if __name__ == "__main__":
    main()
