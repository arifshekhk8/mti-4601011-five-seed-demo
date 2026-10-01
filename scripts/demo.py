"""Offline PIE test-video YOLO + ByteTrack + frozen five-seed BiLSTM inference."""
import argparse
import csv
import json
import os
from collections import deque
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np

from common import CONFIG, ROOT, THRESHOLD, load_ensemble, predict_per_seed, sha256

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "private/cache/ultralytics"))
os.environ.setdefault("XDG_CONFIG_HOME", str(ROOT / "private/cache/config"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "private/cache/matplotlib"))
for key in ("YOLO_CONFIG_DIR", "XDG_CONFIG_HOME", "MPLCONFIGDIR"):
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def speed_map(video):
    root = ET.parse(ROOT / "private/annotations" / f"{video}_obd.xml").getroot()
    return {int(f.attrib["id"]): float(f.attrib["OBD_speed"]) for f in root.findall("frame")}


def run_scene(scene, models, stats, yolo_device):
    from ultralytics import YOLO

    scene_id, video = scene["scene"], scene["video"]
    start, end = scene["start_frame"], scene["end_frame"]
    cap = cv2.VideoCapture(str(ROOT / "private/videos" / f"{video}.mp4"))
    assert cap.isOpened()
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (1920, 1080)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    detector = YOLO(str(ROOT / "private/checkpoints/yolo26m.pt"))
    speeds = speed_map(video)
    buffers, last_seen = {}, {}
    output = ROOT / "results" / f"scene_{scene_id}_predictions.csv"
    fields = ["scene", "video", "frame", "window_start", "window_end", "track_id", "track_key",
              "x1", "y1", "x2", "y2", "ego_speed_kmh", "prob_cross", "flagged",
              "seed42_prob", "seed0_prob", "seed1_prob", "seed2_prob", "seed3_prob"]
    rows = []
    for frame_id in range(start, end + 1):
        ok, frame = cap.read()
        if not ok: raise RuntimeError(f"Video ended before {video} frame {frame_id}")
        speed = speeds.get(frame_id)
        if speed is None or not np.isfinite(speed):
            raise ValueError(f"Missing OBD speed for {video} frame {frame_id}")
        r = detector.track(frame, classes=[0], conf=CONFIG["detector"]["confidence"],
                           imgsz=CONFIG["detector"]["image_size"], tracker="bytetrack.yaml",
                           persist=True, device=yolo_device, verbose=False)[0]
        if r.boxes.id is None: continue
        ids = r.boxes.id.int().cpu().tolist()
        boxes = r.boxes.xyxy.cpu().numpy()
        pending = []
        for tid, box in zip(ids, boxes):
            if tid in last_seen and frame_id != last_seen[tid] + 1:
                buffers.pop(tid, None)
            last_seen[tid] = frame_id
            buf = buffers.setdefault(tid, deque(maxlen=16))
            buf.append(np.array([*box, speed], dtype=np.float32))
            if len(buf) == 16:
                pending.append((tid, box, np.stack(buf)))
        if pending:
            probs = predict_per_seed(models, stats, np.stack([item[2] for item in pending]), device="cpu", batch=64)
            ensembles = probs.mean(axis=0)
            for i, (tid, box, _) in enumerate(pending):
                row = {"scene": scene_id, "video": video, "frame": frame_id,
                       "window_start": frame_id - 15, "window_end": frame_id,
                       "track_id": tid, "track_key": f"{scene_id}:{video}:{tid}",
                       "x1": float(box[0]), "y1": float(box[1]),
                       "x2": float(box[2]), "y2": float(box[3]),
                       "ego_speed_kmh": speed, "prob_cross": float(ensembles[i]),
                       "flagged": int(ensembles[i] >= THRESHOLD)}
                row.update({f"seed{s}_prob": float(probs[j, i]) for j, s in enumerate(CONFIG["seeds"])})
                rows.append(row)
        if (frame_id - start) % 25 == 0 or frame_id == end:
            print(f"{scene_id} {video} frame {frame_id}/{end} predictions={len(rows)}", flush=True)
    cap.release()
    with output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    return {"scene": scene_id, "video": video, "start_frame": start, "end_frame": end,
            "panel_frame": scene["panel_frame"], "frames_processed": end - start + 1,
            "prediction_rows": len(rows), "unique_tracker_ids": len({r["track_key"] for r in rows}),
            "csv": output.name, "detector_device": yolo_device,
            "video_sha256": sha256(ROOT / "private/videos" / f"{video}.mp4")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda:0"])
    ap.add_argument("--scene", choices=["A", "B", "C", "all"], default="all")
    args = ap.parse_args()
    gate = json.loads((ROOT / "results/verification.json").read_text())
    if not gate["pass"]: raise RuntimeError("Quantitative parity gate failed")
    models, stats = load_ensemble("cpu")
    selected = CONFIG["scenes"] if args.scene == "all" else [s for s in CONFIG["scenes"] if s["scene"] == args.scene]
    manifests = [run_scene(scene, models, stats, args.device) for scene in selected]
    path = ROOT / "results/demo_run.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update({m["scene"]: m for m in manifests})
    path.write_text(json.dumps({k: old[k] for k in sorted(old)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
