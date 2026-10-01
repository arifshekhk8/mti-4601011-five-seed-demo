"""Recompute all displayed scene counts and speed summaries from new CSVs."""
import csv
import json
from collections import Counter
from xml.etree import ElementTree as ET

import numpy as np
from scipy.stats import pearsonr

from common import CONFIG, ROOT, THRESHOLD


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return intersection / (aa + bb - intersection) if aa + bb > intersection else 0.0


def gt_info(video, frame, target):
    attrs = {p.attrib["id"]: p.attrib for p in ET.parse(ROOT / "private/annotations" / f"{video}_attributes.xml").getroot().findall("pedestrian")}
    boxes = []
    onset = None
    for track in ET.parse(ROOT / "private/annotations" / f"{video}_annt.xml").getroot().findall("track"):
        if track.get("label") != "pedestrian": continue
        track_boxes = track.findall("box")
        if not track_boxes: continue
        pid = next((a.text for a in track_boxes[0].findall("attribute") if a.get("name") == "id"), None)
        if pid is None: continue
        for box in track_boxes:
            if box.get("outside") == "1": continue
            f = int(box.get("frame"))
            if pid == target:
                state = next((a.text for a in box.findall("attribute") if a.get("name") == "cross"), None)
                if state == "crossing" and (onset is None or f < onset): onset = f
            if f == frame:
                boxes.append({"ped_id": pid, "box": [float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")],
                              "crossing_code": int(attrs[pid]["crossing"]),
                              "crossing_point": int(attrs[pid]["crossing_point"])})
    target_attr = attrs[target]
    target_box = next((b for b in boxes if b["ped_id"] == target), None)
    if target_box is None: raise ValueError(f"Target {target} absent from {video} frame {frame}")
    return {"target": target_box, "first_crossing_state_frame": onset, "all_annotated_boxes": boxes,
            "crossing_code": int(target_attr["crossing"]), "crossing_point": int(target_attr["crossing_point"])}


def load_scene(scene):
    p = ROOT / "results" / f"scene_{scene['scene']}_predictions.csv"
    with p.open(newline="") as f: rows = list(csv.DictReader(f))
    for row in rows:
        for k in ("frame", "window_start", "window_end", "track_id", "flagged"):
            row[k] = int(row[k])
        for k in ("x1", "y1", "x2", "y2", "ego_speed_kmh", "prob_cross"):
            row[k] = float(row[k])
        seed_mean = sum(float(row[f"seed{s}_prob"]) for s in CONFIG["seeds"]) / 5
        assert abs(seed_mean - row["prob_cross"]) < 3e-7  # float32 mean rounding
    if not rows: raise ValueError(f"No 16-frame predictions in scene {scene['scene']}")
    assert all(row["window_end"] - row["window_start"] == 15 for row in rows)
    assert all(row["window_end"] == row["frame"] for row in rows)
    assert all(scene["start_frame"] <= row["frame"] <= scene["end_frame"] for row in rows)
    assert all(row["flagged"] == int(row["prob_cross"] >= THRESHOLD) for row in rows)
    assert len(rows) == len({(r["scene"], r["frame"], r["track_id"]) for r in rows})
    return rows


def main():
    all_rows, panel_records, scene_counts = [], [], []
    for scene in CONFIG["scenes"]:
        rows = load_scene(scene)
        all_rows += rows
        panel = [r for r in rows if r["frame"] == scene["panel_frame"]]
        gt = gt_info(scene["video"], scene["panel_frame"], scene["target_ped_id"])
        target_box = gt["target"]["box"]
        ranked = sorted(((iou(target_box, [r[k] for k in ("x1","y1","x2","y2")]), r) for r in panel), key=lambda z: z[0], reverse=True)
        best_iou, match = ranked[0] if ranked else (0, None)
        if best_iou < 0.1: match = None
        speed_root = ET.parse(ROOT / "private/annotations" / f"{scene['video']}_obd.xml").getroot()
        speed = next(float(f.get("OBD_speed")) for f in speed_root.findall("frame") if int(f.get("id")) == scene["panel_frame"])
        if gt["crossing_code"] == 1:
            assert gt["first_crossing_state_frame"] is not None
            assert 30 <= gt["first_crossing_state_frame"] - scene["panel_frame"] <= 60
            assert scene["panel_frame"] < gt["crossing_point"]
        else:
            assert gt["crossing_code"] == -1
        record = {"scene": scene["scene"], "video": scene["video"], "frame": scene["panel_frame"],
                  "ego_speed_kmh": speed, "target_ped_id": scene["target_ped_id"],
                  "ground_truth_code": gt["crossing_code"], "crossing_point": gt["crossing_point"],
                  "first_crossing_state_frame": gt["first_crossing_state_frame"],
                  "frames_to_first_crossing_state": None if gt["first_crossing_state_frame"] is None else gt["first_crossing_state_frame"] - scene["panel_frame"],
                  "target_gt_box": target_box, "all_annotated_boxes": gt["all_annotated_boxes"],
                  "matched_track_id": match["track_id"] if match else None,
                  "match_iou": best_iou, "prob_cross": match["prob_cross"] if match else None,
                  "flagged": match["flagged"] if match else None,
                  "window_start": match["window_start"] if match else None,
                  "detected_box": [match[k] for k in ("x1","y1","x2","y2")] if match else None,
                  "panel_prediction_count": len(panel), "panel_flagged_count": sum(r["flagged"] for r in panel),
                  "panel_boxes": [{"track_id": r["track_id"], "box": [r[k] for k in ("x1","y1","x2","y2")],
                                   "prob_cross": r["prob_cross"], "flagged": r["flagged"]} for r in panel]}
        panel_records.append(record)
        scene_counts.append({"scene": scene["scene"], "video": scene["video"], "start_frame": scene["start_frame"],
                             "end_frame": scene["end_frame"], "frames_processed": scene["end_frame"] - scene["start_frame"] + 1,
                             "prediction_rows": len(rows), "unique_tracker_ids": len({r["track_key"] for r in rows}),
                             "flagged_rows": sum(r["flagged"] for r in rows),
                             "panel_frame": scene["panel_frame"], "panel_prediction_rows": len(panel),
                             "panel_flagged_rows": sum(r["flagged"] for r in panel)})
    p0 = [r for r in all_rows if r["ego_speed_kmh"] == 0]
    p20 = [r for r in all_rows if r["ego_speed_kmh"] > 20]
    assert p0 and p20
    corr = pearsonr([r["ego_speed_kmh"] for r in all_rows], [r["prob_cross"] for r in all_rows])
    groups = {}
    for label, group in [("zero_kmh", p0), ("above_20_kmh", p20)]:
        groups[label] = {"pedestrian_frame_records": len(group), "flagged_records": sum(r["flagged"] for r in group),
                         "flagged_percent": 100 * sum(r["flagged"] for r in group) / len(group),
                         "unique_tracker_ids": len({r["track_key"] for r in group}),
                         "tracker_ids_flagged_at_least_once": len({r["track_key"] for r in group if r["flagged"]})}
    summary = {"scope": "three annotation-and-speed-selected PIE test-video segments, not the full PIE test set",
               "frames_processed": sum(s["frames_processed"] for s in scene_counts),
               "pedestrian_frame_records": len(all_rows), "unique_tracker_ids": len({r["track_key"] for r in all_rows}),
               "pearson_r_pedestrian_frame": float(corr.statistic), "pearson_pvalue_naive": float(corr.pvalue),
               "speed_groups": groups, "scene_counts": scene_counts, "panels": panel_records}
    (ROOT / "results/analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (ROOT / "results/scene_counts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scene_counts[0]), lineterminator="\n"); w.writeheader(); w.writerows(scene_counts)
    with (ROOT / "results/panel_predictions.csv").open("w", newline="") as f:
        cols = ["scene","video","frame","ego_speed_kmh","target_ped_id","ground_truth_code","crossing_point",
                "first_crossing_state_frame","frames_to_first_crossing_state","matched_track_id","match_iou",
                "window_start","prob_cross","flagged","panel_prediction_count","panel_flagged_count"]
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n");w.writeheader();w.writerows({k:p[k] for k in cols} for p in panel_records)
    with (ROOT / "results/predictions_all.csv").open("w", newline="") as f:
        fields = list(all_rows[0]);w=csv.DictWriter(f, fieldnames=fields, lineterminator="\n");w.writeheader();w.writerows(all_rows)
    print(json.dumps({"rows":len(all_rows),"tracks":summary["unique_tracker_ids"],"r":summary["pearson_r_pedestrian_frame"],
                      "speed_groups":groups,"panels":[{k:p[k] for k in ("scene","match_iou","prob_cross","flagged","frames_to_first_crossing_state")} for p in panel_records]}, indent=2))


if __name__ == "__main__":
    main()
