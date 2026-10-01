# Issue 5 offline demonstration: selection and measurements

## Frozen model and quantitative gate

The deployed predictor is the **event-anchored, box-plus-OBD-speed BiLSTM** from the current controlled matched comparison, not the older F1-selected demo arm. It uses seeds 42, 0, 1, 2, and 3 from `runs/matched/BiLSTM`, best validation-AUC checkpoints, a 64-dimensional input projection, two bidirectional LSTM layers with hidden size 256 and dropout 0.2, and a linear output head. The five sigmoid probabilities are averaged. The operating threshold is the recorded, validation-selected **0.4579877257347107**. Input order is `[x1, y1, x2, y2, OBD_speed]`, in original 1920 × 1080 pixel coordinates and km/h, over 16 contiguous frames. Each feature is standardized using the saved training-split mean and standard deviation.

`results/verification.json` and `reports/verification.md` show exact parity against the controlled comparison: ensemble test F1 0.831334, ROC AUC 0.931970, PR AUC 0.876826 on 2,094 test windows. The *mean of five individual checkpoint metrics* is a different quantity: F1 0.827587 ± 0.017371 and ROC AUC 0.924225 ± 0.008598. Validation recomputation returned the exact recorded threshold. The older repository's F1 arm A3 uses threshold 0.5164303779602051 and has different checkpoints and results.

## Scene choice and ground truth

Scenes were fixed using PIE pedestrian annotations and OBD speed before viewing the new model predictions. They are short, illustrative segments from two **set03 test videos**, not a random sample and not the full test split. No annotation enters the detector or predictor. The XML is used only to select and interpret the examples.

| Panel | Video, processed frames | Display frame | PIE target ID / label | Ground-truth timing | Display speed |
|---|---|---:|---|---|---:|
| A | `video_0012`, 5650–5760 | 5720 | `3_12_741`, crossing | First `crossing` state and annotated crossing point at frame 5766; observation window 5705–5720 ends **46 frames (1.53 s) before onset** | 0.0 km/h |
| B | `video_0012`, 3580–3690 | 3650 | `3_12_726`, non-crossing (`-1`) | No crossing onset in the target annotation; its reference crossing point is frame 3718 | 0.0 km/h |
| C | `video_0016`, 1225–1335 | 1295 | `3_16_900`, non-crossing (`-1`) | No crossing onset in the target annotation; its reference crossing point is frame 1341 | 28.0026 km/h |

Each processed segment contains 111 video frames. YOLO26m person detection (`conf=0.3`, `imgsz=640`) and ByteTrack association were run on every frame. A prediction row is emitted for each tracked person once its track has 16 consecutive boxes. A gap resets its feature buffer. The ego speed is the recorded OBD value at the original frame index; missing speeds cause the script to stop. The detector box replaces the annotated box in the model input. The public figure blurs the head region of annotated pedestrians, low-confidence detected bystanders, and tracked pedestrians; only blurred frames are exported.

## Newly calculated output

| Scene | Pedestrian-frame prediction rows | Distinct tracker IDs | Flagged rows | At display frame |
|---|---:|---:|---:|---|
| A | 179 | 3 | 179 | 2 of 2 tracked pedestrians flagged; target p = 0.9383 |
| B | 170 | 4 | 162 | 2 of 2 tracked pedestrians flagged; non-crossing target p = 0.8430 |
| C | 477 | 16 | 0 | 0 of 7 tracked pedestrians flagged; non-crossing target p = 0.0078 |

Across the **333 processed video frames**, the exported CSVs contain **826 pedestrian-frame predictions from 23 segment-scoped tracker IDs**. Across those 826 records, Pearson's correlation between OBD speed and ensemble probability is **r = −0.970184**. At exactly **0 km/h**, **300/300 pedestrian-frame records (100.0%)** are flagged, spanning 7 tracker IDs. Above **20 km/h**, **0/477 records (0.0%)** are flagged, spanning 16 tracker IDs. The remaining 49 records have intermediate speeds. These percentages use prediction rows as their denominators, not unique pedestrians. Tracker IDs can fragment or switch and are not PIE identities; the 23 IDs should not be interpreted as 23 independently sampled pedestrians. The repeated-frame correlation is descriptive, and its ordinary independent-observation p-value is not used for inference.

The target matches in panels A, B, and C have detector-to-annotation box IoU 0.913, 0.884, and 0.761, respectively. Their probabilities are taken from the new prediction CSVs. The stationary non-crossing example in B is a visible false alert. The opposing speed groups were chosen in advance, so the large pooled correlation must not be presented as a full-video or population estimate. The demo does not provide new test-set classification metrics, calibrated risk, or a claim about deployment safety.

## Public and private material

The repository contains the scripts, preselected scene specification, verification hashes, prediction CSVs, summary tables, and blurred PDF/PNG figure. Raw PIE videos and XML, the manuscript, and model weights remain in the ignored `private/` directory. No unblurred frame is published.
