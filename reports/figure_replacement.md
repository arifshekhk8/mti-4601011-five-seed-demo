# Drop-in manuscript demo figure

The newly supplied manuscript ZIP (SHA-256 `71c714b0f968e5fe3447660ae26380ef70de7fadb72a342cb7de7574d7ca3f99`) was inspected without modifying it. Its `final.tex` references `figures/fig6_demo.png` for the model demonstration. The original image is a wide, three-panel montage of 2793 × 666 pixels. `figures/fig7_scenes.png` is a separate PIE/JAAD/IDD-PeD scene comparison and was not changed.

The new [`fig6_demo.png`](../figures/fig6_demo.png) is a three-panel, 4200 × 1002 pixel replacement with the same approximate aspect ratio, so the manuscript's existing `\includegraphics[width=\textwidth]{figures/fig6_demo.png}` can use it directly. [`fig6_demo.pdf`](../figures/fig6_demo.pdf) is a vector-text companion. [`fig6_demo_dropin.zip`](../figures/fig6_demo_dropin.zip) contains only `figures/fig6_demo.png` at the manuscript's expected path. It does not contain or alter any manuscript source.

The boxes and probabilities come from the newly verified five-seed ensemble and YOLO/ByteTrack prediction CSVs. The thick white-backed outline marks the target matched to a PIE annotation; thinner outlines show other tracked pedestrians. Orange means the frozen threshold 0.4579877 was met; teal means it was not. Head regions identified through PIE annotations, a low-confidence person-detection pass, and tracked boxes are blurred in the public image. Text sits outside the images to prevent overlapping probability labels.

| Panel | Example | Ensemble probability | OBD speed | Ground-truth check |
|---|---|---:|---:|---|
| A | `video_0012`, frame 5720 | 0.9383 | 0.0 km/h | Crossing onset frame 5766, 46 frames after the displayed frame |
| B | `video_0012`, frame 3650 | 0.8430 | 0.0 km/h | Non-crossing target; false alert |
| C | `video_0016`, frame 1295 | 0.0078 | 28.0 km/h | Non-crossing target; not flagged |

**Text consistency:** The supplied ZIP's existing Results paragraph and caption still describe the older demo, including `r=-0.892`, the older scene order, and a 14 km/h scene. Replacing only its PNG will leave those statements inconsistent with the figure. The original ZIP is unchanged; suggested matching text is in [`issue5_replacements.md`](issue5_replacements.md).
