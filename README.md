# MTI 4601011 Issue 5: offline five-seed model demonstration

This repository contains a **new** video demonstration of the event-anchored, box-plus-speed BiLSTM from the [controlled evaluation](https://github.com/arifshekhk8/pedestrian-intention-temporal-validity) for manuscript `mti-4601011`. It uses seeds **42, 0, 1, 2, 3** and the frozen validation threshold **0.4579877257347107**. The [older PIE repository](https://github.com/arifshekhk8/pedestrian-crossing-intention-pie) includes an F1-optimized demo with threshold 0.5164303779602051; that is a different model arm.

The figure and statistics are an **offline, deliberately selected illustration** from two recorded PIE set03 test videos. They are not a full-video evaluation or a new estimate of model accuracy.

## Deliverables

- [`figures/issue5_three_panel.pdf`](figures/issue5_three_panel.pdf) and [`figures/issue5_three_panel.png`](figures/issue5_three_panel.png): face-blurred journal figure.
- [`figures/fig6_demo.png`](figures/fig6_demo.png) and [`figures/fig6_demo.pdf`](figures/fig6_demo.pdf): wide, three-panel figure matching the newly supplied manuscript ZIP's demo layout. [`fig6_demo_dropin.zip`](figures/fig6_demo_dropin.zip) contains the PNG at the manuscript's expected path.
- [`results/predictions_all.csv`](results/predictions_all.csv): 826 new pedestrian-frame predictions, with all five seed probabilities, ensemble probability, speed, track ID, and 16-frame window bounds. Separate scene CSVs are also provided.
- [`results/analysis.json`](results/analysis.json), [`results/scene_counts.csv`](results/scene_counts.csv), and [`results/panel_predictions.csv`](results/panel_predictions.csv): recalculated statistics and ground-truth timing.
- [`reports/verification.md`](reports/verification.md), [`results/verification.json`](results/verification.json), and [`results/verification_metrics.csv`](results/verification_metrics.csv): quantitative parity, checkpoint identities and SHA-256 hashes.
- [`reports/demo_analysis.md`](reports/demo_analysis.md): selection rules, denominators, interpretation, and limitations.
- [`reports/source_audit.md`](reports/source_audit.md): manuscript and repository provenance, including the older F1-arm distinction.
- [`reports/figure_replacement.md`](reports/figure_replacement.md): inspection of the newly supplied ZIP, figure mapping, and text consistency note.
- [`reports/issue5_replacements.md`](reports/issue5_replacements.md): concise manuscript Methods, Results, Discussion, and caption proposals.
- [`run_manifest.json`](run_manifest.json): input/output hashes and versioned run configuration.

## Reproduce locally

Python 3.13 and FFmpeg/Poppler are useful for video/PDF inspection. Create the environment **inside this checkout**:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Obtain the controlled-evaluation checkout with its locally trained `runs/matched/BiLSTM/seed{42,0,1,2,3}` checkpoints, the PIE dataset and two raw `set03` clips, and the local YOLO26m weights. The five checkpoints are *not* in the source repository's tracked Git files. Copy inputs into this checkout's ignored `private/` folder without overwriting any destination:

```bash
.venv/bin/python scripts/prepare_inputs.py \
  --evaluation-root /path/to/pedestrian-intention-temporal-validity \
  --pie-root /path/to/PIE \
  --clips-root /path/to/PIE_clips \
  --detector-weights /path/to/yolo26m.pt
```

The program checks that every source exists and every destination is empty **before copying**. If `private/` already contains inputs, retain them and start at verification. Compare local hashes with `run_manifest.json` and `results/verification.json` to establish exact input identity. Then run:

```bash
.venv/bin/python scripts/verify.py
.venv/bin/python scripts/demo.py --scene all --device mps
.venv/bin/python scripts/analyze.py
.venv/bin/python scripts/figure.py
.venv/bin/python scripts/figure_manuscript.py
.venv/bin/python scripts/manifest.py
```

Use `--device cpu` for the detector if MPS is unavailable. The model parity gate always runs on CPU. Detector and tracker outputs may change with hardware or software versions, so the published CSVs and package versions are the exact output record. `scripts/demo.py` refuses to run unless the quantitative parity gate has passed.

## Protocol and scope

The source configuration is `experiments/02_model_comparison/matched_comparison_results.json:families.BiLSTM` at source commit `c3c535c2f80eac1957b0572d0c1a2a164c2c7d3c`. Input features are `[x1, y1, x2, y2, vehicle_speed]`, with original 1920 × 1080 frame pixels and OBD speed in km/h. Each seed's saved train-only z-score normalization is applied before inference. The five probabilities are averaged and compared with the one threshold fitted on the averaged validation probabilities. No test/video threshold optimization occurs.

The full video inference front end is YOLO26m person class, confidence 0.3, 640-pixel inference size, and ByteTrack. The three 111-frame segments and target pedestrians are fixed in `config.json`. Panel A's crossing window is 5705–5720 and ends 46 frames before the first ground-truth crossing state at 5766. Panels B and C show explicit PIE non-crossing codes (`-1`). `reports/demo_analysis.md` lists all timing and denominators.

The PIE video and annotations are attributed to Rasouli et al., *PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and Trajectory Prediction*, ICCV 2019. The public outputs include only blurred stills. Raw videos, annotations, unblurred frames, manuscript copies, and model weights remain ignored under `private/`.
