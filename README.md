# SpineSight

A low-cost trainer for pedicle screw placement: a 3D-printed vertebra made from a real CT scan, a webcam, and a model that scores the student's trajectory and refuses to score when it is not sure.

This repository is at step 1: CT segmentation → printable vertebra (STL) → planned screw trajectories.

![L3 with planned screws](outputs/preview.png)

## What `spine.py` does

1. Loads a VerSe vertebra mask (NIfTI) and keeps one vertebra (default: label 22 = L3).
2. Builds a smooth surface (Gaussian blur + marching cubes) and saves it as STL, in millimetres.
3. Finds the spinal canal and, on each side, the narrowest slice of the pedicle (the isthmus).
4. Tries screw directions through the isthmus (medial −10…44°, cranial −14…14°, step 2°) and keeps the one with the largest distance to the bone surface that still reaches the middle of the vertebral body.
5. Reports entry point, tip, length, angles, pedicle size, clearance and breach (clearance < 0 means the 6.5 mm screw would leave the bone).

## Run

```
pip install -r requirements.txt
python spine.py data/sub-verse031_seg-vert_msk.nii.gz outputs 22
python -m pytest -q
```

Input data (not in the repo): VerSe 2019, case `sub-verse031`, from https://osf.io/jtfa5/
- mask: https://osf.io/download/92efg/ → `data/sub-verse031_seg-vert_msk.nii.gz`
- CT: https://osf.io/download/wxc4t/ → `data/sub-verse031_ct.nii.gz`

## Outputs

| File | What |
|---|---|
| `outputs/vertebra.stl` | L3, ready to print (mm) |
| `outputs/vertebra_with_screws.stl` | L3 with the two planned screws, for viewing |
| `outputs/plan.json` | sizes for caliper check, trajectories, clearance |
| `outputs/preview.png` | top and side view |

## Current result (L3, sub-verse031)

| | Left | Right |
|---|---|---|
| Medial angle | 20° | 2° |
| Cranial angle | 6° | 10° |
| Screw length | 44.5 mm | 40.5 mm |
| Clearance (6.5 mm screw) | 1.33 mm | 0.87 mm |

Vertebral body: 48 × 36 × 32 mm (width × depth × height).

## Camera rig and ground truth (`rig/`)

Ground truth for the real test set: a ChArUco board and the printed vertebra glued to one rigid plate, side by side. The board gives the camera pose; a one-time registration gives the vertebra pose on the board.

Run from the `rig` folder:

```
python make_board.py
python calibrate.py
python register.py
python capture.py
python -m pytest -q
```

1. `make_board.py` writes `board.pdf` (A4, 7×5 squares of 30 mm). Print at 100% and measure a square; if it is not 30 mm, set `SQUARE_MM` in `common.py`.
2. `calibrate.py` — show the board from 15–30 angles, press space for each view, q to finish. Saves `camera.json`. Good result: below 0.5 px.
3. `register.py` — writes `landmarks.png` with 4 points (spinous tip, left and right transverse tips, front of the body). Take 3+ views (space), then click each point in each view (enter = ok, s = not visible). Rays from all views are triangulated in board coordinates and matched to the STL points (Kabsch). Saves `vertebra_pose.json` with the error per point in mm.
4. `capture.py` — live view with the STL outline drawn on the vertebra. Space saves the frame and its pose to `test_real/`. Collect 300–500 frames (light, background, distance, occlusion) and do not change them later.

## Synthetic training data (`synth/`)

```
cd synth
python generate.py 10000
python -m pytest -q
```

- 10 keypoints are picked on the STL surface by farthest point sampling (`synth_out/keypoints.json`).
- Each image: random rotation, distance 250–600 mm, a simple numpy/OpenCV renderer (painter's algorithm, diffuse + specular light), random background, 0–2 occluders, blur, noise, brightness and JPEG quality.
- Each label (`NNNNNN.json`) has the same pose format as `rig/capture.py` plus `keypoints_2d` and `visible` (hidden by the vertebra itself or by an occluder).
- Uses `rig/camera.json` when it exists, at half resolution.
- Put your own background photos (jpg/png) into `synth/backgrounds/`; without them only generated backgrounds are used.
- `synth_out/preview.jpg` shows 16 samples (green = visible keypoint, red = hidden); `preview_real.jpg` shows 16 real frames when `test_real/` exists.

## Pose model (`model/`)

Keypoint heatmaps + PnP, following "Simple Baselines for Human Pose Estimation" (Xiao et al., 2018): ResNet-18 (ImageNet weights) → 3 transposed convolutions → one heatmap per keypoint at 1/4 resolution. Input 640×352. Keypoints are read as the weighted mean around the heatmap peak, then `solvePnPRansac` (EPnP) + Levenberg–Marquardt refinement gives the vertebra pose.

```
cd model
python train.py ../synth_out 30 0
python evaluate.py model_s0.pt ../synth_out
python evaluate.py model_s0.pt ../test_real
python -m pytest -q
```

- `train.py FOLDER EPOCHS SEED` trains on the first 90% of synthetic images, keeps 5% for calibration (not touched) and the last 5% for validation, and saves the best `model_s{SEED}.pt`.
- `evaluate.py` on `synth_out` uses only those validation images; on `test_real` it uses every frame. It prints median and 90th percentile of: rotation error, translation error, ADD (mean distance of model points), screw entry point error and screw direction error for both planned screws, keypoint error in pixels, and the share of frames with ADD below 10% of the vertebra size. Per-frame numbers go to `results.csv`.

### Training on Google Colab (free GPU)

```
!git clone https://github.com/mynameisaltynbek/spinesight
%cd spinesight
!pip install -q -r requirements.txt
!python synth/generate.py 10000
%cd model
!for s in 0 1 2 3 4; do python train.py ../synth_out 30 $s; done
```

Upload your `synth/backgrounds/` photos before `generate.py`, and download `model/model_s*.pt` when training ends.

## Knowing when not to answer (`model/ensemble.py`)

An ensemble of 3–5 models trained with different seeds. For each frame the heatmaps are averaged for the final pose, and three uncertainty signals are computed:

| Signal | Meaning |
|---|---|
| `spread_mm` | how far the screw entry points of the single models are from their mean (worst screw) |
| `low_confidence` | 1 − mean heatmap peak of the averaged heatmaps |
| `reprojection_px` | mean distance between predicted keypoints and the keypoints projected from the final pose |

A frame is *bad* when the screw entry point error is above 3 mm or no pose is found.

```
python ensemble.py calibrate ../synth_out
python ensemble.py test ../synth_out
python ensemble.py test ../test_real
python -m pytest -q
```

- `calibrate` uses the 5% calibration split, draws the risk–coverage curve for each signal (`risk_coverage_calibrate.png`, AURC: lower is better) and picks the largest `spread_mm` threshold with at most 5% bad answers (`model/threshold.json`).
- `test` applies that threshold and reports how many frames are answered and how many answers are bad.

## Trainer mode (`trainer/`)

The student holds a Ø4 mm rod in the printed `marker_holder.stl` with a 30 mm ArUco marker (`rig/make_marker.py` → `tool_marker.pdf`). The vertebra sits in `socket_L3.stl` next to the ChArUco board.

```
cd rig
python make_marker.py
python pivot.py
cd ../trainer
python trainer.py P01 spinesight train
python trainer.py P01 spinesight post --pose model --serial COM3
python -m pytest -q
```

- `pivot.py` finds the rod tip in the marker frame: put the tip in a small hole and tilt the tool in all directions while recording (r). Least-squares pivot calibration; saves `rig/tool.json` with the error in mm.
- `trainer.py PARTICIPANT GROUP PHASE` shows the vertebra outline, the planned screws and the tool axis, and scores the current aim: distance from the tool tip to the planned entry point, angle to the planned direction, and the predicted cortical breach of a 6.5 mm screw along the tool axis, graded A–E like Gertsbein–Robbins (A none, B < 2 mm, C < 4, D < 6, E ≥ 6). Feedback is shown only for group `spinesight` in phase `train`; in `pre`, `post` and for `control` it is recorded but hidden. Space appends the attempt to `study/attempts.csv` (not in the repo).
- `--pose board` (default) takes the vertebra pose from the board and `rig/vertebra_pose.json`; `--pose model` uses the ensemble and gives no score when the spread is above `model/threshold.json`.
- `--serial COMx` reads the breach sensor.

### Breach sensor (`trainer/breach_sensor/breach_sensor.ino`)

Arduino Uno or Nano. Conductive paint or copper tape on the inner walls of the canal and the medial pedicle wall → pin D2; the metal rod → GND. Touching the wall closes the circuit: LED 13 lights, a buzzer on D8 beeps and the board sends `B` over USB serial at 115200 baud.

### From attempts to statistics

```
cd study
python summarize.py attempts.csv
python stats.py results.csv
```

`summarize.py` averages the entry point error per participant for `pre` and `post` and marks a breach when any attempt in the phase is grade C–E or touched the sensor.

## Known limits

- The mask has 2 mm voxels left-right, so pedicle width is measured in 2 mm steps.
- The planner maximises clearance only; it does not follow surgical rules for the entry point yet, so left and right angles can differ.
- Trajectories are not validated by a surgeon.

## Licenses

- Code: MIT (`LICENSE`).
- Everything in `outputs/` is derived from VerSe and is CC BY-SA 4.0 (`outputs/LICENSE.md`).

VerSe: Sekuboyina A et al., Medical Image Analysis 2021; Löffler M et al., Radiology: AI 2020; Liebl H, Schinz D et al., 2021.
