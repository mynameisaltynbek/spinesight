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

## Known limits

- The mask has 2 mm voxels left-right, so pedicle width is measured in 2 mm steps.
- The planner maximises clearance only; it does not follow surgical rules for the entry point yet, so left and right angles can differ.
- Trajectories are not validated by a surgeon.

## Licenses

- Code: MIT (`LICENSE`).
- Everything in `outputs/` is derived from VerSe and is CC BY-SA 4.0 (`outputs/LICENSE.md`).

VerSe: Sekuboyina A et al., Medical Image Analysis 2021; Löffler M et al., Radiology: AI 2020; Liebl H, Schinz D et al., 2021.
