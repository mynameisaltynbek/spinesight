import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage import measure

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "print"))
sys.path.insert(0, str(ROOT / "rig"))
from common import TOOL_DICT, TOOL_ID, load_stl
from make_parts import STEP, load, marker_holder, socket, solid


def light(vox, step, every):
    v, f, _, _ = measure.marching_cubes(np.pad(vox, 2).astype(float), 0.5, spacing=(step,) * 3, step_size=every)
    return v - 2 * step, f


def pack(v, f):
    return {
        "positions": base64.b64encode(np.asarray(v, "<f4").tobytes()).decode(),
        "index": base64.b64encode(np.asarray(f, "<u4")[:, ::-1].tobytes()).decode(),
    }


def indexed(tris):
    v, inv = np.unique(tris.reshape(-1, 3).round(3), axis=0, return_inverse=True)
    return v, inv.reshape(-1, 3)


out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "viewer" / "assets"
out.mkdir(parents=True, exist_ok=True)
m, sp = load(ROOT / "data" / "sub-verse031_seg-vert_msk.nii.gz", 22)
box, offset = socket(solid(m, sp))
vox, step = marker_holder()
meshes = {
    "vertebra": pack(*indexed(load_stl(ROOT / "print" / "parts" / "vertebra_L3.stl"))),
    "socket": pack(*light(box, STEP, 3)),
    "holder": pack(*light(vox, step, 3)),
}
json.dump(meshes, open(out / "meshes.json", "w"))
plan = json.load(open(ROOT / "outputs" / "plan.json"))
plan["socket_offset"] = offset.round(2).tolist()
plan["marker_bits"] = (cv2.aruco.generateImageMarker(TOOL_DICT, TOOL_ID, 6, borderBits=1) > 127).astype(int).tolist()
json.dump(plan, open(out / "data.json", "w"), indent=1)
print({p.name: p.stat().st_size for p in out.iterdir()})
