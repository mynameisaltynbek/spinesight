import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rig"))
sys.path.insert(0, str(ROOT / "model"))
from common import FONT, board_pose, draw_outline, load_camera, load_stl, open_camera, read, tool_pose
from score import score, tool_in_vertebra
from spine import load

COLORS = {"A": (0, 200, 0), "B": (0, 200, 200), "-": (200, 200, 200)}
FIELDS = ["time", "participant", "group", "phase", "pose_source", "side", "entry_mm", "angle_deg", "breach_mm", "grade", "wall", "depth_mm", "sensor_breach"]


def project(points, R, t, cam):
    p, _ = cv2.projectPoints(np.array(points, float), cv2.Rodrigues(R)[0], t, cam["K"], cam["dist"])
    return [tuple(int(v) for v in q) for q in p.reshape(-1, 2)]


def put(img, lines):
    for k, (s, color) in enumerate(lines):
        cv2.putText(img, s, (10, 30 + 30 * k), FONT, 0.8, color, 2)


ap = argparse.ArgumentParser()
ap.add_argument("participant")
ap.add_argument("group", choices=["control", "spinesight"])
ap.add_argument("phase", choices=["pre", "train", "post"])
ap.add_argument("--pose", choices=["board", "model"], default="board")
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--serial")
args = ap.parse_args()

cam = load_camera(ROOT / "rig" / "camera.json")
tris = load_stl(ROOT / "outputs" / "vertebra.stl")
screws = json.load(open(ROOT / "outputs" / "plan.json"))["screws"]
m, sp = load(ROOT / "data" / "sub-verse031_seg-vert_msk.nii.gz", 22)
dist = ndimage.distance_transform_edt(m, sampling=sp)
tip_m = np.array(json.load(open(ROOT / "rig" / "tool.json"))["tip"])
if args.pose == "board":
    reg = json.load(open(ROOT / "rig" / "vertebra_pose.json"))
    Rb, tb = np.array(reg["R"]), np.array(reg["t"])
else:
    from ensemble import load_models, predict
    nets, kps = load_models(ROOT / "model")
    th = json.load(open(ROOT / "model" / "threshold.json"))["spread_mm"]
ser = None
if args.serial:
    import serial
    ser = serial.Serial(args.serial, 115200, timeout=0)
show = args.phase == "train" and args.group == "spinesight"
log = ROOT / "study" / "attempts.csv"
new = not log.exists()
cap = open_camera(args.camera)
attempts, sensor_breach = 0, False
while True:
    frame = read(cap)
    view = frame.copy()
    vpose, res, status = None, None, ""
    if args.pose == "board":
        bp = board_pose(frame, cam)
        vpose = (bp[0] @ Rb, bp[0] @ tb + bp[1]) if bp else None
        status = "" if vpose else "board not visible"
    else:
        pose, sig = predict(nets, frame, kps, cam["K"], cam["dist"], screws)
        vpose = pose if pose and sig["spread_mm"] <= th else None
        status = "" if vpose else "NOT SURE: no score"
    tp = tool_pose(frame, cam)
    if vpose and tp:
        tip_v, d_v = tool_in_vertebra(tp, tip_m, vpose)
        res = score(tip_v, d_v, screws, m, sp, dist)
    if ser and ser.in_waiting and b"B" in ser.read(ser.in_waiting):
        sensor_breach = True
    if vpose:
        draw_outline(view, tris, *vpose, cam)
    if show and vpose:
        for s in screws:
            a, b = project([s["entry"], s["tip"]], *vpose, cam)
            cv2.line(view, a, b, (0, 200, 0), 2)
    if show and res:
        a, b = project([tip_v, tip_v + 45 * d_v], *vpose, cam)
        cv2.line(view, a, b, COLORS.get(res["grade"], (0, 0, 255)), 3)
    lines = [(f"{args.participant} {args.group} {args.phase}  attempts {attempts}  space: record  q: quit", (255, 255, 255))]
    if status:
        lines.append((status, (0, 0, 255)))
    elif not tp:
        lines.append(("tool marker not visible", (0, 0, 255)))
    elif show and res:
        lines.append((f"{res['side']}: entry {res['entry_mm']:.1f} mm, angle {res['angle_deg']:.1f} deg, grade {res['grade']}", COLORS.get(res["grade"], (0, 0, 255))))
        if res["wall"]:
            lines.append((f"breach {res['breach_mm']:.1f} mm through {res['wall']} wall at {res['depth_mm']:.0f} mm depth", COLORS.get(res["grade"], (0, 0, 255))))
    elif res:
        lines.append(("feedback hidden in this phase", (255, 255, 255)))
    if sensor_breach:
        lines.append(("SENSOR: wall touched", (0, 0, 255)))
    put(view, lines)
    cv2.imshow("SpineSight trainer", view)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" ") and res:
        with open(log, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
                new = False
            w.writerow({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "participant": args.participant, "group": args.group,
                        "phase": args.phase, "pose_source": args.pose, "sensor_breach": int(sensor_breach), **res})
        attempts += 1
        sensor_breach = False
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
