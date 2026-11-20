import json
import sys
from pathlib import Path

import cv2
import numpy as np

from common import board_pose, draw_outline, load_camera, load_stl, open_camera, read, text

cam = load_camera()
reg = json.load(open("vertebra_pose.json"))
Rb, tb = np.array(reg["R"]), np.array(reg["t"])
tris = load_stl(sys.argv[1] if len(sys.argv) > 1 else "../outputs/vertebra.stl")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "../test_real")
out.mkdir(exist_ok=True)
n = len(list(out.glob("*.png")))
cap = open_camera(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
while True:
    frame = read(cap)
    pose = board_pose(frame, cam)
    view = frame.copy()
    if pose:
        R, t = pose[0] @ Rb, pose[0] @ tb + pose[1]
        draw_outline(view, tris, R, t, cam)
    text(view, f"saved {n}   board {'ok' if pose else 'not found'}   space: save   q: quit")
    cv2.imshow("capture", view)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" ") and pose:
        cv2.imwrite(str(out / f"{n:04d}.png"), frame)
        json.dump({"R": R.tolist(), "t": t.tolist(), "K": cam["K"].tolist(), "dist": cam["dist"].tolist()}, open(out / f"{n:04d}.json", "w"), indent=2)
        n += 1
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
