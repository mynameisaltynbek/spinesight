import json
import sys

import cv2
import numpy as np

from common import load_camera, open_camera, read, solve_pivot, text, tool_pose

cam = load_camera()
cap = open_camera(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
poses, recording = [], False
while True:
    frame = read(cap)
    pose = tool_pose(frame, cam)
    if recording and pose:
        poses.append(pose)
    view = frame.copy()
    if pose:
        cv2.drawFrameAxes(view, cam["K"], cam["dist"], cv2.Rodrigues(pose[0])[0], pose[1], 20)
    text(view, f"poses {len(poses)}   {'RECORDING' if recording else 'paused'}   r: record on/off   q: finish")
    cv2.imshow("pivot", view)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        recording = not recording
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
if len(poses) < 50:
    raise SystemExit("need at least 50 poses: keep the tip in the hole and tilt the tool in all directions")
z = np.array([R[:, 2] for R, _ in poses])
spread = float(np.degrees(np.arccos(np.clip(z @ z.mean(0) / np.linalg.norm(z.mean(0)), -1, 1))).max())
tip, pivot, rms = solve_pivot(poses)
json.dump({"tip": tip.tolist(), "rms_mm": rms, "tilt_deg": spread, "poses": len(poses)}, open("tool.json", "w"), indent=2)
print(f"tool.json saved: tip {tip.round(1)} mm in marker frame, error {rms:.2f} mm, tilt range {spread:.0f} deg (good: error < 1 mm, tilt > 20 deg)")
