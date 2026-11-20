import json
import sys

import cv2
import numpy as np

from common import board_points, open_camera, read, text

cap = open_camera(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
objs, imgs = [], []
while True:
    frame = read(cap)
    obj, img = board_points(frame)
    view = frame.copy()
    if obj is not None:
        for p in img.reshape(-1, 2):
            cv2.circle(view, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
    text(view, f"saved {len(objs)}   space: save view   q: finish")
    cv2.imshow("calibrate", view)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" ") and obj is not None:
        objs.append(obj)
        imgs.append(img)
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
if len(objs) < 10:
    raise SystemExit("need at least 10 views")
size = frame.shape[1::-1]
rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None)
json.dump({"K": K.tolist(), "dist": dist.ravel().tolist(), "rms_px": rms, "size": size, "views": len(objs)}, open("camera.json", "w"), indent=2)
print(f"camera.json saved, reprojection error {rms:.3f} px (good: below 0.5)")
