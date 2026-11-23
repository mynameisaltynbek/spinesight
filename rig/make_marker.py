import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import TOOL_DICT, TOOL_ID, TOOL_MM

img = cv2.aruco.generateImageMarker(TOOL_DICT, TOOL_ID, 600, borderBits=1)
fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4))
for k, y in enumerate((200, 120)):
    ax = fig.add_axes([(105 - TOOL_MM / 2) / 210, y / 297, TOOL_MM / 210, TOOL_MM / 297])
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.axis("off")
fig.text(0.5, 0.88, f"Tool marker: ArUco 4x4 id {TOOL_ID}. Black square must measure {TOOL_MM:.0f} mm.", ha="center", fontsize=11)
fig.text(0.5, 0.84, "Glue on the marker holder plate, centred, upright, with the rod pointing down.", ha="center", fontsize=11)
fig.savefig("tool_marker.pdf")
print(f"tool_marker.pdf: A4, two copies. Print at 100% and check the black square is {TOOL_MM:.0f} mm.")
