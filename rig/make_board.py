import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import BOARD, SQUARE_MM

w, h = 7 * SQUARE_MM, 5 * SQUARE_MM
img = BOARD.generateImage((round(w * 12), round(h * 12)))
fig = plt.figure(figsize=(297 / 25.4, 210 / 25.4))
ax = fig.add_axes([(297 - w) / 2 / 297, (210 - h) / 2 / 210, w / 297, h / 210])
ax.imshow(img, cmap="gray", interpolation="nearest")
ax.axis("off")
fig.savefig("board.pdf")
print(f"board.pdf: A4 landscape. Print at 100% (actual size), then check that one square is {SQUARE_MM} mm.")
