import cv2
import numpy as np

from generate import farthest_points, render, sample, visible

K = np.array([[500.0, 0, 320], [0, 500, 180], [0, 0, 1]])


def cube():
    v = np.array([[x, y, z] for x in (-20, 20) for y in (-20, 20) for z in (-20, 20)], float)
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return v[np.array(faces)]


def test_farthest_points_spread():
    v = np.random.default_rng(0).normal(size=(500, 3))
    p = farthest_points(v, 6)
    d = np.linalg.norm(p[:, None] - p[None], axis=2) + np.eye(6) * 99
    assert len(p) == 6 and d.min() > 1


def test_front_visible_back_hidden():
    img = np.zeros((360, 640, 3), np.uint8)
    R, t = np.eye(3), np.array([0, 0, 200.0])
    ids, depth = render(img, cube(), R, t, K, np.full(3, 200.0), np.array([0, 0, -1.0]), 0)
    kps = np.array([[0, 0, -20.0], [0, 0, 20.0]])
    kp_cam = kps + t
    uv = kp_cam @ K.T
    uv = uv[:, :2] / uv[:, 2:]
    assert visible(kp_cam, uv, ids, depth) == [True, False]
    assert img[180, 320].sum() > 0 and img[5, 5].sum() == 0


def test_sample_labels_match_projection():
    tris = cube()
    kps = tris.reshape(-1, 3)[:4]
    img, label, q = sample(3, tris, kps, K, (640, 360), [])
    R, t = np.array(label["R"]), np.array(label["t"])
    uv, _ = cv2.projectPoints(kps, cv2.Rodrigues(R)[0], t, K, np.zeros(5))
    assert img.shape == (360, 640, 3) and 50 <= q < 96
    assert np.allclose(uv.reshape(-1, 2), label["keypoints_2d"], atol=0.01)
    assert abs(np.linalg.det(R) - 1) < 1e-9
