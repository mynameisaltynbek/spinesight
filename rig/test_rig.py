import cv2
import numpy as np

from common import BOARD, TOOL_DICT, TOOL_ID, board_pose, kabsch, load_stl, ray, solve_pivot, stl_landmarks, tool_pose, triangulate

CAM = {"K": np.array([[1000.0, 0, 700], [0, 1000, 500], [0, 0, 1]]), "dist": np.zeros(5)}


def test_kabsch_recovers_motion():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 3)) * 30
    R = cv2.Rodrigues(np.array([0.3, -0.5, 1.1]))[0]
    t = np.array([10.0, -4, 7])
    R2, t2 = kabsch(a, a @ R.T + t)
    assert np.allclose(R2, R) and np.allclose(t2, t)


def test_rays_meet_at_point():
    X = np.array([40.0, 20, 10])
    centers, dirs = [], []
    for rv, tv in (([0.1, 0.2, 0], [-50, -30, 400]), ([-0.2, 0.4, 0.1], [-20, -60, 350])):
        R = cv2.Rodrigues(np.array(rv))[0]
        t = np.array(tv, float)
        uv, _ = cv2.projectPoints(X[None], np.array(rv), t, CAM["K"], CAM["dist"])
        c, d = ray(R, t, CAM, uv.ravel())
        centers.append(c)
        dirs.append(d)
    assert np.linalg.norm(triangulate(centers, dirs) - X) < 1e-6


def test_board_is_detected():
    img = cv2.cvtColor(BOARD.generateImage((1400, 1000), marginSize=50), cv2.COLOR_GRAY2BGR)
    pose = board_pose(img, CAM)
    assert pose is not None and abs(np.linalg.det(pose[0]) - 1) < 1e-6


def test_stl_landmarks(tmp_path):
    tris = np.array([[[0, 0, 0], [10, 5, 0], [-3, 2, 1]], [[1, 9, 0], [2, -4, 0], [0, 0, 5]]], np.float32)
    data = np.zeros(2, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    data["v"] = tris
    p = tmp_path / "t.stl"
    p.write_bytes(bytes(80) + np.uint32(2).tobytes() + data.tobytes())
    lm = stl_landmarks(load_stl(p))
    assert lm[0][1] == -4 and lm[1][0] == -3 and lm[2][0] == 10 and lm[3][1] == 9


def test_pivot_recovers_tip():
    rng = np.random.default_rng(2)
    tip, pivot = np.array([0.0, -70, -1]), np.array([20.0, 10, 400])
    poses = []
    for _ in range(60):
        R = cv2.Rodrigues(rng.normal(0, 0.4, 3))[0]
        poses.append((R, pivot - R @ tip + rng.normal(0, 0.05, 3)))
    got_tip, got_pivot, rms = solve_pivot(poses)
    assert np.linalg.norm(got_tip - tip) < 0.5 and np.linalg.norm(got_pivot - pivot) < 0.5 and rms < 0.2


def test_tool_marker_is_detected():
    marker = cv2.aruco.generateImageMarker(TOOL_DICT, TOOL_ID, 300, borderBits=1)
    img = np.full((1000, 1400), 255, np.uint8)
    img[350:650, 550:850] = marker
    pose = tool_pose(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), CAM)
    assert pose is not None and pose[1][2] > 0
