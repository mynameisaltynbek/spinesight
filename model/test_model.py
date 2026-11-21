import cv2
import numpy as np
import torch

from net import IN_H, IN_W, STRIDE, Net, decode, errors, solve_pose, target

K = np.array([[500.0, 0, 320], [0, 500, 180], [0, 0, 1]])


def test_heatmap_roundtrip():
    uv = np.array([[123.4, 77.7], [500.2, 300.9], [10.0, 340.0]])
    got, conf = decode(target(uv, 3))
    assert np.abs(got - uv).max() < 0.05 and (conf > 0.8).all()


def test_net_output_shape():
    out = Net(10, pretrained=False)(torch.zeros(1, 3, IN_H, IN_W))
    assert out.shape == (1, 10, IN_H // STRIDE, IN_W // STRIDE)


def test_pose_from_exact_keypoints():
    kps = np.random.default_rng(1).uniform(-40, 40, (10, 3))
    Rg = cv2.Rodrigues(np.array([0.4, -0.2, 0.9]))[0]
    tg = np.array([10.0, -5, 400])
    uv, _ = cv2.projectPoints(kps, cv2.Rodrigues(Rg)[0], tg, K, np.zeros(5))
    R, t = solve_pose(uv.reshape(-1, 2), np.ones(10), kps, K, np.zeros(5))
    screws = [{"side": "left", "entry": [0, 0, 0], "tip": [0, 40, 0]}]
    e = errors(R, t, Rg, tg, kps, screws)
    assert e["rot_deg"] < 0.1 and e["trans_mm"] < 0.1 and e["left_entry_mm"] < 0.1 and e["left_angle_deg"] < 0.1


def test_pose_fails_with_few_keypoints():
    assert solve_pose(np.zeros((10, 2)), np.zeros(10), np.zeros((10, 3)), K, np.zeros(5)) is None
