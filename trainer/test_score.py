import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from score import grade, score
from spine import plan
from test_spine import phantom


def setup():
    m, sp = phantom(4), np.ones(3)
    return m, sp, ndimage.distance_transform_edt(m, sampling=sp), plan(m, sp)["screws"]


def test_grades():
    assert [grade(b) for b in (0, 1.9, 2, 3.9, 5, 6, 9)] == ["A", "B", "C", "C", "D", "E", "E"]


def test_tool_on_plan_is_grade_a():
    m, sp, dist, screws = setup()
    s = screws[0]
    e, t = np.array(s["entry"]), np.array(s["tip"])
    r = score(e, t - e, screws, m, sp, dist)
    assert r["side"] == s["side"] and r["entry_mm"] < 0.01 and r["angle_deg"] < 0.1 and r["grade"] == "A"


def test_tilted_tool_breaches():
    m, sp, dist, screws = setup()
    s = screws[0]
    e = np.array(s["entry"])
    d = np.array([np.sin(np.radians(30)), np.cos(np.radians(30)), 0])
    r = score(e, d, screws, m, sp, dist)
    assert r["angle_deg"] > 20 and r["grade"] != "A"


def test_breach_reports_wall_and_depth():
    m, sp, dist, screws = setup()
    s = screws[0]
    e, t = np.array(s["entry"]), np.array(s["tip"])
    u = (t - e) / np.linalg.norm(t - e)
    sign = 1 if s["side"] == "left" else -1
    for push, wall in (([sign, 0, 0], "medial"), ([-sign, 0, 0], "lateral")):
        r = score(e, u + 0.7 * np.array(push, float), screws, m, sp, dist)
        assert r["grade"] != "A" and r["wall"] == wall and r["depth_mm"] >= 0, (wall, r)


def test_tool_outside_bone_has_no_grade():
    m, sp, dist, screws = setup()
    r = score(np.array([-50.0, -50, -50]), np.array([0, 0, -1.0]), screws, m, sp, dist)
    assert r["grade"] == "-" and r["breach_mm"] is None


def test_tool_pose_maps_to_planned_screw():
    import cv2
    from score import tool_in_vertebra
    m, sp, dist, screws = setup()
    s = screws[0]
    e, t = np.array(s["entry"]), np.array(s["tip"])
    u = (t - e) / np.linalg.norm(t - e)
    Rv, tv = cv2.Rodrigues(np.array([0.3, -0.2, 0.5]))[0], np.array([5.0, -3, 300])
    tip_m = np.array([0.0, -70, -1])
    y = -u
    x = np.cross(y, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    Rm_v = np.column_stack([x, y, np.cross(x, y)])
    Rm = Rv @ Rm_v
    tm = Rv @ e + tv - Rm @ tip_m
    tip_v, d_v = tool_in_vertebra((Rm, tm), tip_m, (Rv, tv))
    assert np.allclose(tip_v, e, atol=1e-6)
    assert np.dot(d_v, u) > 0.999
    assert score(tip_v, d_v, screws, m, sp, dist)["grade"] == "A"
