import numpy as np

from spine import cylinder, plan, write_stl


def phantom(pedicle_half_width):
    m = np.zeros((80, 80, 40), bool)
    x, y = np.mgrid[:80, :80]
    m[(x - 40) ** 2 + (y - 55) ** 2 <= 18 ** 2, 5:35] = True
    w = pedicle_half_width
    m[30 - w:31 + w, 22:45, 14:27] = True
    m[50 - w:51 + w, 22:45, 14:27] = True
    m[22:59, 14:22, 10:31] = True
    m[38:43, 2:14, 14:27] = True
    return m


def test_finds_pedicle_centers():
    res = plan(phantom(4), np.ones(3))
    left, right = res["screws"]
    assert abs(left["isthmus"][0] - 30) < 1 and abs(left["isthmus"][2] - 20) < 1
    assert abs(right["isthmus"][0] - 50) < 1 and abs(right["isthmus"][2] - 20) < 1
    assert left["entry"][1] < left["isthmus"][1] < left["tip"][1]
    assert left["tip"][1] > 55
    assert abs(left["medial_angle_deg"]) <= 4 and not left["breach"]


def test_thin_pedicle_is_breach():
    res = plan(phantom(1), np.ones(3))
    assert all(s["breach"] for s in res["screws"])


def test_stl_size(tmp_path):
    v, f = cylinder(np.zeros(3), np.array([0, 0, 10.0]), 2)
    write_stl(tmp_path / "c.stl", [(v, f)])
    assert (tmp_path / "c.stl").stat().st_size == 84 + 50 * len(f)
