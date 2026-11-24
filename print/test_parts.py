import numpy as np

from make_parts import STEP, grams, marker_holder, measure_mesh, socket, surface


def cube_vox():
    vox = np.zeros((60, 60, 60), bool)
    vox[20:40, 20:40, 10:50] = True
    return vox


def test_socket_has_floor_open_top_and_walls():
    s, offset = socket(cube_vox(), depth=8, floor=3, margin=5, gap=1)
    assert np.allclose(offset, [(20 - 1 - 10) * STEP, (20 - 1 - 10) * STEP, (10 - 6) * STEP])
    floor = int(3 / STEP)
    c = s.shape[0] // 2
    assert s[c, c, :floor].all()
    assert not s[c, c, floor:].any()
    assert s[0, c, floor:].all()


def test_cube_volume_and_area():
    vox = np.zeros((40, 40, 40), bool)
    vox[10:30, 10:30, 10:30] = True
    v, f = surface(vox, 0.5)
    area, volume = measure_mesh(v[f])
    assert abs(volume - 1000) < 60 and abs(area - 600) < 40
    assert 0 < grams(area, volume) < 1.24


def test_marker_holder_hole_is_open():
    vox, step = marker_holder()
    cx, cy = vox.shape[0] // 2, vox.shape[1] // 2
    assert not vox[cx, cy, : int(20 / step)].any()
    assert vox[cx, cy, int(30 / step)]
