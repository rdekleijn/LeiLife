import pytest

from alife.geometry import calc_dir_and_dist, gen_rand_location, gen_rand_orientation


@pytest.mark.parametrize("dx,dy,orientation,expected_deg", [
    (0, 1, 0, 0),      # straight ahead (north) at orientation 0
    (1, 0, 0, 90),     # east, relative to orientation 0
    (0, 1, 90, 270),   # straight ahead, but agent is now facing east
])
def test_calc_dir_and_dist_bearing(dx, dy, orientation, expected_deg):
    deg, dist = calc_dir_and_dist(dx, dy, orientation)
    assert deg == pytest.approx(expected_deg)
    assert dist == pytest.approx((dx ** 2 + dy ** 2) ** 0.5)


def test_gen_rand_location_within_bounds():
    for _ in range(50):
        loc = gen_rand_location(10)
        assert 0 <= loc[0] <= 10
        assert 0 <= loc[1] <= 10


def test_gen_rand_orientation_within_bounds():
    for _ in range(50):
        assert 0 <= gen_rand_orientation() <= 360
