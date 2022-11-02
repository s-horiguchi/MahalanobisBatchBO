import pytest
import torch
from problem import DesignMethod, get_design


@pytest.fixture
def bounds():
    return [(-5.6, +2.8), (-1.9, -0.2), (+10, +10.2)]


@pytest.fixture
def bounds_invalid():
    return [(+5.6, +2.8), (-1.9, -0.2), (+10, +10.2)]


@pytest.fixture
def bounds_degenerate():
    return [(+2.8, +2.8), (-1.9, -0.2), (+10, +10.2)]


@pytest.mark.parametrize("method", list(DesignMethod))
def test_get_design_valid(bounds, method):
    num_points = 10000
    bounds_min = torch.Tensor(bounds)[:, 0]
    bounds_max = torch.Tensor(bounds)[:, 1]

    points = get_design(bounds, num_points, method)
    assert (
        (points >= bounds_min) & (points <= bounds_max)
    ).all()


@pytest.mark.parametrize("method", list(DesignMethod))
def test_get_design_invalid(bounds_invalid, method):
    num_points = 10000

    with pytest.raises(ValueError):
        get_design(bounds_invalid, num_points, method)


@pytest.mark.parametrize("method", list(DesignMethod))
def test_get_design_degenerate(bounds_degenerate, method):
    num_points = 10000
    bounds_min = torch.Tensor(bounds_degenerate)[:, 0]
    bounds_max = torch.Tensor(bounds_degenerate)[:, 1]

    points = get_design(bounds_degenerate, num_points, method)
    assert (
        (points >= bounds_min) & (points <= bounds_max)
    ).all()
