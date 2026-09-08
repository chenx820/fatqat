"""Public neutral-atom arrangement value tests."""

from math import inf, nan
from operator import setitem

import numpy as np
import pytest

from fatqat.emulator import AtomArrangement


def test_rectangular_arrangement_is_row_major_3d_immutable_value():
    arrangement = AtomArrangement.rectangular(rows=2, cols=3, spacing=1.5)

    assert arrangement.coordinates == (
        (0.0, 0.0, 0.0),
        (1.5, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (0.0, 1.5, 0.0),
        (1.5, 1.5, 0.0),
        (3.0, 1.5, 0.0),
    )
    assert (
        arrangement.num_sites == len(arrangement) == len(arrangement.coordinates) == 6
    )
    assert arrangement.distance_unit == "um"
    assert (arrangement.rows, arrangement.cols, arrangement.spacing) == (2, 3, 1.5)
    assert arrangement.spatial_dimension == 2
    assert not hasattr(arrangement, "cardinality")
    assert not hasattr(arrangement, "occupancy")
    assert arrangement == AtomArrangement.rectangular(2, 3, 1.5)
    assert hash(arrangement) == hash(AtomArrangement.rectangular(2, 3, 1.5))

    with pytest.raises((AttributeError, TypeError)):
        setitem(arrangement.coordinates, 0, (99.0, 0.0, 0.0))
    with pytest.raises((AttributeError, TypeError)):
        arrangement.rows = 99


def test_chain_arrangement_is_the_one_row_convenience_geometry():
    arrangement = AtomArrangement.chain(num_sites=3, spacing=1.5)

    assert arrangement == AtomArrangement.rectangular(1, 3, 1.5)
    assert arrangement.spatial_dimension == 2
    assert arrangement.coordinates == (
        (0.0, 0.0, 0.0),
        (1.5, 0.0, 0.0),
        (3.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError):
        AtomArrangement.chain(num_sites=0, spacing=1.5)


@pytest.mark.parametrize(
    "rows, cols", [(True, 1), (1, False), (1.0, 1), (1, 2.5), (0, 1), (1, -1)]
)
def test_rectangular_arrangement_rejects_invalid_dimensions(rows, cols):
    with pytest.raises(ValueError):
        AtomArrangement.rectangular(rows, cols, 1.0)


@pytest.mark.parametrize("spacing", [True, 0, -1.0, inf, -inf, nan])
def test_rectangular_arrangement_rejects_invalid_spacing(spacing):
    with pytest.raises(ValueError):
        AtomArrangement.rectangular(1, 1, spacing)


@pytest.mark.parametrize(
    "coordinates, dimension, expected",
    [
        ([(3, -5), (0, 0)], 2, ((3.0, -5.0, 0.0), (0.0, 0.0, 0.0))),
        ([(3, -5, 0), (0, 0, 0)], 3, ((3.0, -5.0, 0.0), (0.0, 0.0, 0.0))),
        ([(3, -5, 4), (0, 0, -2)], 3, ((3.0, -5.0, 4.0), (0.0, 0.0, -2.0))),
    ],
)
def test_explicit_coordinates_preserve_order_and_input_dimension(
    coordinates, dimension, expected
):
    arrangement = AtomArrangement.from_coordinates(coordinates)

    assert arrangement.coordinates == expected
    assert arrangement.spatial_dimension == dimension
    assert arrangement.num_sites == len(arrangement) == len(coordinates)
    assert arrangement.distance_unit == "um"
    assert (arrangement.rows, arrangement.cols, arrangement.spacing) == (
        None,
        None,
        None,
    )
    assert arrangement == AtomArrangement.from_coordinates(coordinates)
    assert hash(arrangement) == hash(AtomArrangement.from_coordinates(coordinates))


@pytest.mark.parametrize("container", [list, tuple, iter, np.asarray])
def test_explicit_coordinates_accept_ordered_iterables(container):
    arrangement = AtomArrangement.from_coordinates(container([(6, 0), (0, 0)]))

    assert arrangement.coordinates == ((6.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert all(
        isinstance(value, float) for point in arrangement.coordinates for value in point
    )


def test_explicit_coordinates_copy_input_and_remain_immutable():
    coordinates = [[0, 0, 6]]
    arrangement = AtomArrangement.from_coordinates(coordinates)
    coordinates[0][2] = 9
    coordinates.append([0, 0, 12])

    assert arrangement.coordinates == ((0.0, 0.0, 6.0),)
    assert arrangement.num_sites == 1
    with pytest.raises((AttributeError, TypeError)):
        setitem(arrangement.coordinates, 0, (0.0, 0.0, 9.0))
    with pytest.raises((AttributeError, TypeError)):
        arrangement.spatial_dimension = 2


@pytest.mark.parametrize(
    "coordinates, message",
    [
        ([], "at least one site"),
        (None, "ordered iterable"),
        ("12", "ordered iterable"),
        ({(0, 0), (1, 1)}, "ordered iterable"),
        ({(0, 0): "site"}, "ordered iterable"),
        ([None], "ordered iterable"),
        (["12"], "ordered iterable"),
        ([{0, 1}], "ordered iterable"),
        ([{0: 1, 2: 3}], "ordered iterable"),
        ([()], "two or three components"),
        ([(0,)], "two or three components"),
        ([(0, 1, 2, 3)], "two or three components"),
        ([(0, 0), (1, 1, 1)], "must have 2 components"),
        ([(0, 0, 0), (1, 1)], "must have 3 components"),
        ([(0, 0), (0.0, -0.0)], "distinct positions"),
        ([(0, 0, 0), (0, 0, 0)], "distinct positions"),
        ([(2**54, 0), (2**54 + 1, 0)], "distinct positions"),
    ],
)
def test_explicit_coordinates_reject_invalid_layouts(coordinates, message):
    with pytest.raises(ValueError, match=message):
        AtomArrangement.from_coordinates(coordinates)


@pytest.mark.parametrize(
    "component", [True, np.bool_(False), None, "1", 1j, inf, -inf, nan, 10**400]
)
def test_explicit_coordinates_reject_invalid_components(component):
    with pytest.raises(ValueError, match=r"coordinates\[0\]\[2\].*finite real number"):
        AtomArrangement.from_coordinates([(0, 0, component)])
