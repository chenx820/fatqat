"""Immutable public arrangements for neutral-atom emulators."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, ClassVar


def _dimension(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _spacing(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("spacing must be a positive finite number")
    result = float(value)
    if not isfinite(result) or result <= 0:
        raise ValueError("spacing must be a positive finite number")
    return result


def _ordered_tuple(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, Mapping, Set)):
        raise ValueError(f"{name} must be an ordered iterable")
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an ordered iterable") from exc


def _coordinate(value: Any, name: str) -> tuple[float, ...]:
    components = _ordered_tuple(value, name)
    if len(components) not in (2, 3):
        raise ValueError(f"{name} must contain exactly two or three components")
    converted = []
    for index, component in enumerate(components):
        error = f"{name}[{index}] must be a finite real number, not bool"
        if isinstance(component, bool) or not isinstance(component, Real):
            raise ValueError(error)
        try:
            number = float(component)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(error) from exc
        if not isfinite(number):
            raise ValueError(error)
        converted.append(number)
    return tuple(converted)


@dataclass(frozen=True, init=False)
class AtomArrangement:
    """An immutable arrangement of physical atom sites in micrometres.

    Use ``from_coordinates()`` for explicit two- or three-dimensional sites,
    or ``chain()`` and ``rectangular()`` for regular layouts. Coordinates are
    always stored as immutable ``(x, y, z)`` tuples. Explicit coordinates keep
    their input order; regular layouts use row-major order with x advancing
    across columns. Coordinate order defines the site indices.

    Attributes:
        rows: Positive number of rows for a regular layout, otherwise None.
        cols: Positive number of columns for a regular layout, otherwise None.
        spacing: Uniform horizontal/vertical nearest-neighbor spacing in
            micrometres for a regular layout, otherwise None.
        coordinates: Immutable ``(x, y, z)`` coordinates in micrometres.
            Two-dimensional input has ``z == 0.0``.
        spatial_dimension: Coordinate-space dimension, 2 or 3, determined by
            the number of input components. Explicit z values, including
            zero, give 3. Regular layouts use 2, including chains embedded
            in the xy plane. This is independent of the atom's energy levels.

    Examples:
        >>> import fatqat as fq
        >>> arrangement = fq.emulator.AtomArrangement.chain(3, spacing=6.0)
        >>> arrangement.num_sites
        3
        >>> arrangement.coordinates[2]
        (12.0, 0.0, 0.0)
    """

    rows: int | None
    cols: int | None
    spacing: float | None
    coordinates: tuple[tuple[float, float, float], ...]
    spatial_dimension: int
    distance_unit: ClassVar[str] = "um"

    @classmethod
    def from_coordinates(cls, coordinates: Iterable[Iterable[Real]]) -> AtomArrangement:
        """Create a fixed layout from ordered two- or three-dimensional sites.

        Args:
            coordinates: Nonempty ordered iterable of coordinate iterables
                in micrometres. Every site must have exactly two components
                ``(x, y)`` or every site exactly three ``(x, y, z)``. Components
                must be finite real numbers, excluding booleans; negative
                values are allowed. Positions must be distinct after float
                conversion. Strings, mappings, and sets are not accepted as
                the outer iterable or as individual coordinates.

        Returns:
            AtomArrangement: An immutable copy of the coordinates in input
                order, with zero z values added to two-dimensional input.
                ``spatial_dimension`` records the input dimension, including
                3 when all explicit z values are zero. ``rows``, ``cols``, and
                ``spacing`` are None, even if the sites form a regular grid.

        Raises:
            ValueError: If the input is empty, unordered, has invalid or mixed
                dimensions, contains invalid components, or repeats a position.

        Examples:
            >>> import fatqat as fq
            >>> sites = fq.emulator.AtomArrangement.from_coordinates(
            ...     [(3, 5), (0, 0)]
            ... )
            >>> sites.coordinates
            ((3.0, 5.0, 0.0), (0.0, 0.0, 0.0))
            >>> sites.spatial_dimension
            2
        """
        points = _ordered_tuple(coordinates, "coordinates")
        if not points:
            raise ValueError("coordinates must contain at least one site")
        normalized = []
        spatial_dimension = None
        for index, point in enumerate(points):
            values = _coordinate(point, f"coordinates[{index}]")
            if spatial_dimension is None:
                spatial_dimension = len(values)
            elif len(values) != spatial_dimension:
                raise ValueError(
                    f"coordinates[{index}] must have {spatial_dimension} components "
                    "like every other site"
                )
            normalized.append(
                (values[0], values[1], values[2] if len(values) == 3 else 0.0)
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError("coordinates must contain distinct positions")

        instance = object.__new__(cls)
        object.__setattr__(instance, "rows", None)
        object.__setattr__(instance, "cols", None)
        object.__setattr__(instance, "spacing", None)
        object.__setattr__(instance, "coordinates", tuple(normalized))
        object.__setattr__(instance, "spatial_dimension", spatial_dimension)
        return instance

    @classmethod
    def chain(cls, num_sites: int, spacing: float) -> AtomArrangement:
        """Create a one-dimensional chain ordered along the x axis.

        Args:
            num_sites: Positive number of physical sites.
            spacing: Positive finite nearest-neighbor spacing in micrometres.

        Returns:
            An immutable one-row arrangement with ``num_sites`` coordinates.

        Raises:
            ValueError: If ``num_sites`` is not a positive integer or spacing
                is not a positive finite real number. Booleans are rejected.
        """
        num_sites = _dimension(num_sites, "num_sites")
        return cls.rectangular(rows=1, cols=num_sites, spacing=spacing)

    @classmethod
    def rectangular(cls, rows: int, cols: int, spacing: float) -> AtomArrangement:
        """Create a row-major rectangular arrangement with zero z coordinates.

        Args:
            rows: Positive number of rows.
            cols: Positive number of columns.
            spacing: Positive finite nearest-neighbor spacing in micrometres
                for the current neutral-atom model families.

        Returns:
            An immutable arrangement with ``rows * cols`` declared sites and
            coordinates ``(column * spacing, row * spacing, 0)``.

        Raises:
            ValueError: If a dimension is not a positive integer or spacing
                is not a positive finite real number. Booleans are rejected.
        """
        rows = _dimension(rows, "rows")
        cols = _dimension(cols, "cols")
        spacing = _spacing(spacing)
        coordinates = tuple(
            (column * spacing, row * spacing, 0.0)
            for row in range(rows)
            for column in range(cols)
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "rows", rows)
        object.__setattr__(instance, "cols", cols)
        object.__setattr__(instance, "spacing", spacing)
        object.__setattr__(instance, "coordinates", coordinates)
        object.__setattr__(instance, "spatial_dimension", 2)
        return instance

    @property
    def num_sites(self) -> int:
        """Return the exact number of declared physical sites.

        Returns:
            The number of coordinates in this immutable geometry. This is a
            site count, not a dynamic atom-occupancy count.
        """
        return len(self.coordinates)

    def __len__(self) -> int:
        """Return the number of sites.

        Returns:
            The same site count as ``num_sites``.
        """
        return self.num_sites
