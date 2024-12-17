from typing import Union

from arcgis.geometry import Polygon, Point
from h3.api import memview_int as h3_int

__all__ = [
    "get_arcpy_point_for_h3_index",
    "get_arcgis_polygon_for_h3_index",
]


def preprocess_h3_index(h3_index: Union[str, int]) -> int:
    """Helper function to preprocess the h3 index to integer representation."""
    # if the input value the H3 numeric value as a string, convert to integer
    if isinstance(h3_index, str):
        if h3_index.isnumeric():
            h3_index = int(h3_index)

    # if hex string, convert to int
    if isinstance(h3_index, str):
        h3_index = h3_int.str_to_int(h3_index)

    return h3_index


def get_arcgis_polygon_for_h3_index(h3_index: Union[str, int]) -> Polygon:
    """
    For a single H3 index, get the ArcGIS Polygon geometry for the index.

    Args:
        h3_index: H3 index.

    Returns:
        ArcGIS Polygon geometry for the index.
    """
    h3_index = preprocess_h3_index(h3_index)

    # get the coordinates for the index
    coord_lst = h3_int.cell_to_boundary(h3_index)

    # create an ArcPy geometry object for the index
    geom = Polygon(
        {
            "rings": [[[coords[1], coords[0]] for coords in coord_lst]],
            "spatialReference": {"wkid": 4326},
        }
    )

    return geom


def get_arcpy_point_for_h3_index(h3_index: Union[str, int]) -> Point:
    """
    For a single H3 index, get the ArcGIS Point geometry for the index.

    Args:
        h3_index: H3 index.

    Returns:
        ArcPy point geometry for the index.
    """
    h3_index = preprocess_h3_index(h3_index)

    # get the coordinates for the index
    coords = h3_int.cell_to_latlng(h3_index)

    # create an ArcPy geometry object for the index
    geom = Point({"x": coords[1], "y": coords[0], "spatialReference": {"wkid": 4326}})

    return geom
