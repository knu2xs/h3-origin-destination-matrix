__title__ = "h3-arcpy"
__author__ = "Joel McCune"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2023 by Joel McCune"

__all__ = ["get_h3_indices_for_esri_polygon", "get_k_neighbors"]

from typing import List, Union, Tuple, Set, Iterable

import arcpy
import h3
import h3.api.basic_int as h3_int
import dask.dataframe as dd
import numpy as np


def handle_features(fn):
    """Decorator to take care of input_features validation and handling possibility of including a UID in a tuple."""

    def hndl_feat(*args, **kwargs):
        # get the input feature and update either args or kwargs
        if len(args) > 0:
            input_feature = args[0]
            args = args[1:]
        else:
            input_feature = kwargs.pop("input_feature")

        # if it's a tuple, get the parts and validate
        if isinstance(input_feature, tuple):
            uid, geom = input_feature

            if not isinstance(uid, (str, int)):
                raise ValueError(
                    f"First input_feature for UID must be either a string or integer."
                )

            elif not isinstance(geom, arcpy.Polygon):
                raise ValueError(
                    f"Second input_feature for Geometry must be an arcpy.Polygon."
                )

        # if not a tuple, validate the polygon
        elif not isinstance(input_feature, arcpy.Polygon):
            raise ValueError(f"The input_feature MUST be an arcpy.Polygon")

        # if a single feature with a polygon, save to geometry
        else:
            geom = input_feature

        # invoke the function
        res = fn(geom, *args, **kwargs)

        # reassemble results
        if isinstance(input_feature, tuple):
            ret_val = (uid, res)
        else:
            ret_val = res

        return ret_val

    return hndl_feat


def transpose_coordinate_list(coord_list: List[Tuple[float]]) -> List[Tuple[float]]:
    """
    Since the ``h3`` library expects coordinate pairs as ``y, x``, use this function to transpose coordinate lists.

    .. note::
        This is useful for converting lists of ``x, y`` coordinates to ``y, x`` for input into H3 functions, but
        also is useful for converting outputs from H3 functions from ``y, x`` to ``x,  y`` for use with ArcPy and
        ArcGIS as well.

    Args:
        coord_list: List of coordinates to be transposed.

    Returns:
        List of coordinates transposed.
    """

    # use a list comprehension to transpose the coordinate pairs.
    transposed_lst = [(coord_02, coord_01) for coord_01, coord_02 in coord_list]

    return transposed_lst


def get_h3_polygon_from_geojson(geojson: dict) -> h3.LatLngPoly:
    """
    Get an ``h3.Polygon`` object instance from GeoJSON.

    ..note ::
        The GeoJSON *must* be a single part polygon. Multipart polygons are *not supported.*.

    Args:
        geojson: Geometry formatted as GeoJSON

    Returns:
        An H3 Polygon object instance.
    """

    # extract the coordinates from the geojson
    feat_json = geojson.get("coordinates")[0]

    # get the outer polygon loop
    poly_coords = feat_json[0]

    # transpose the x,y coordinate pairs to y,x pairs
    poly_coords = transpose_coordinate_list(poly_coords)

    # use the hole coordinates if present
    if len(feat_json) == 2:
        # prep the hole list by transposing the x,y coordinate pairs to y,x
        hole_lst = [
            transpose_coordinate_list(hole_coords) for hole_coords in feat_json[1]
        ]

        # create an H3 Polygon object with the holes
        h3_poly = h3.LatLngPoly(poly_coords, hole_lst)

    else:
        # create an H3 Polygon object without the holes
        h3_poly = h3.LatLngPoly(poly_coords)

    return h3_poly


def get_h3_polygon_from_esri_polygon(polygon: arcpy.Polygon) -> h3.LatLngPoly:
    """
    Get an ``h3.Polygon`` object instance from an ``arcpy.Polygon``.

    Args:
        polygon: An ArcPy Polygon object.

    Returns:
        An H3 Polygon object instance.
    """

    # extract the GeoJSON using the geo interface from the geometry object and use it to get an H3 Polygon
    h3_poly = get_h3_polygon_from_geojson(polygon.__geo_interface__)

    return h3_poly


@handle_features
def polygon_to_h3_indices(
    input_feature: Union[arcpy.Polygon, Tuple[Union[int, str], arcpy.Polygon]],
    h3_resolution: int,
) -> Set[str]:
    """
    Wrap ``h3.polygon_to_cells`` to handle converting ArcPy Polygon geometry to H3 Polygon geometry to
        get the H3 indices for the input geometry.

    Args:
        input_feature: Polygon with area to get H3 indices for.
        h3_resolution: H3 resolution to retrieve indices for.

    Returns:

    """
    # convert the Esri Polygon to an H3 Polygon
    h3_poly = get_h3_polygon_from_esri_polygon(input_feature)

    # get the H3 index for the geometry
    idx = h3.h3shape_to_cells(h3_poly, res=h3_resolution)

    return idx


def get_h3_indices_for_esri_polygon(geom: arcpy.Polygon, resolution: int) -> Set[str]:
    """
    Get a non-repeating Python set of H3 indices for an ``arcpy.Polygon``.

    .. note::
        Multipart polygons *are not supported*.

    Args:
        geom: A single ArcPy Polygon geometry object.
        resolution: H3 resolution to retrieve indices for.

    Returns:
        Set of unique H3 indices.
    """

    # start by getting an H3 Polygon from the ArcPy Polygon
    h3_poly = get_h3_polygon_from_esri_polygon(geom)

    # use the H3 Polygon to get a set of unique indices
    h3_idx_set = h3.h3shape_to_cells(h3_poly, res=resolution)

    return h3_idx_set


def get_esri_polygon_for_h3_index(h3_index: Union[str, int]) -> arcpy.Polygon:
    """
    For a single H3 index, get the ArcPy polygon geometry for the index.

    Args:
        h3_index: H3 index.

    Returns:
        ArcPy polygon geometry for the index.
    """
    # if the input value the H3 numeric value as a string, convert to integer
    if isinstance(h3_index, str):
        if h3_index.isnumeric():
            h3_index = int(h3_index)

    # convert the index by using the correct method based on the input index format
    if isinstance(h3_index, int):
        coord_lst = h3_int.cell_to_boundary(h3_index)
    else:
        coord_lst = h3.cell_to_boundary(h3_index)

    # create an ArcPy geometry object for the index
    geom = arcpy.Polygon(
        inputs=arcpy.Array([arcpy.Point(coords[1], coords[0]) for coords in coord_lst]),
        spatial_reference=arcpy.SpatialReference(4326),
    )

    return geom


def get_esri_point_for_h3_index(h3_index: Union[str, int]) -> arcpy.PointGeometry:
    """
    For a single H3 index, get the ArcPy point geometry for the index.

    Args:
        h3_index: H3 index.

    Returns:
        ArcPy point geometry for the index.
    """
    # if the input value the H3 numeric value as a string, convert to integer
    if isinstance(h3_index, str):
        if h3_index.isnumeric():
            h3_index = int(h3_index)

    # convert the index by using the correct method based on the input index format
    if isinstance(h3_index, int):
        coords = h3_int.cell_to_latlng(h3_index)
    else:
        coords = h3.cell_to_latlng(h3_index)

    # convert to the coordinates to a point geometry
    geom = arcpy.PointGeometry(
        inputs=arcpy.Point(coords[1], coords[0]),
        spatial_reference=arcpy.SpatialReference(4326),
    )

    return geom


def get_single_origin_k_neighbors(
    origin_index: Union[str, int], k_dist: int
) -> list[Union[int, str]]:
    """Get K-distance neighbors for a single H3 index."""
    if isinstance(origin_index, str):
        k_neighbors = h3.grid_disk(origin_index, k_dist)
    else:
        k_neighbors = h3_int.grid_disk(origin_index, k_dist)

    return k_neighbors


def get_k_neighbors(
    origin_indices: list[str, int], k_dist: int
) -> list[Union[int, str]]:
    """Get non-repeating K-distance neighbors for multiple origin H3 indices."""
    # create a dask dataframe to parallelize processing
    df = dd.from_array(np.array(origin_indices), columns=["origin_idx"])

    # TODO: lookup number of threads available and distribute across cores

    # use dask to distribut the k_neighbors lookup across multiple threads and speed up this bottleneck
    dest_idx_lst = (
        df["origin_idx"]
        .apply(
            lambda idx: get_single_origin_k_neighbors(idx, 5), meta=("origin_idx", str)
        )
        .explode()  # get every value in single row
        .drop_duplicates()  # remove duplicate values
        .compute()  # invoke parallel processing using Dask
        .values.tolist()  # convert from Dask series to a list
    )

    return dest_idx_lst
