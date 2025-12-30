__title__ = "h3-arcpy"
__author__ = "Joel McCune"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2023 by Joel McCune"

__all__ = ["get_h3_indices_for_esri_polygon", "get_k_neighbors"]

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, Optional, Literal

import arcpy
import h3
import h3.api.basic_int as h3_int

from ._logging import get_logger

# configure logging
logger = get_logger("h3_od.utils.h3_arcpy", level="DEBUG", add_stream_handler=False)


def get_h3_resolution(h3_index: Union[str, int]) -> int:
    """Get the H3 resolution for a given H3 index."""
    # get the resolution using the correct method based on the input index format
    if isinstance(h3_index, int):
        res = h3_int.get_resolution(h3_index)
    else:
        res = h3.get_resolution(h3_index)

    return res

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
                msg = f"First input_feature for UID must be either a string or integer."
                logger.error(msg)
                raise ValueError(msg)

            elif not isinstance(geom, arcpy.Polygon):
                msg = f"Second input_feature for Geometry must be an arcpy.Polygon."
                logger.error(msg)
                raise ValueError(msg)

        # if not a tuple, validate the polygon
        elif not isinstance(input_feature, arcpy.Polygon):
            msg = f"The input_feature for Geometry must be an arcpy.Polygon."
            logger.error(msg)
            raise ValueError(msg)

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


def transpose_coordinate_list(
    coord_list: list[tuple[float]],
) -> list[tuple[float, float]]:
    """
    Since the `h3` library expects coordinate pairs as `y, x`, use this function to transpose coordinate lists.

    !! note

        This is useful for converting lists of ``x, y`` coordinates to ``y, x`` for input into H3 functions, but
        also is useful for converting outputs from H3 functions from ``y, x`` to ``x,  y`` for use with ArcPy and
        ArcGIS as well.

    Args:
        coord_list: List of coordinates to be transposed.

    Returns:
        List of coordinates transposed.
    """

    # use a list comprehension to transpose the coordinate pairs
    transposed_lst: list[tuple[float, float]] = [
        (coord_02, coord_01) for coord_01, coord_02 in coord_list
    ]

    return transposed_lst


def get_h3_polygon_from_geojson(geojson: dict) -> h3.LatLngPoly:
    """
    Get an ``h3.Polygon`` object instance from GeoJSON.

    !! note

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
    Get an `h3.Polygon` object instance from an `arcpy.Polygon`.

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
    input_feature: Union[arcpy.Polygon, tuple[Union[int, str], arcpy.Polygon]],
    h3_resolution: int,
    contain: Literal["center", "full", "overlap", "bbox_overlap"] = "overlap",
) -> set[str]:
    """
    Wrap ``h3.polygon_to_cells`` to handle converting ArcPy Polygon geometry to H3 Polygon geometry to
        get the H3 indices for the input geometry.

    Args:
        input_feature: Polygon with area to get H3 indices for.
        h3_resolution: H3 resolution to retrieve indices for.
        contain: How the output indices' geometry is evaluated against the input area of interest geometry.
          This value is passed directly to the ``h3.h3shape_to_cells_experimental`` function. Available options
          include the following.
          - ``center`` Cell center is contained in the shape
          - ``full`` Cell is fully contained in the shape
          - ``overlap`` Cell overlaps the shape at any point (default)
          - ``bbox_overlap`` Cell bounding box overlaps shape

    Returns:

    """
    # ensure contain is lowercase
    contain = contain.lower()

    # ensure contain is valid
    valid_contain_opts = ["center", "full", "overlap", "bbox_overlap"]

    if contain not in valid_contain_opts:
        msg = f"Invalid contain option '{contain}'. Valid options are: {valid_contain_opts}"
        logger.error(msg)
        raise ValueError(msg)

    # convert the Esri Polygon to an H3 Polygon
    h3_poly = get_h3_polygon_from_esri_polygon(input_feature)

    # get the H3 index for the geometry including all cells touching the geometry
    idx = h3.h3shape_to_cells_experimental(h3_poly, res=h3_resolution, contain=contain)

    return idx


def get_h3_indices_for_esri_polygon(
        geom: arcpy.Polygon,
        resolution: int,
        contain: Literal["center", "full", "overlap", "bbox_overlap"] = "overlap",
) -> set[str]:
    """
    Get a non-repeating Python set of H3 indices for an ``arcpy.Polygon``.

    !! note

        Multipart polygons *are not supported*.

    Args:
        geom: A single ArcPy Polygon geometry object.
        resolution: H3 resolution to retrieve indices for.
        contain: How the output indices' geometry is evaluated against the input area of interest geometry.
          This value is passed directly to the ``h3.h3shape_to_cells_experimental`` function. Available options
          include the following.
          - ``center`` Cell center is contained in the shape
          - ``full`` Cell is fully contained in the shape
          - ``overlap`` Cell overlaps the shape at any point (default)
          - ``bbox_overlap`` Cell bounding box overlaps shape

    Returns:
        Set of unique H3 indices.
    """
    # ensure contain is lowercase
    contain = contain.lower()

    # ensure contain is valid
    valid_contain_opts = ["center", "full", "overlap", "bbox_overlap"]

    if contain not in valid_contain_opts:
        msg = f"Invalid contain option '{contain}'. Valid options are: {valid_contain_opts}"
        logger.error(msg)
        raise ValueError(msg)

    # start by getting an H3 Polygon from the ArcPy Polygon
    h3_poly = get_h3_polygon_from_esri_polygon(geom)

    # get the H3 index for the geometry including all cells touching the geometry
    h3_idx_set = h3.h3shape_to_cells_experimental(h3_poly, res=resolution, contain=contain)

    return h3_idx_set


def get_arcpy_polygon_for_h3_index(h3_index: Union[str, int]) -> arcpy.Polygon:
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


def get_arcpy_point_for_h3_index(h3_index: Union[str, int]) -> arcpy.PointGeometry:
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
    origin_indices: list[Union[str, int]], k_dist: int, max_workers: int = None
) -> list[Union[int, str]]:
    """Get non-repeating K-distance neighbors for multiple origin H3 indices.
    
    Args:
        origin_indices: List of H3 indices to get neighbors for.
        k_dist: K-ring distance for neighbors.
        max_workers: Maximum number of threads to use. If None, defaults to min(32, cpu_count() + 4).
    
    Returns:
        List of unique destination H3 indices including all k-neighbors.
    """
    # ensure the origin indices are all a single type
    idx_types = set(type(idx) for idx in origin_indices)
    if len(idx_types) > 1:
        msg = f"All origin_indices must be of the same type (either all str or all int). Found types: {idx_types}"
        logger.error(msg)
        raise ValueError(msg)

    # use a set to collect all unique neighbors
    all_neighbors = set()
    
    # use ThreadPoolExecutor to parallelize the k_neighbors lookup across multiple threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # submit all tasks
        future_to_idx = {
            executor.submit(get_single_origin_k_neighbors, idx, k_dist): idx 
            for idx in origin_indices
        }
        
        # collect results as they complete
        for future in as_completed(future_to_idx):
            try:
                neighbors = future.result()
                all_neighbors.update(neighbors)
            except Exception as e:
                idx = future_to_idx[future]
                logger.error(f"Error getting neighbors for index {idx}: {e}")
                raise
    
    # convert set to list
    all_neighbors_lst = list(all_neighbors)

    return all_neighbors_lst
