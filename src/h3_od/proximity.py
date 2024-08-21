#!/usr/bin/env python
# coding: utf-8

"""
Proximity streamlines the process of calculating distance metrics using Esri Network Analyst.
"""
__all__ = [
    "get_distance_between_h3_indices",
    "get_distance_between_coordinates_using_h3",
    "get_h3_origin_destination_distance_parquet",
    "get_h3_neighbors",
    "get_nearest_origin_destination_neighbor",
    "get_origin_destination_distance_parquet",
    "get_origin_destination_neighbors",
]

import itertools
import logging
import math
import os
import uuid
from pathlib import Path
from typing import List, Union, Tuple, Optional, Iterable
from warnings import warn

import arcgis.geometry
from arcgis.geoenrichment._business_analyst import Country
import arcpy
import h3
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq

from h3_od.utils import h3_arcpy

# basic setup to ensure network solves work
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("network")

# variables for network solves - some may have to be updated for doing international analysis
iso2 = "US"


def validate_h3_index_list(
    input_features: List[Union[str, int]]
) -> List[Tuple[str, arcpy.PointGeometry]]:
    # get the first item to interrogate
    for first_itm in input_features:
        # if integers, convert to hexadecimal string
        if isinstance(first_itm, int):
            input_features = [
                h3.int_to_str(h3_int_idx) for h3_int_idx in input_features
            ]

        # if integers buried in a big string, convert to hexadecimal string
        elif isinstance(first_itm, str):
            if first_itm.isdigit():
                input_features = [
                    h3.int_to_str(int(h3_int_idx)) for h3_int_idx in input_features
                ]

        break

    # iterate the input h3 indices and create a tuple of the h3 index and coordinates (admittedly reversed)
    input_features = [(h3_idx, h3.cell_to_latlng(h3_idx)) for h3_idx in input_features]

    # iterate the tuples, and replace the transposed coordinates with PointGeometry
    input_features = [
        (
            h3_idx,
            arcpy.PointGeometry(
                arcpy.Point(h3_coords[1], h3_coords[0]), spatial_reference=4326
            ),
        )
        for h3_idx, h3_coords in input_features
    ]

    return input_features


def validate_origin_destination_inputs(
    input_features: Union[
        List[Union[int, str]],
        Iterable[Tuple[Union[int, str], arcpy.Geometry]],
        arcpy._mp.Layer,
        pd.DataFrame,
        str,
        Path,
    ]
) -> Union[arcpy._mp.Layer, str, arcpy.FeatureSet]:
    # first check if the dataframe is spatially enabled AND point or polygon
    if isinstance(input_features, pd.DataFrame):
        if not input_features.spatial.validate():
            raise ValueError(
                "Input features' data frame does not appear to be validated. Please try "
                "df.spatial.set_geometry"
            )

        if (
            input_features.spatial.geometry_type != "Point"
            and input_features.spatial.geometry_type != "Polygon"
        ):
            raise ValueError(
                "Input features' geometry must be either Point or Polygon."
            )

    # if a list, can be just h3 indices, but also list of unique ids and geometries
    elif isinstance(input_features, Iterable):
        # get the first item to interrogate
        for first_itm in input_features:
            # if just a list of identifiers, process as H3 indices
            if isinstance(first_itm, (str, int)):
                input_features = validate_h3_index_list(input_features)

            # otherwise, make sure geometries are valid, and convert geometries to Python API geometries
            else:
                geom = first_itm[1]

                # ensure point is PointGeometry
                if isinstance(geom, arcpy.Point):
                    geom = arcpy.PointGeometry(geom)

                # ensure is correct geometry type
                if not isinstance(geom, [arcpy.PointGeometry, arcpy.Polygon]):
                    raise ValueError(
                        "Input features' geometry must be either PointGeometry or Polygon."
                    )

                # convert ArcPy geometries to Python API geometries
                input_features = [
                    (oid, arcgis.geometry.Geometry.from_arcpy(arcpy_geom))
                    for oid, arcpy_geom in input_features
                ]

            break

    # if a layer or path to data, use describe to check the geometry type
    else:
        if isinstance(input_features, Path):
            input_features = str(input_features)

        geom_typ = arcpy.Describe(input_features).shapeType
        if geom_typ != "Point" and geom_typ != "Polygon":
            raise ValueError(
                "Input features' geometry type must be either Point or Polygon."
            )

    # now, if a list, is tuples of oid and geometry...convert to spatially enabled data frame
    if isinstance(input_features, list):
        input_features = pd.DataFrame(input_features, columns=["oid", "SHAPE"])
        input_features.spatial.set_geometry("SHAPE")

    # finally, if spatially enabled data frame, convert to temporary feature class
    if isinstance(input_features, pd.DataFrame):
        # check and set the spatial reference if not already set for the features
        if input_features.spatial.sr["wkid"] is None:
            input_features.spatial.sr = arcgis.geometry.SpatialReference(4326)

        # converting SeDF to Feature Set and passing through Copy Features since much more reliable
        input_features = arcpy.management.CopyFeatures(
            arcpy.FeatureSet(input_features.spatial.to_featureset()),
            os.path.join(arcpy.env.scratchGDB, f"f_{uuid.uuid4().hex}"),
        )[0]

    return input_features


def get_origin_destination_oid_col(
    input_features: Union[arcpy._mp.Layer, str, Path, Iterable, pd.DataFrame],
    id_column: str,
) -> str:
    # if working with a feature class
    if isinstance(input_features, (arcpy._mp.Layer, str, Path)):
        # get destination unique id column if not explicitly provided for feature classes
        if id_column is None:
            id_column = arcpy.Describe(str(input_features)).OIDFieldName

        # otherwise, make sure provided column exists
        else:
            if id_column not in [f.name for f in arcpy.ListFields(str(input_features))]:
                raise ValueError(
                    f'The provided destination_id_column "{id_column}" does not appear to be in the '
                    f"destination_features schema."
                )

    # if working with a data frame, it will be the first column
    elif isinstance(input_features, pd.DataFrame):
        id_column = input_features.columns[0]

    # otherwise, just use the oid
    else:
        id_column = "oid"

    return id_column


def get_network_dataset_layer(
    network_dataset: Optional[Path] = None,
) -> arcpy._mp.Layer:
    """
    Get a network dataset layer, optionally using default.
    Args:
        network_dataset: Optional path to network dataset being used.

    .. note::
        If not specified, uses network solver set in Environment settings.

    Returns:
        NAX Layer.
    """
    # get the path to the country network dataset if it does not exist
    if network_dataset is None:
        network_dataset = Country(iso2).properties.network_path

    # ensure is string for GP
    if isinstance(network_dataset, Path):
        network_dataset = str(network_dataset)

    # create a network dataset layer
    nds_lyr = arcpy.nax.MakeNetworkDatasetLayer(network_dataset)[0]

    return nds_lyr


def get_network_travel_modes(network_dataset: Optional[Path] = None) -> list:
    """
    Get the travel modes, which can be used when solving for a network.

    Args:
        network_dataset: Optional path to network dataset being used.

    .. note::
        If not specified, uses network solver set in Environment settings.

    Returns:

    """
    # get the network dataset layer
    nds_lyr = get_network_dataset_layer(network_dataset)

    # retrieve the network dataset travel modes
    nd_travel_modes = arcpy.nax.GetTravelModes(nds_lyr)

    # get the travel modes as a list
    nd_travel_modes = list(nd_travel_modes.keys())

    return nd_travel_modes


def get_origin_destination_distance_parquet(
    origin_features: Union[str, Path, Iterable, pd.DataFrame],
    destination_features: Union[str, Path, Iterable, pd.DataFrame],
    parquet_path: Union[str, Path],
    origin_id_column: Optional[str] = None,
    destination_id_column: Optional[str] = None,
    network_dataset: Optional[Path] = None,
    travel_mode: Optional[str] = "Walking Distance",
    max_distance: Optional[float] = 5.0,
    search_distance: Optional[float] = 0.25,
    origin_batch_size: Optional[int] = 40000,
    partition_by_origin: Optional[bool] = True,
) -> Path:
    """
    Create an origin-destination matrix and save it to parquet.

    Args:
        origin_features: Origin features, the starting locations, for the origin-destination solve.
        destination_features: Destination features, the ending locations, for the origin-destination solve.
        parquet_path: Path where the origin-destination table will be saved as Parquet.
        origin_id_column: String column in input data with unique identifiers for each origin location.
        destination_id_column: String column in input data with unique identifiers for each destination.
        network_dataset: Optional path to network dataset to use.
        travel_mode: Travel mode to use with the network dataset. Default is ``Walking Distance``.
        max_distance: Maximum distance (in miles) to search from the origin to the destinations. Default is `5.0`.
        search_distance: Distance to search from the origin or destination locations to find a routable edge.
            Default is `0.25`.
        origin_batch_size: Number of origin locations to use per origin-destination solve. If experiencing memory
            overruns, reduce the batch size. The default is 40,000.
        partition_by_origin: If desired to partition the output by the ``origin_id_column``.

    Returns:
        Path to where Parquet dataset is saved.
    """
    # variable for network dataset layer name - potentially change if collisions occur
    nds_lyr_nm = "nds_lyr"

    # use the default network dataset if not provided
    if network_dataset is None:
        network_dataset = Country(iso2).properties.network_path

    # ensure path for network dataset is a string for geoprocessing
    if isinstance(network_dataset, Path):
        network_dataset = str(network_dataset)

    # check to ensure network dataset exists
    if not arcpy.Exists(network_dataset):
        raise FileNotFoundError(
            f"Cannot locate or access network dataset at {network_dataset}."
        )

    # ensure paths are actual path objects
    origin_features = (
        Path(origin_features) if isinstance(origin_features, str) else origin_features
    )
    destination_features = (
        Path(destination_features)
        if isinstance(destination_features, str)
        else destination_features
    )
    parquet_path = Path(parquet_path) if isinstance(parquet_path, str) else parquet_path

    # validate origin and destination geometry types, and if H3 indices, add geometries
    origin_features = validate_origin_destination_inputs(origin_features)
    destination_features = validate_origin_destination_inputs(destination_features)

    # if validation created feature sets, the unique identifier columns will be set to 'oid'
    if isinstance(origin_features, arcpy.FeatureSet):
        origin_id_column = "oid"
    if isinstance(destination_features, arcpy.FeatureSet):
        destination_id_column = "oid"

    # create a network dataset layer
    nds_lyr = arcpy.nax.MakeNetworkDatasetLayer(network_dataset)[0]
    logging.debug("Created network dataset layer.")

    # instantiate origin-destination cost matrix solver object
    odcm = arcpy.nax.OriginDestinationCostMatrix(nds_lyr)
    logging.debug("Created origin-destination cost matrix object.")

    # set the desired travel mode for analysis
    nd_travel_modes = arcpy.nax.GetTravelModes(nds_lyr)
    odcm.travelMode = nd_travel_modes[travel_mode]
    logging.debug(f'Origin-destination cost matrix travel mode set to "{travel_mode}"')

    # use miles for the distance units
    odcm.distanceUnits = arcpy.nax.DistanceUnits.Miles
    logging.debug("Origin-destination cost matrix distance units set to miles.")

    # maximum distance to solve for based on the distance units above
    odcm.defaultImpedanceCutoff = max_distance
    logging.info(
        f"Origin-destination cost matrix maximum solve distance (defaultImpedanceCutoff) set to "
        f"{max_distance} miles."
    )

    # use miles for the search distance - how far to "snap" points to nearest routable network edge
    odcm.searchToleranceUnits = arcpy.nax.DistanceUnits.Miles
    logging.debug(
        "Origin-destination cost matrix search tolerance (snap distance) units set to miles."
    )

    # set the search distance
    odcm.searchTolerance = search_distance
    logging.info(
        f"Origin-destination cost matrix search tolerance (snap distance) set to {search_distance} miles."
    )

    # don't need geometry, just the origin, destination and output
    odcm.lineShapeType = arcpy.nax.LineShapeType.NoLine
    logging.debug("Origin-destination cost matrix set to not return line geometry.")

    # get origin unique id column if not explicitly provided
    origin_id_column = get_origin_destination_oid_col(origin_features, origin_id_column)
    destination_id_column = get_origin_destination_oid_col(
        destination_features, destination_id_column
    )

    # get oid list for the input features to use for batching if not explicitly provided
    oid_lst = [
        r[0] for r in arcpy.da.SearchCursor(str(origin_features), origin_id_column)
    ]

    # get the count of input features for batching
    origin_cnt = len(oid_lst)

    # batch the solve based on the input feature count
    origin_batch_cnt = math.ceil(origin_cnt / origin_batch_size)

    logging.info(
        f"The origin-destination matrix solution will require {origin_batch_cnt:,} iterations."
    )

    # create a layers to use for speeding up the batching
    if isinstance(origin_features, Path):
        origin_features = str(origin_features)
    origin_lyr = arcpy.management.MakeFeatureLayer(origin_features)[0]

    if isinstance(destination_features, Path):
        destination_features = str(destination_features)
    dest_lyr = arcpy.management.MakeFeatureLayer(destination_features)[0]

    # make sure the location to save the parquet exists
    if not parquet_path.exists():
        parquet_path.mkdir(parents=True)

    # iterate the number of times it takes to process all the input features
    for batch_idx in range(origin_batch_cnt):
        logging.info(
            f"Starting the origin-destination cost matrix batch {batch_idx + 1} of {origin_batch_cnt:,}."
        )

        # create a list of the object identifiers in the input data for this batch
        start_idx = batch_idx * origin_batch_size
        end_idx = start_idx + origin_batch_size
        batch_oid_lst = oid_lst[start_idx:end_idx]

        # use the list of object identifiers to create an sql string to use for identifying features
        if isinstance(batch_oid_lst[0], str):
            oid_str_lst = [f"'{oid}'" for oid in batch_oid_lst]
        else:
            oid_str_lst = [f"{oid}" for oid in batch_oid_lst]

        oid_str = ",".join(oid_str_lst)
        sql_str = f"{origin_id_column} IN ({oid_str})"

        # apply the selection to the origin features
        arcpy.management.SelectLayerByAttribute(
            origin_lyr, selection_type="NEW_SELECTION", where_clause=sql_str
        )

        # select destination features within the 120% of the distance (straight line) of the origin features
        sel_dist = max_distance * 1.2
        arcpy.management.SelectLayerByLocation(
            dest_lyr,
            overlap_type="WITHIN_A_DISTANCE",
            select_features=origin_lyr,
            search_distance=f"{sel_dist} Miles",
            selection_type="NEW_SELECTION",
        )

        # create an NAX insert cursor to load the origin locations with centroids to allow more flexible geometry input
        with odcm.insertCursor(
            arcpy.nax.OriginDestinationCostMatrixInputDataType.Origins,
            ["Name", "SHAPE@XY"],
        ) as insert_origin:
            # use a search cursor to get the starting locations with unique identifiers (using centroids)
            for origin_row in arcpy.da.SearchCursor(
                origin_lyr, [origin_id_column, "SHAPE@XY"]
            ):
                # load the location
                insert_origin.insertRow(origin_row)

        logging.debug(
            f"Loaded {len(batch_oid_lst):,} origin features into the origin-destination solver "
            f"from {origin_features}."
        )

        # create an input cursor for the destinations, again using centroids
        with odcm.insertCursor(
            arcpy.nax.OriginDestinationCostMatrixInputDataType.Destinations,
            ["Name", "SHAPE@XY"],
        ) as insert_dest:
            # counter for added records
            dest_add_cnt = 0

            # use a search cursor to get the starting locations with unique identifiers (using centroids)
            for dest_row in arcpy.da.SearchCursor(
                dest_lyr, [destination_id_column, "SHAPE@XY"]
            ):
                # load the location
                insert_dest.insertRow(dest_row)

                # index the counter
                dest_add_cnt += 1

            logging.debug(
                f"Loaded {dest_add_cnt:,} destination candidates into the origin-destination solver "
                f"from {destination_features}."
            )

        # solve the origin-destination matrix
        logging.debug(f"Starting the origin-destination cost matrix solve.")
        result = odcm.solve()
        logging.debug("Completed the origin-destination cost matrix solve.")

        # grab the results as an arrow table
        logging.debug(
            "Starting to export the origin-destination result to an Arrow table."
        )

        # columns to retrieve from the solve result
        res_cols = ["OriginName", "DestinationName", "Total_Distance", "Total_Time"]

        # list to populate with the solve result
        res_lst = []

        # iterate the solve result
        for res_row in result.searchCursor(
            arcpy.nax.OriginDestinationCostMatrixOutputDataType.Lines, res_cols
        ):
            # create a dictionary
            res_dict = {
                "origin_id": res_row[0],
                "destination_id": res_row[1],
                "distance_miles": res_row[2],
                "time": res_row[3],
            }

            # add the dictionary to the list
            res_lst.append(res_dict)

        # create the schema to use for converting the list to a pyarrow table
        pa_schema = pa.schema(
            [
                pa.field("origin_id", pa.string()),
                pa.field("destination_id", pa.string()),
                pa.field("distance_miles", pa.int64()),
                pa.field("time", pa.int64()),
            ]
        )

        # create a pyarrow table from the result list
        solve_tbl = pa.Table.from_pylist(res_lst, schema=pa_schema)
        logging.debug(
            "Completed exporting the origin-destination result to an Arrow table."
        )

        # if  want the data organized by the origins
        if partition_by_origin:
            logging.debug(
                f"Starting to save the origin-destination cost matrix parquet to {parquet_path} and partition using "
                f'the origin id column "{origin_id_column}"'
            )
            # https://arrow.apache.org/docs/python/generated/pyarrow.dataset.partitioning.html
            # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html
            pq.write_to_dataset(
                solve_tbl,
                root_path=parquet_path,
                partition_cols=["origin_id"],
                compression="snappy",
                existing_data_behavior="overwrite_or_ignore",
            )
            logging.debug(
                f"Successfully saved the origin-destination cost matrix parquet to {parquet_path}"
            )

        else:
            # save the origin-destination table to a parquet dataset
            # ref: https://arrow.apache.org/cookbook/py/io.html#reading-a-parquet-
            logging.debug(
                f"Starting to save the origin-destination cost matrix parquet to {parquet_path}"
            )
            pq.write_to_dataset(
                solve_tbl,
                root_path=parquet_path,
                compression="snappy",
                existing_data_behavior="overwrite_or_ignore",
            )
            logging.debug(
                f"Successfully saved the origin-destination cost matrix parquet to {parquet_path}"
            )

    logging.info(
        f"Successfully created origin-destination cost matrix and saved the result as parquet to "
        f"{parquet_path}"
    )

    return parquet_path


def get_h3_origin_destination_distance_parquet(
    h3_features: Union[str, Path, List[str]],
    parquet_path: Union[str, Path],
    h3_index_column: str = "GRID_ID",
    network_dataset: Optional[Path] = None,
    travel_mode: str = "Walking Distance",
    max_distance: float = 5.0,
    search_distance: float = 1.0,
) -> Path:
    """
    Create an origin-destination matrix and save it to parquet.

    Args:
        h3_features: Path to H3 feature class created for area of interest using ArcGIS Pro.
        parquet_path: Path where the origin-destination table will be saved as Parquet.
        h3_index_column: Column in H3 feature class containing the H3 indices. Default is ``GRID_ID``.
        network_dataset: Optional path to network dataset to use.
        travel_mode: Travel mode to use with the network dataset. Default is ``Walking Distance``.
        max_distance: Maximum distance (in miles) to search from the origin to the destinations. Default is `5.0`.
        search_distance: Distance to search from the origin or destination locations to find a routable edge.
            Default is `1.0`.

    Returns:
        Path to where Parquet dataset is saved.
    """
    parquet_path = get_origin_destination_distance_parquet(
        origin_features=h3_features,
        origin_id_column=h3_index_column,
        destination_features=h3_features,
        destination_id_column=h3_index_column,
        network_dataset=network_dataset,
        parquet_path=parquet_path,
        travel_mode=travel_mode,
        max_distance=max_distance,
        search_distance=search_distance,
    )

    return parquet_path


def get_arrow_dataset(parquet_path: Union[str, Path]) -> ds.Dataset:
    ads = ds.dataset(parquet_path, format="parquet")
    return ads


def get_distance_between_h3_indices(
    origin_destination_dataset: Union[ds.Dataset, str, Path],
    h3_origin: str,
    h3_destination: str,
    warn_on_fail: bool = True,
) -> float:
    """
    Given an origin and destination H3 index, get the distance between the indices.

    Args:
        origin_destination_dataset: Origin-destination PyArrow dataset or path to Parquet dataset.
        h3_origin: Origin H3 index.
        h3_destination: Destination H3 index.
        warn_on_fail: Whether to warn if no results found.

    Returns:
        Distance between H3 indices.
    """
    # handle various ways origin-destination dataset can be provided, and ensure is PyArrow Dataset
    if not isinstance(origin_destination_dataset, ds.Dataset):
        origin_destination_dataset = get_arrow_dataset(origin_destination_dataset)

    # create the filter to find the record
    fltr = (pc.field("origin_id") == h3_origin) & (
        pc.field("destination_id") == h3_destination
    )

    # read in the table with the filter and convert to a list of dictionaries
    fltr_lst = origin_destination_dataset.to_table(filter=fltr).to_pylist()

    # handle contingency of not finding a match, but if found, provide the distance
    if len(fltr_lst) == 0:
        if warn_on_fail:
            warn(
                f"Cannot route between {h3_origin} and {h3_destination}. This may be due to the origin, destination "
                f"or both being unroutable, simply too far apart, or possibly not using the correct resolution "
                f"indices."
            )

        dist = None

    else:
        dist = fltr_lst[0]["distance_miles"]

    return dist


def get_distance_between_coordinates_using_h3(
    origin_destination_dataset: Union[ds.Dataset, str, Path],
    origin_coordinates: Union[Tuple[float], List[float]],
    destination_coordinates: Union[Tuple[float], List[float]],
    h3_resolution: int = 10,
    warn_on_fail: bool = True,
) -> float:
    """
    Given origin and destination coordinates, get the distance between using an H3 lookup.

    Args:
        origin_destination_dataset: Origin-destination PyArrow dataset or path to Parquet dataset.
        origin_coordinates: Origin coordinates in WGS84.
        destination_coordinates: Destination coordinates in WGS84.
        h3_resolution: H3 resolution origin-destination dataset is using.
        warn_on_fail: Whether to warn if no results found.

    Returns:
        Distance between origin and destination locations.
    """
    # get the indices for the origin and destination locations
    h3_origin = h3.latlng_to_cell(
        origin_coordinates[1], origin_coordinates[9], h3_resolution
    )
    h3_dest = h3.latlng_to_cell(
        destination_coordinates[1], destination_coordinates[0], h3_resolution
    )

    # get the distance between the locations
    dist = get_distance_between_h3_indices(
        origin_destination_dataset, h3_origin, h3_dest, warn_on_fail
    )

    return dist


def get_h3_neighbors(
    origin_destination_dataset: Union[ds.Dataset, str, Path],
    h3_origin: str,
    distance: float = 3.75,
    warn_on_fail: bool = False,
) -> pd.DataFrame:
    """
    Get neighbor H3 indices with distance from an origin H3 index.

    Args:
        origin_destination_dataset: Origin-destination PyArrow dataset or path to Parquet dataset.
        h3_origin: Origin H3 index.
        distance: Distance around origin to search for. Default is ``3.75``.
        warn_on_fail: Whether to warn if no results found. Default is ``False``.

    Returns:
        Pandas dataframe with destination indices and distance.
    """
    # handle various ways origin-destination dataset can be provided, and ensure is PyArrow Dataset
    if not isinstance(origin_destination_dataset, ds.Dataset):
        origin_destination_dataset = get_arrow_dataset(origin_destination_dataset)

    # create the filter for retrieving data
    fltr = (pc.field("origin_id") == h3_origin) & (
        pc.field("distance_miles") <= distance
    )

    # read in the table with the filter
    od_tbl = origin_destination_dataset.to_table(filter=fltr)

    # handle if no matches found
    if od_tbl.num_rows == 0:
        # provide status message if warning user
        if warn_on_fail:
            warn(
                f"Cannot find destinations for origin {h3_origin}. This potentially is due to the origin being "
                f"unroutable, or searching using the wrong H3 resolution."
            )

        od_df = None

    else:
        od_df = od_tbl.to_pandas()

    return od_df


def get_origin_destination_neighbors(
    origin_destination_dataset: Union[ds.Dataset, str, Path],
    origin_id: int,
    distance: float = 0.5,
    warn_on_fail: bool = False,
) -> pd.DataFrame:
    """
    Get neighbor unique identifiers surrounding an origin identifier.

    Args:
        origin_destination_dataset: Origin-destination PyArrow dataset or path to Parquet dataset.
        origin_id: Unique identifier for origin identifier.
        distance: Distance around origin to search for. Default is ``0.5``.
        warn_on_fail: Whether to warn if no results found. Default is ``False``.

    Returns:
        Pandas dataframe with destination indices and distance.
    """
    # handle various ways origin-destination dataset can be provided, and ensure is PyArrow Dataset
    if not isinstance(origin_destination_dataset, ds.Dataset):
        origin_destination_dataset = get_arrow_dataset(origin_destination_dataset)

    # create the filter for retrieving data
    fltr = (pc.field("origin_id") == origin_id) & (
        pc.field("distance_miles") <= distance
    )

    # read in the table with the filter
    od_tbl = origin_destination_dataset.to_table(filter=fltr)

    # handle if no matches found
    if od_tbl.num_rows == 0:
        # provide status message if warning user
        if warn_on_fail:
            warn(
                f'Cannot find destinations for OriginID, "{origin_id}". This likely is due to the origin being '
                f"unroutable."
            )

        od_df = None

    else:
        od_df = od_tbl.to_pandas()

    return od_df


def get_nearest_origin_destination_neighbor(
    origin_destination_dataset: Union[ds.Dataset, str, Path],
    origin_id: int,
    distance: float = 0.5,
    warn_on_fail: bool = False,
) -> Union[str, int]:
    """
    Get nearest neighbor unique identifier to an origin identifier.

    Args:
        origin_destination_dataset: Origin-destination PyArrow dataset or path to Parquet dataset.
        origin_id: Unique identifier for origin.
        distance: Distance around origin to search for. Default is ``0.5``.
        warn_on_fail: Whether to warn if no results found. Default is ``False``.

    Returns:
        Unique identifier for the destination.
    """
    # read in the table with the filter
    od_df = get_origin_destination_neighbors(
        origin_destination_dataset, origin_id, distance, warn_on_fail
    )

    # based on the minimum distance, not locations potentially were returned
    if len(od_df.index) == 0:
        # no destination id exists
        dest_id = None

    else:
        # get the minimum distance
        min_dist = od_df["distance_miles"].min()

        # use the minimum distance to get the destination id of the nearest
        dest_id = od_df[od_df["distance_miles"] == min_dist].iloc[0]["destination_id"]

        # ensure the value is an integer (typically it is a float)
        if isinstance(dest_id, float):
            dest_id = int(dest_id)

    return dest_id


def get_aoi_h3_origin_destination_distance_parquet(
    area_of_interest: Union[str, Path, arcpy.Geometry, List[arcpy.Geometry]],
    h3_resolution: int,
    parquet_path: Union[str, Path],
    # h3_index_column: Optional[str] = "h3_idx",
    network_dataset: Optional[Path] = None,
    travel_mode: Optional[str] = "Walking Distance",
    max_distance: Optional[float] = 5.0,
    search_distance: Optional[float] = 1.0,
) -> Path:
    """
    Create an origin-destination matrix and save it to parquet.

    Args:
        area_of_interest: Feature Clas or Geometry object describing the area of interest to generate
            an origin-destination matrix for using H3 indices.
        h3_resolution: H3 resolution to use when generating an origin-destination matrix.
        parquet_path: Path where the origin-destination table will be saved as Parquet.
        network_dataset: Optional path to network dataset to use.
        travel_mode: Travel mode to use with the network dataset. Default is ``Walking Distance``.
        max_distance: Maximum distance (in miles) to search from the origin to the destinations. Default is `5.0`.
        search_distance: Distance to search from the origin or destination locations to find a routable edge.
            Default is `1.0`.

    Returns:
        Path to where Parquet dataset is saved.
    """
    ### commented out parameter
    # h3_index_column: Column in H3 feature class containing the H3 indices. Default is ``GRID_ID``.

    # setting input batch size
    origin_batch_size = 500

    # if the AOI is described with a feature class, standardize the path to a string
    if isinstance(area_of_interest, Path):
        area_of_interest = str(area_of_interest)

    # if a path to a feature class, get an iterable of geometries to work with
    if isinstance(area_of_interest, str):
        # make sure multiple runs do not cause problems
        arcpy.env.overwriteOutput = True

        # ensure the area of interest is NOT multipart
        area_of_interest = arcpy.management.MultipartToSinglepart(
            area_of_interest, "memory/aoi"
        )

        # get a list of geometries
        area_of_interest = [
            geom
            for geom in [
                r[0] for r in arcpy.da.SearchCursor(area_of_interest, "SHAPE@")
            ]
        ]

    # if just a single geometry, make into a list
    if not isinstance(area_of_interest, Iterable):
        area_of_interest = [area_of_interest]

    # iterate the geometries getting nested iterable (generator) of h3 indices
    h3_idx_gen = (
        list(h3_arcpy.get_h3_indices_for_esri_polygon(geom, resolution=h3_resolution))
        for geom in area_of_interest
    )

    # iterate the generators into single iterable
    h3_origin_tpl = tuple(itertools.chain(*h3_idx_gen))

    # create a set to store potential destination indices
    h3_dest_set = set()

    # get all the neighbors for the origin h3 indices
    for h3_origin in h3_origin_tpl:
        # get potential destinations for the origin
        h3_dest_lst = h3.grid_disk(h3_origin)

        # add potential destinations indices to the set (eliminates duplicates)
        for h3_dest in h3_dest_lst:
            h3_dest_set.add(h3_dest)

    # solve the batch and save the incremental result
    get_origin_destination_distance_parquet(
        origin_features=h3_origin_tpl,
        destination_features=tuple(h3_dest_set),
        parquet_path=parquet_path,
        network_dataset=network_dataset,
        travel_mode=travel_mode,
        max_distance=max_distance,
        search_distance=search_distance,
        origin_batch_size=origin_batch_size,
        partition_by_origin=True,
    )
