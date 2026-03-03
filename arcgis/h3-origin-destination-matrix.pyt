# -*- coding: utf-8 -*-
__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "Apache 2.0"

# Python native imports
import importlib.util
from pathlib import Path
import sys
import os

# Third-party package imports
import arcpy
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds

from h3_od.utils import get_logger
import h3_od.utils.h3_arcpy as h3_arcpy


def find_pkg_source(package_name) -> Path:
    file_dir = Path(__file__).parent
    for idx in range(4):
        tmp_pth = file_dir.parent / "src" / package_name
        if tmp_pth.exists():
            return tmp_pth.parent
        else:
            file_dir = file_dir.parent
    raise ImportError(f"Could not find package source for {package_name}")


# Ensure src/h3_od is importable
if importlib.util.find_spec("h3_od") is None:
    src_dir = find_pkg_source("h3_od")
    if src_dir is not None:
        sys.path.append(str(src_dir))


class Toolbox:
    def __init__(self):
        self.label = "H3 Origin Destination Matrix"
        self.alias = "h3_od_matrix"
        self.tools = [AddDestinationDistance]


class AddDestinationDistance:
    def __init__(self):
        self.label = "Add Destination Distance"
        self.description = "Add the distance, and optionally the time, from a solved H3 OD matrix to the source data."
        self.category = "Analysis"
        logger_name = f"h3_od.Toolbox.{self.__class__.__name__}"
        self.logger = get_logger(logger_name, level="INFO", add_arcpy_handler=True)

    def getParameterInfo(self):
        od_matrix = arcpy.Parameter(
            displayName="H3 OD Matrix Parquet Dataset",
            name="od_matrix",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input",
        )
        input_features = arcpy.Parameter(
            displayName="Input Features/Table",
            name="input_features",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )
        h3_destination = arcpy.Parameter(
            displayName="Destination H3 Index (optional)",
            name="h3_destination",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        add_geometry = arcpy.Parameter(
            displayName="Add Geometry",
            name="add_geometry",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        add_geometry.value = False
        distance_field = arcpy.Parameter(
            displayName="Distance Field Name",
            name="distance_field",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        distance_field.value = "h3_destination_distance"
        time_field = arcpy.Parameter(
            displayName="Time Field Name (optional)",
            name="time_field",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        time_field.value = "h3_destination_time"
        time_field.enabled = False
        return [
            od_matrix,
            input_features,
            h3_destination,
            add_geometry,
            distance_field,
            time_field,
        ]

    def updateParameters(self, parameters):
        od_matrix = parameters[0]
        time_field = parameters[5]

        od_matrix.setErrorMessage("")
        od_path = od_matrix.valueAsText
        schema_path = None
        if od_path:
            if os.path.isfile(od_path):
                ext = os.path.splitext(od_path)[1].lower()
                if ext not in [".parquet", ".part"]:
                    od_matrix.setErrorMessage(
                        "OD matrix file must have .parquet or .part extension."
                    )
                else:
                    schema_path = od_path
            elif os.path.isdir(od_path):
                files = [
                    f
                    for f in os.listdir(od_path)
                    if f.lower().endswith(".part")
                    and os.path.isfile(os.path.join(od_path, f))
                ]
                if not files:
                    od_matrix.setErrorMessage(
                        "OD matrix folder must contain at least one .part file."
                    )
                else:
                    schema_path = os.path.join(od_path, files[0])
            else:
                od_matrix.setErrorMessage(
                    "OD matrix path must be a valid file or folder."
                )
        # Only check schema if a valid file is found and no error
        if schema_path and not od_matrix.errorMessage:
            try:
                schema = pq.read_schema(schema_path)
                if "time" in [field.name for field in schema]:
                    time_field.enabled = True
                else:
                    time_field.enabled = False
            except Exception:
                time_field.enabled = False
        else:
            time_field.enabled = False
        return

    def execute(self, parameters, messages):

        od_matrix_folder = parameters[0].valueAsText
        input_features = parameters[1].valueAsText
        h3_destination = parameters[2].valueAsText if parameters[2].value else None
        add_geometry = parameters[3].value if parameters[3].value is not None else False
        distance_field = parameters[4].valueAsText or "h3_destination_distance"
        time_field = (
            parameters[5].valueAsText
            if parameters[5].enabled and parameters[5].value
            else None
        )

        # Ensure OD matrix folder contains at least one .part file
        if not os.path.isdir(od_matrix_folder):
            self.logger.error("OD matrix path must be a folder.")
            return

        part_files = [
            f for f in os.listdir(od_matrix_folder)
            if f.lower().endswith(".part") and os.path.isfile(os.path.join(od_matrix_folder, f))
        ]

        if not part_files:
            self.logger.error("OD matrix folder must contain at least one .part file.")
            return

        od_matrix_path = os.path.join(od_matrix_folder, part_files[0])
        self.logger.info(f"Reading OD matrix from: {od_matrix_path}")
        self.logger.info(
            f"Input features: {input_features}, Destination: {h3_destination}, Add Geometry: {add_geometry}"
        )
        self.logger.info(f"Distance field: {distance_field}, Time field: {time_field}")

        # Read OD matrix and get resolution from first origin_id
        od_dataset = ds.dataset(od_matrix_path, format="parquet")
        od_tbl = od_dataset.to_table()
        od_df = od_tbl.to_pandas()

        if od_df.empty:
            self.logger.error("OD matrix is empty.")
            return

        first_origin = od_df.iloc[0]["origin_id"]
        resolution = h3_arcpy.get_h3_resolution(first_origin)
        self.logger.info(f"Detected H3 resolution from OD matrix: {resolution}")

        # Read input features to DataFrame
        arr = arcpy.da.FeatureClassToNumPyArray(
            input_features, ["SHAPE@"], skip_nulls=True
        )

        df_features = pd.DataFrame(arr)
        df_features[distance_field] = None

        if time_field:
            df_features[time_field] = None

        # For each feature, get H3 index and lookup OD matrix
        for idx, row in df_features.iterrows():

            geom = row["SHAPE@"]
            h3_origin = h3_arcpy.get_h3_index_for_esri_geometry(geom, resolution)

            try:
                od_result = od_df[od_df["origin_id"] == h3_origin]

                if h3_destination:
                    od_result = od_result[od_result["destination_id"] == h3_destination]
                if not od_result.empty:

                    # If multiple records, sort by time if present, else by distance
                    if len(od_result) > 1:

                        if "time" in od_result.columns:
                            od_result = od_result.sort_values("time")
                            self.logger.info(
                                f"Multiple records found for origin {h3_origin}. Sorted by 'time'. Using nearest."
                            )

                        else:
                            od_result = od_result.sort_values("distance_miles")
                            self.logger.info(
                                f"Multiple records found for origin {h3_origin}. Sorted by 'distance_miles'. "
                                f"Using nearest."
                            )

                    df_features.at[idx, distance_field] = od_result.iloc[0]["distance_miles"]

                    if time_field and "time" in od_result.columns:
                        df_features.at[idx, time_field] = od_result.iloc[0]["time"]

            except Exception as e:
                self.logger.warning(
                    f"Failed to get OD distance for origin {h3_origin}: {e}"
                )

        self.logger.info(
            f"Distance and time fields added to input features (not saved to disk by this tool)"
        )

        return
