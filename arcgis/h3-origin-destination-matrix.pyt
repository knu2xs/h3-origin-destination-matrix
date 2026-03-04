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
        self.tools = [AddDestinationDistance, GetH3Indices, CreateOriginH3FeatureClass]


class CreateOriginH3FeatureClass:
    def __init__(self):
        self.label = "Create Origin H3 Feature Class"
        self.description = "Create a feature class of origin H3 indices from an OD matrix parquet dataset."
        self.category = "Utilities"
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
        output_fc = arcpy.Parameter(
            displayName="Output Feature Class (Polygons)",
            name="output_fc",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output",
        )
        return [od_matrix, output_fc]

    def execute(self, parameters, messages):
        od_matrix_folder = parameters[0].valueAsText
        output_fc = parameters[1].valueAsText

        # Only validate for .part files in the directory
        if not os.path.isdir(od_matrix_folder):
            self.logger.error("OD matrix must be a directory containing .part files.")
            return

        self.logger.info(f"Reading OD matrix from: {od_matrix_folder}")

        part_files = [
            f
            for f in os.listdir(od_matrix_folder)
            if f.lower().endswith(".part")
            and os.path.isfile(os.path.join(od_matrix_folder, f))
        ]

        if not part_files:
            self.logger.error(
                "OD matrix directory must contain at least one .part file."
            )
            return

        schema_path = os.path.join(od_matrix_folder, part_files[0])

        # Efficiently read only the origin_id column
        od_dataset = ds.dataset(schema_path, format="parquet")
        origin_id_col = od_dataset.to_table(columns=["origin_id"]).column("origin_id")
        unique_origin_ids = origin_id_col.unique().to_pylist()

        if not unique_origin_ids:
            self.logger.error("No origin H3 indices found in OD matrix.")
            return

        self.logger.info(f"Found {len(unique_origin_ids)} unique origin H3 indices.")

        # Get resolution from first origin_id
        resolution = h3_arcpy.get_h3_resolution(unique_origin_ids[0])
        self.logger.info(f"Detected H3 resolution: {resolution}")

        # Create output feature class (always WGS84)
        sr = arcpy.SpatialReference(4326)
        arcpy.management.CreateFeatureclass(
            out_path=os.path.dirname(output_fc),
            out_name=os.path.basename(output_fc),
            geometry_type="POLYGON",
            spatial_reference=sr,
        )
        arcpy.management.AddField(output_fc, "h3_index", "TEXT")

        # Insert each geometry on the fly
        with arcpy.da.InsertCursor(output_fc, ["SHAPE@", "h3_index"]) as cursor:

            for h3_index in unique_origin_ids:

                try:
                    poly = h3_arcpy.get_arcpy_polygon_for_h3_index(h3_index)
                    cursor.insertRow([poly, h3_index])

                except Exception as e:
                    self.logger.warning(
                        f"Failed to create/insert polygon for H3 index {h3_index}: {e}"
                    )

        self.logger.info(f"Created origin H3 polygons feature class: {output_fc}")
        return


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
            f
            for f in os.listdir(od_matrix_folder)
            if f.lower().endswith(".part")
            and os.path.isfile(os.path.join(od_matrix_folder, f))
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

        # Prepare the list of fields to update in the cursor
        update_fields = ["SHAPE@", distance_field]
        if time_field:
            update_fields.append(time_field)

        # Use UpdateCursor to iterate through each feature and update OD results
        with arcpy.da.UpdateCursor(input_features, update_fields) as cursor:
            for row in cursor:

                # Get geometry and compute H3 index for the origin feature
                geom = row[0]
                h3_origin = h3_arcpy.get_h3_index_for_esri_geometry(geom, resolution)

                try:
                    # Filter OD matrix for matching origin
                    od_result = od_df[od_df["origin_id"] == h3_origin]

                    # If a destination is specified, further filter by destination
                    if h3_destination:
                        od_result = od_result[
                            od_result["destination_id"] == h3_destination
                        ]

                    # If a matching OD record is found, update the feature
                    if not od_result.empty:

                        # If multiple records, sort by time (if present) or by distance
                        if len(od_result) > 1:
                            if "time" in od_result.columns:
                                od_result = od_result.sort_values("time")
                                self.logger.debug(
                                    f"Multiple records found for origin {h3_origin}. Sorted by 'time'. Using nearest."
                                )
                            else:
                                od_result = od_result.sort_values("distance_miles")
                                self.logger.debug(
                                    f"Multiple records found for origin {h3_origin}. Sorted by 'distance_miles'. "
                                    f"Using nearest."
                                )

                        # Update the distance field
                        row[update_fields.index(distance_field)] = od_result.iloc[0][
                            "distance_miles"
                        ]

                        # Update the time field if enabled and present
                        if time_field and "time" in od_result.columns:
                            row[update_fields.index(time_field)] = od_result.iloc[0][
                                "time"
                            ]

                        # Commit the update to the feature
                        cursor.updateRow(row)

                except Exception as e:
                    self.logger.warning(
                        f"Failed to get OD distance for origin {h3_origin}: {e}"
                    )

        self.logger.info(f"Distance and time fields updated in input features.")

        return


class GetH3Indices:
    def __init__(self):
        self.label = "Get H3 Indices"
        self.description = "Get H3 indices for an area of interest polygon feature class, with options for selection method and centroid output."
        self.category = "Utilities"
        logger_name = f"h3_od.Toolbox.{self.__class__.__name__}"
        self.logger = get_logger(logger_name, level="INFO", add_arcpy_handler=True)

    def getParameterInfo(self):
        input_aoi_features = arcpy.Parameter(
            displayName="Input AOI Polygon Features",
            name="input_aoi_features",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )
        h3_resolution = arcpy.Parameter(
            displayName="H3 Resolution",
            name="h3_resolution",
            datatype="GPLong",
            parameterType="Required",
            direction="Input",
        )
        selection_method = arcpy.Parameter(
            displayName="Selection Method",
            name="selection_method",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        selection_method.filter.type = "ValueList"
        selection_method.filter.list = [
            "completely_within",
            "centroid_within",
            "polygon_intersecting",
        ]
        selection_method.value = "polygon_intersecting"
        output_fc = arcpy.Parameter(
            displayName="Output Feature Class (Polygons)",
            name="output_fc",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output",
        )
        create_centroids = arcpy.Parameter(
            displayName="Create Centroids Feature Class",
            name="create_centroids",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        create_centroids.value = False
        return [
            input_aoi_features,
            h3_resolution,
            selection_method,
            output_fc,
            create_centroids,
        ]

    def execute(self, parameters, messages):
        input_aoi_features = parameters[0].valueAsText
        h3_resolution = int(parameters[1].value)
        selection_method = parameters[2].value
        output_fc = parameters[3].valueAsText
        create_centroids = (
            parameters[4].value if parameters[4].value is not None else False
        )

        self.logger.info(f"Input AOI features: {input_aoi_features}")
        self.logger.info(f"H3 resolution: {h3_resolution}")
        self.logger.info(f"Selection method: {selection_method}")
        self.logger.info(f"Output polygons feature class: {output_fc}")
        self.logger.info(f"Create centroids: {create_centroids}")

        # Read AOI polygons
        arr = arcpy.da.FeatureClassToNumPyArray(
            input_aoi_features, ["SHAPE@"], skip_nulls=True
        )
        aoi_geoms = arr["SHAPE@"]
        h3_indices = set()
        for geom in aoi_geoms:
            try:
                if selection_method == "completely_within":
                    indices = h3_arcpy.get_h3_indices_for_esri_polygon(
                        geom, h3_resolution, contain="full"
                    )
                elif selection_method == "centroid_within":
                    indices = h3_arcpy.get_h3_indices_for_esri_polygon(
                        geom, h3_resolution, contain="center"
                    )
                else:
                    indices = h3_arcpy.get_h3_indices_for_esri_polygon(
                        geom, h3_resolution, contain="overlap"
                    )
                h3_indices.update(indices)
            except Exception as e:
                self.logger.warning(f"Failed to get H3 indices for AOI polygon: {e}")

        # Create polygons from H3 indices
        polygons = []
        for h3_index in h3_indices:
            try:
                poly = h3_arcpy.get_arcpy_polygon_for_h3_index(h3_index)
                polygons.append((h3_index, poly))
            except Exception as e:
                self.logger.warning(
                    f"Failed to create polygon for H3 index {h3_index}: {e}"
                )

        # Write polygons to output feature class
        sr = arcpy.Describe(input_aoi_features).spatialReference
        arcpy.management.CreateFeatureclass(
            out_path=os.path.dirname(output_fc),
            out_name=os.path.basename(output_fc),
            geometry_type="POLYGON",
            spatial_reference=sr,
        )
        arcpy.management.AddField(output_fc, "h3_index", "TEXT")
        with arcpy.da.InsertCursor(output_fc, ["SHAPE@", "h3_index"]) as cursor:
            for h3_index, poly in polygons:
                cursor.insertRow([poly, h3_index])

        self.logger.info(f"Created polygons feature class: {output_fc}")

        # Optionally create centroids feature class
        if create_centroids:
            centroid_fc = output_fc + "_centroids"
            arcpy.management.CreateFeatureclass(
                out_path=os.path.dirname(centroid_fc),
                out_name=os.path.basename(centroid_fc),
                geometry_type="POINT",
                spatial_reference=sr,
            )
            arcpy.management.AddField(centroid_fc, "h3_index", "TEXT")
            with arcpy.da.InsertCursor(centroid_fc, ["SHAPE@", "h3_index"]) as cursor:
                for h3_index in h3_indices:
                    try:
                        centroid = h3_arcpy.get_arcpy_point_for_h3_index(h3_index)
                        cursor.insertRow([centroid, h3_index])
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to create centroid for H3 index {h3_index}: {e}"
                        )
            self.logger.info(f"Created centroids feature class: {centroid_fc}")
        return
