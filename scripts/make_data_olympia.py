"""Script to generate an origin-destination matrix for the Olympia area of interest."""
import datetime
from pathlib import Path
import importlib.util
import os
import sys

import arcpy

# ---------------------------------------------------------------------------
# Set the environment to olympia_walk (or override with PROJECT_ENV env var)
# This MUST be set before importing h3_od so config.py picks it up.
# ---------------------------------------------------------------------------
if "PROJECT_ENV" not in os.environ:
    os.environ["PROJECT_ENV"] = "olympia_walk"

# path to the root of the project
dir_prj = Path(__file__).parent.parent

# if the project package is not installed in the environment
if importlib.util.find_spec("h3_od") is None:
    # get the relative path to where the source directory is located
    src_dir = dir_prj / "src"

    # throw an error if the source directory cannot be located
    if not src_dir.exists():
        raise EnvironmentError("Unable to import h3_od.")

    # add the source directory to the paths searched when importing
    sys.path.insert(0, str(src_dir))

# import h3_od
import h3_od
from h3_od.utils import get_logger
from h3_od.config import (
    LOG_LEVEL,
    H3_RESOLUTION,
    AOI_POLYGON,
    OUTPUT_OD_PARQUET,
    SNAP_DISTANCE,
    NETWORK_DATASET,
    TRAVEL_MODE,
    MAX_DISTANCE,
    ORIGIN_BATCH_SIZE,
)

if __name__ == "__main__":

    # resolve paths
    aoi_features = Path(AOI_POLYGON)
    od_parquet = Path(OUTPUT_OD_PARQUET)
    network_dataset = Path(NETWORK_DATASET)
    h3_resolution = int(H3_RESOLUTION)

    # path for saving logging
    dt_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
    log_pth = od_parquet.parent / f"od_solve_{dt_str}.log"

    # configure logging
    logger = get_logger(logger_name=Path(__file__).stem, level=LOG_LEVEL, logfile_path=log_pth)

    logger.info(
        f"Solving origin-destination matrix using {network_dataset} using H3 resolution {h3_resolution}, and "
        f"saving to {od_parquet}."
    )

    h3_od.proximity.get_aoi_h3_origin_destination_distance_parquet(
        area_of_interest=aoi_features,
        h3_resolution=h3_resolution,
        parquet_path=od_parquet,
        network_dataset=network_dataset,
        travel_mode=TRAVEL_MODE,
        max_distance=MAX_DISTANCE,
        search_distance=SNAP_DISTANCE,
        origin_batch_size=ORIGIN_BATCH_SIZE,
    )
