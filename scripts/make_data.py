import datetime
from pathlib import Path
import importlib.util
import sys
import os

# Environment name constant
ENV_NAME = "rivco_walk"  # Change as needed

# Set the environment variable before importing config
os.environ["PROJECT_ENV"] = ENV_NAME

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

    # configure logging - configure default logger to log to a file with the specified log level
    logger = get_logger(level=LOG_LEVEL, logfile_path=log_pth, add_stream_handler=True)

    logger.info(
        f"[ENV: {ENV_NAME}] Solving origin-destination matrix using {network_dataset} using H3 resolution {h3_resolution}, and "
        f"saving to {od_parquet}."
    )

    # create the origin-destination matrix
    h3_od.proximity.get_aoi_h3_origin_destination_distance_parquet(
        area_of_interest=aoi_features,
        parquet_path=od_parquet,
        h3_resolution=h3_resolution,
        network_dataset=network_dataset,
        travel_mode=TRAVEL_MODE,
        max_distance=MAX_DISTANCE,
        search_distance=SNAP_DISTANCE,
        origin_batch_size=ORIGIN_BATCH_SIZE,
    )
