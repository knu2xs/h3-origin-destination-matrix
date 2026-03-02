"""Script to generate an origin-destination matrix for an area of interest."""
import datetime
from pathlib import Path
import importlib.util
import sys

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

    # path for saving logging
    dt_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
    log_pth = od_parquet.parent / f"{Path(__file__).stem}_{dt_str}.log"

    # configure logging
    logger = get_logger(logger_name=Path(__file__).stem, level=LOG_LEVEL, logfile_path=log_pth)

    logger.info(
        f"Solving origin-destination matrix using {network_dataset} and saving to {od_parquet}."
    )

    # create the origin-destination matrix
    h3_od.proximity.get_aoi_h3_origin_destination_distance_parquet(
        area_of_interest=aoi_features,
        parquet_path=od_parquet,
        h3_resolution=int(h3_od.config.H3_RESOLUTION),
        network_dataset=network_dataset,
        travel_mode=TRAVEL_MODE,
        max_distance=MAX_DISTANCE,
        search_distance=SNAP_DISTANCE,
        origin_batch_size=ORIGIN_BATCH_SIZE,
    )
