import datetime
from configparser import ConfigParser
import logging
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

# read and configure
config = ConfigParser()
config.read(Path(__file__).parent / "config.ini")

LOG_LEVEL = config.get("DEFAULT", "LOG_LEVEL")
AOI_FEATURES = dir_prj / config.get("DEFAULT", "AOI_POLYGON")
OD_PARQUET = dir_prj / config.get("DEFAULT", "OUTPUT_OD_PARQUET")
SNAP_DISTANCE = float(config.get('DEFAULT', 'SNAP_DISTANCE'))
NETWORK_DATASET = Path(config.get("DEFAULT", "NETWORK_DATASET"))
TRAVEL_MODE = config.get('DEFAULT', 'TRAVEL_MODE')
ORIGIN_BATCH_SIZE = int(config.get("DEFAULT", "ORIGIN_BATCH_SIZE"))

# path for saving logging
dt_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
log_pth = OD_PARQUET.parent / f"{Path(__file__).stem}_{dt_str}.log"

# configure logging
logger = h3_od.utils._logging.get_logger(level=log_level, logfile_path=log_pth)

logging.info(
    f"Solving origin-destination matrix using {NETWORK_DATASET} and saving to {od_parquet}."
)

# create the origin-destination matrix
h3_od.proximity.get_aoi_h3_origin_destination_distance_parquet(
    area_of_interest=aoi_features,
    parquet_path=od_parquet,
    h3_resolution=h3_resolution,
    network_dataset=network_dataset,
    travel_mode=travel_mode,
    max_distance=180.0,
    search_distance=snap_distance,
    origin_batch_size=origin_batch_size,
)
