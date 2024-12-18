import datetime
from configparser import ConfigParser
import logging
from pathlib import Path
import importlib.util
import sys

import arcpy

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

# config group to retrieve
config_group = "SCOTTSDALE_WALK"

log_level = config.get(config_group, "LOG_LEVEL")
aoi_features = Path(config.get(config_group, "AOI_POLYGON"))
h3_features = Path(config.get(config_group, "H3_FEATURES"))
od_parquet = Path(config.get(config_group, "OUTPUT_OD_PARQUET"))
snap_distance = float(config.get(config_group, "SNAP_DISTANCE"))
max_distance = float(config.get(config_group, "MAX_DISTANCE"))
network_dataset = Path(config.get(config_group, "NETWORK_DATASET"))
travel_mode = config.get(config_group, "TRAVEL_MODE")
h3_resolution = int(config.get(config_group, "H3_RESOLUTION"))
origin_batch_size = int(config.get(config_group, "ORIGIN_BATCH_SIZE"))

# path for saving logging
dt_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
log_pth = od_parquet.parent / f"od_solve_{dt_str}.log"

# configure logging
h3_od.utils.logging_utils.configure_logging(log_level, logfile_path=log_pth)

# get a list of h3 indices to work from
h3_idx_lst = [r[0] for r in arcpy.da.SearchCursor(str(h3_features), "GRID_ID")]

logging.info(
    f"Solving origin-destination matrix using {network_dataset} using H3 resolution {h3_resolution}, and "
    f"saving to {od_parquet}."
)

# create the origin-destination matrix
# h3_od.proximity.get_aoi_h3_origin_destination_distance_parquet(
#     area_of_interest=aoi_features,
#     parquet_path=od_parquet,
#     h3_resolution=h3_resolution,
#     network_dataset=network_dataset,
#     travel_mode=travel_mode,
#     max_distance=max_distance,
#     search_distance=snap_distance,
#     origin_batch_size=origin_batch_size,
# )

h3_od.proximity.get_origin_destination_parquet(
    origin_h3_indices=h3_idx_lst,
    parquet_path=od_parquet,
    network_dataset=network_dataset,
    travel_mode=travel_mode,
    max_distance=max_distance,
    search_distance=snap_distance,
    origin_batch_size=origin_batch_size,
)
