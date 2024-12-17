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
od_parquet = Path(config.get(config_group, "OUTPUT_OD_PARQUET"))

# path for saving logging
# dt_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
# log_pth = od_parquet.parent / f"od_solve_{dt_str}.log"

# configure logging
h3_od.utils.configure_logging(log_level)

# origin and destination
h3_origin, h3_dest = "8829b60921fffff", "8829b60929fffff"

# retrieve dist
res = h3_od.proximity.get_h3_origin_destination_distance(
    origin_destination_dataset=od_parquet,
    h3_origin=h3_origin,
    # h3_destination=h3_dest,
    add_geometry=True,
)

assert res
