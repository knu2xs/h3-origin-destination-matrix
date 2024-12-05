"""
Licensing

Copyright 2020 Esri

Licensed under the Apache License, Version 2.0 (the "License"); You
may not use this file except in compliance with the License. You may
obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing
permissions and limitations under the License.

A copy of the license is available in the repository's
LICENSE file.
"""
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

log_level = config.get("DEFAULT", "LOG_LEVEL")
aoi_features = dir_prj / config.get("DEFAULT", "AOI_POLYGON")
od_parquet = dir_prj / config.get("DEFAULT", "OUTPUT_OD_PARQUET")
snap_distance = float(config.get('DEFAULT', 'SNAP_DISTANCE'))
network_dataset = Path(config.get("DEFAULT", "NETWORK_DATASET"))
travel_mode = config.get('DEFAULT', 'TRAVEL_MODE')
h3_resolution = int(config.get("DEFAULT", "H3_RESOLUTION"))
origin_batch_size = int(config.get("DEFAULT", "ORIGIN_BATCH_SIZE"))

# path for saving logging
dt_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
log_pth = od_parquet.parent / f"od_solve_{dt_str}.log"

# configure logging
h3_od.utils.logging_utils.configure_logging(log_level, logfile_path=log_pth)

logging.info(
    f"Solving origin-destination matrix using {network_dataset} using H3 resolution {h3_resolution}, and "
    f"saving to {od_parquet}."
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
