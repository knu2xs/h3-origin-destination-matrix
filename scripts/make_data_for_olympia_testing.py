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
from h3_od.config import load_config

if __name__ == "__main__":
    # Load olympia_walk config
    config = load_config(environment="olympia_walk")

    # resolve paths from olympia_walk config
    aoi_features = Path(config.data.aoi_polygon)
    od_parquet = Path(config.data.output_od_parquet)
    network_dataset = Path(config.network.dataset)

    # path for saving logging
    dt_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
    log_pth = od_parquet.parent / f"{Path(__file__).stem}_{dt_str}.log"

    # configure logging
    logger = get_logger(level=config.logging.level, logfile_path=log_pth)

    logger.info(
        f"Solving origin-destination matrix using {network_dataset} and saving to {od_parquet}."
    )

    # create the origin-destination matrix
    h3_od.proximity.get_aoi_h3_origin_destination_distance_parquet(
        area_of_interest=aoi_features,
        parquet_path=od_parquet,
        h3_resolution=int(config.h3.resolution),
        network_dataset=network_dataset,
        travel_mode=config.network.travel_mode,
        max_distance=config.network.max_distance,
        search_distance=config.network.snap_distance,
        origin_batch_size=config.network.origin_batch_size,
    )
