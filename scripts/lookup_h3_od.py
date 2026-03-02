"""Script to look up H3 origin-destination distances from a solved matrix."""
from pathlib import Path
import importlib.util
import os
import sys

# ---------------------------------------------------------------------------
# Set the environment (or override with PROJECT_ENV env var)
# This MUST be set before importing h3_od so config.py picks it up.
# ---------------------------------------------------------------------------
if "PROJECT_ENV" not in os.environ:
    os.environ["PROJECT_ENV"] = "dev"

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
from h3_od.config import LOG_LEVEL, OUTPUT_OD_PARQUET

if __name__ == "__main__":

    od_parquet = Path(OUTPUT_OD_PARQUET)

    # configure logging
    logger = get_logger(logger_name=Path(__file__).stem, level=LOG_LEVEL)

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
