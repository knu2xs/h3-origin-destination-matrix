import datetime
import logging
from pathlib import Path
import shutil
import sys
from tempfile import mkdtemp

import arcpy.management
import pytest

# get paths to useful resources
self_pth = Path(__file__)
dir_test = self_pth.parent
dir_test_data = dir_test / "data"

dir_prj = dir_test.parent
dir_src = dir_prj / "src"

# insert the src directory into the path and import the project package
sys.path.insert(0, str(dir_src))
import h3_od
from h3_od.utils._logging import get_logger

# configure logging
dt_str = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
logger = get_logger(logfile_path=dir_test / f"test_h3_od_proximity_{dt_str}.log")

# global resources
NETWORK_DATASET = Path(
    r"D:\data\ba_data\usa_2024\Data\StreetMap Premium Data\northamerica.geodatabase\main.Routing\main.Routing_ND"
)
GDB = dir_test_data / "olympia.gdb"
H3_FEATURES = GDB / "h3_08"
H3_CENTROID_FEATURES = GDB / "h3_08_centroids"
AOI_FEATURES = GDB / "olympia_aoi"


@pytest.fixture(scope="function")
def tmp_dir() -> Path:
    temp_dir = Path(mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def tmp_gdb(temp_dir) -> Path:
    temp_gdb = arcpy.management.CreateFileGDB(str(temp_dir), "temp.gdb")[0]
    temp_gdb = Path(temp_gdb)
    yield temp_gdb
    arcpy.management.Delete(temp_gdb)


@pytest.fixture(scope="session")
def h3_idx_lst():
    idx_lst = [r[0] for r in arcpy.da.SearchCursor(str(H3_FEATURES), "GRID_ID")]
    return idx_lst


class TestOlympia:

    def test_get_origin_destination_parquet(self, tmp_dir, h3_idx_lst):
        tmp_parquet = tmp_dir / "parquet"

        h3_od.proximity.get_origin_destination_parquet(
            origin_h3_indices=h3_idx_lst,
            destination_h3_indices=h3_idx_lst,
            parquet_path=tmp_parquet,
            network_dataset=NETWORK_DATASET,
            travel_mode="Walking Distance",
            max_distance=5.0,
            search_distance=1.0,
            origin_batch_size=40000,
        )

        part_file_lst = list(tmp_parquet.glob("**/*.parquet"))

        assert len(part_file_lst) > 1

    def test_get_origin_destination_distance_parquet_from_arcgis_features(
        self, tmp_dir
    ):
        tmp_parquet = tmp_dir / "parquet"

        h3_od.proximity.get_origin_destination_distance_parquet_from_arcgis_features(
            h3_features=H3_FEATURES,
            parquet_path=tmp_parquet,
            network_dataset=NETWORK_DATASET,
            travel_mode="Walking Distance",
            max_distance=5.0,
            search_distance=1.0,
        )

        part_file_lst = list(tmp_parquet.glob("**/*.parquet"))

        assert len(part_file_lst) > 1

    def test_get_aoi_h3_origin_destination_distance_parquet(self, tmp_dir):
        tmp_parquet = tmp_dir / "parquet"

        h3_od.proximity.get_aoi_h3_origin_destination_distance_parquet(
            area_of_interest=AOI_FEATURES,
            h3_resolution=8,
            parquet_path=tmp_parquet,
            network_dataset=NETWORK_DATASET,
            travel_mode="Walking Distance",
            max_distance=5.0,
            search_distance=1.0,
        )

        part_file_lst = list(tmp_parquet.glob("**/*.parquet"))

        assert len(part_file_lst) > 1
