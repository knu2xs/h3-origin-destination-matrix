"""
This is a stubbed out test file designed to be used with PyTest, but can 
easily be modified to support any testing framework.
"""
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

network_dataset = Path(
    r"D:\data\ba_data\usa_2024\Data\StreetMap Premium Data\northamerica.geodatabase\main.Routing\main.Routing_ND"
)

# insert the src directory into the path and import the project package
sys.path.insert(0, str(dir_src))
import h3_od


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


class TestOlympia:
    gdb = dir_test_data / "olympia.gdb"
    h3_features = gdb / "h3_09"
    h3_centroid_features = gdb / "h3_09_centroids"

    logging.basicConfig(level="DEBUG")

    def test_point_od_get_origin_destination_distance_parquet(self, tmp_dir):
        tmp_parquet = tmp_dir / "parquet"

        h3_od.proximity.get_origin_destination_distance_parquet(
            origin_features=self.h3_centroid_features,
            destination_features=self.h3_centroid_features,
            parquet_path=tmp_parquet,
            origin_id_column="GRID_ID",
            destination_id_column="GRID_ID",
            network_dataset=network_dataset,
            travel_mode="Walking Distance",
            max_distance=500.0,
            search_distance=1.0,
            origin_batch_size=40000,
            partition_by_origin=True,
        )

        part_file_lst = list(tmp_parquet.glob("**/*.parquet"))

        assert len(part_file_lst) > 1
