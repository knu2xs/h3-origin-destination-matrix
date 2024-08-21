from pathlib import Path
import shutil
from tempfile import mkdtemp

import arcpy.management
import pytest


@pytest.fixture(scope="function")
def tmp_dir():
    temp_dir = Path(mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def tmp_gdb(temp_dir):
    temp_gdb = arcpy.management.CreateFileGDB(str(temp_dir), "temp.gdb")[0]
    temp_gdb = Path(temp_gdb)
    yield temp_gdb
    arcpy.management.Delete(temp_gdb)
