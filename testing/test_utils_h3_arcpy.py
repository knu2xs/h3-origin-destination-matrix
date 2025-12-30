import arcpy
import pytest

from h3_od.utils import h3_arcpy

def test_get_arcpy_point_for_h3_index():
    h3_index = "8928d59118bffff"
    result = h3_arcpy.get_arcpy_point_for_h3_index(h3_index)

    assert isinstance(result, arcpy.PointGeometry)


def test_get_k_neighbors_str():
    origin_indices = ["8928d59118bffff", "8928d59114bffff", "8928d59ac23ffff"]
    k = 12

    result = h3_arcpy.get_k_neighbors(origin_indices, k)

    assert isinstance(result, list)
    assert len(result) > len(origin_indices)
    assert all(isinstance(idx, str) for idx in result)