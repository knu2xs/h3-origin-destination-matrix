# Code Review: `h3_od` Package

**Date:** March 2, 2026

## Summary

The package provides H3-based origin-destination matrix computation on top of ArcPy Network Analyst.
The code is generally well-structured and demonstrates strong domain knowledge; however, there are
**1 critical bug**, several **moderate issues**, and numerous **style/standards violations** against
the AGENTS.md guidelines.

---

## Critical — Bug

### 1. Index out-of-bounds typo in `proximity.py`

**File:** `src/h3_od/proximity.py` (line ~770)

```python
h3_origin = h3.latlng_to_cell(
    origin_coordinates[1], origin_coordinates[9], h3_resolution  # [9] should be [0]
)
```

`origin_coordinates[9]` will raise an `IndexError` at runtime for any normal coordinate pair. This
should be `origin_coordinates[0]`.

---

## High — Bugs & Correctness

### 2. `isinstance` with a list instead of a tuple

**File:** `src/h3_od/proximity.py` (line ~122)

```python
if not isinstance(geom, [arcpy.PointGeometry, arcpy.Polygon]):
```

`isinstance()` requires a **tuple** for multiple types, not a list. This will raise a `TypeError`.

**Fix:**

```python
if not isinstance(geom, (arcpy.PointGeometry, arcpy.Polygon)):
```

### 3. Unused parameter `destination_h3_indices`

**File:** `src/h3_od/proximity.py` (line ~340)

`get_origin_destination_parquet` accepts `destination_h3_indices` but never references it in the
function body. Destinations are instead computed from `get_k_neighbors`. This is confusing — either
use the parameter or remove it.

### 4. Undefined variable `origin_idx` in "no valid routes" branch

**File:** `src/h3_od/proximity.py` (lines ~505–520)

When `len(valid_origin_id_set) == 0`, the code references `origin_idx` to build a placeholder row,
but `origin_idx` is the loop variable from the batch origin iteration (the
`for origin_idx in batch_origin_lst` cursor loop), and this is a *different* scope. It should likely
iterate `batch_origin_lst` to record all unroutable origins.

### 5. Inconsistent Parquet partition columns

**File:** `src/h3_od/proximity.py` (lines ~495 vs ~524)

The success path uses `partition_cols=["h3_resolution", "origin_id"]` but the empty-results
placeholder path uses `partition_cols=["origin_id"]`, creating an inconsistent Parquet directory
structure.

### 6. `tmp_gdb` fixture references wrong parameter

**File:** `testing/test_h3_od_proximity.py` (line ~46)

```python
def tmp_gdb(temp_dir) -> Path:  # should be tmp_dir, not temp_dir
```

This will cause a pytest error since there is no `temp_dir` fixture; the correct fixture is
`tmp_dir`.

---

## Moderate — AGENTS.md Standards Violations

### Missing Docstrings (Google Style)

- **7.** `validate_h3_index_list` (`proximity.py`, line ~54) — no docstring at all.
- **8.** `validate_origin_destination_inputs` (`proximity.py`, line ~86) — no docstring.
- **9.** `get_origin_destination_oid_col` (`proximity.py`, line ~164) — no docstring.
- **10.** `preprocess_h3_index` (`utils/h3_arcgis.py`, line ~13) — no `Args:`/`Returns:` sections.
- **11.** `get_pyarrow_dataset_from_parquet` (`utils/_pyarrow.py`, line ~9) — one-liner description
  only, no `Args:`/`Returns:` sections.

### Incomplete Docstrings

- **12.** `get_sigmoid_distance_decay_index` — `steepness` and `offset` params have **empty
  descriptions** in `Args:` (`distance_decay.py`, lines ~27–28).
- **13.** All three `distance_decay` functions are missing `Returns:` sections.
- **14.** `get_network_travel_modes` has an empty `Returns:` section (`proximity.py`, line ~228).
- **15.** `get_origin_destination_cost_matrix_solver` docstring opens with a blank line (empty brief)
  (`proximity.py`, line ~250).
- **16.** `polygon_to_h3_indices` has an empty `Returns:` section (`utils/h3_arcpy.py`, line ~201).
- **17.** `get_logger` docstring says `propagate` default is `False` but the actual signature default
  is `True` (`utils/_logging.py`, line ~123).

### Wrong Admonition Syntax (`!! note` instead of `!!! note`)

**18.** The following files use `!! note` (two bangs) which is not valid MkDocs syntax — should be
`!!! note` (three bangs):

- `utils/h3_arcpy.py` — `transpose_coordinate_list` (line ~94)
- `utils/h3_arcpy.py` — `get_h3_polygon_from_geojson` (line ~117)
- `utils/h3_arcpy.py` — `get_h3_indices_for_esri_polygon` (line ~242)
- `proximity.py` — `get_network_dataset_layer` (line ~200)

### Sphinx-style Directives Instead of Triple Backticks

**19.** AGENTS.md requires triple backticks for code examples, but `_logging.py` uses
`.. code-block:: python` (Sphinx-style) in:

- `ArcpyHandler` class docstring (line ~27)
- `configure_logging` (line ~213)
- `format_pandas_for_logging` (line ~275)

---

## Moderate — Code Quality & Best Practices

### 20. Module-level side effects on import

**File:** `src/h3_od/proximity.py` (lines ~41–43)

```python
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("network")
```

These mutate global ArcPy state when the module is merely imported. They should be deferred to
actual function calls or a setup function.

### 21. Hardcoded country code

**File:** `src/h3_od/proximity.py` (line ~46)

```python
iso2 = "US"
```

AGENTS.md says *"Avoid hardcoding values; use configuration files or environment variables."* This
should be a parameter or pulled from config.

### 22. Use of `os.path.join` instead of `pathlib.Path`

**File:** `src/h3_od/proximity.py` (line ~156)

```python
os.path.join(arcpy.env.scratchGDB, f"f_{uuid.uuid4().hex}")
```

AGENTS.md specifies `pathlib.Path` for all path manipulation.

### 23. `get_origin_destination_parquet` is ~150 lines

**File:** `src/h3_od/proximity.py` (lines ~335–540)

AGENTS.md says *"Keep functions small and focused on a single responsibility."* This function
handles batching, loading origins/destinations, solving, result extraction, and Parquet I/O.
Consider extracting at least: `_solve_batch()`, `_export_results_to_parquet()`.

### 24. `handle_features` decorator missing `functools.wraps`

**File:** `src/h3_od/utils/h3_arcpy.py` (line ~35)

Without `@functools.wraps(fn)`, decorated functions lose their `__name__`, `__doc__`, and other
metadata, breaking introspection and documentation generation.

### 25. Duplicate handler accumulation in `configure_logging`

**File:** `src/h3_od/utils/_logging.py` (line ~198)

Unlike `get_logger` which calls `logger.handlers.clear()`, `configure_logging` does not clear
existing handlers. Repeated calls will add duplicate handlers and produce duplicate log output.

### 26. Module docstring placement in `arcpy_geometry.py`

**File:** `src/h3_od/utils/arcpy_geometry.py` (lines ~5–8)

The module docstring appears after dunder variables, which means Python won't treat it as the
module's `__doc__`. The docstring should come before the dunders.

### 27. Misleading function name `get_arcpy_point_for_h3_index` in `h3_arcgis.py`

**File:** `src/h3_od/utils/h3_arcgis.py` (line ~53)

The function returns `arcgis.geometry.Point` (ArcGIS Python API), not `arcpy.PointGeometry`. The
name with "arcpy" prefix is misleading. Consider renaming to `get_arcgis_point_for_h3_index`.

### 28. Fragile `__all__` mutation

**File:** `src/h3_od/utils/_logging.py` (lines ~11–14)

```python
__all__ = ["configure_logging"]
if ...:
    __all__ = __all__ + ["format_pandas_for_logging"]
```

Conditional `__all__` mutation is fragile. Consider always listing all public names and raising
`ImportError` on use if the dependency is missing.

---

## Low — Style & Minor Issues

- **29.** Several functions in `proximity.py` use **early returns** (e.g.,
  `get_nearest_origin_destination_neighbor`, `get_h3_neighbors`). AGENTS.md says *"Avoid early
  returns in functions."*
- **30.** `_logging.py` inconsistently uses `find_spec("arcpy")` in `get_logger` (line ~173) while
  the imported `has_arcpy` variable is available and used elsewhere in the same file.
- **31.** Tests use **hardcoded absolute path** `D:\data\ba_data\...`
  (`testing/test_h3_od_proximity.py`, line ~32), making tests non-portable. Consider using
  environment variables or a config fixture.
- **32.** Tests use `sys.path.insert` (`testing/test_h3_od_proximity.py`, lines ~17–18) instead of
  installing the package in editable mode (`pip install -e .`) and relying on proper packaging.
- **33.** Very limited test coverage — no tests exist for `distance_decay.py`, `h3_arcgis.py`,
  `_logging.py`, or `_pyarrow.py`. AGENTS.md says *"Write unit tests for new functionality."*
- **34.** Test assertions are minimal — they only verify file counts (`len(part_file_lst) > 1`), not
  data correctness (schema, row counts, value ranges).
- **35.** `arcpy_geometry.py` is an empty module (no functions, only metadata dunders). It should
  either be populated or removed.
- **36.** `typing.Union[float, int]` in `distance_decay.py` could be simplified to `float` since
  Python's `float` accepts `int` in practice, or use `int | float` with modern syntax.

---

## Summary Table

| Severity     | Count | Categories                                                                        |
|--------------|-------|-----------------------------------------------------------------------------------|
| **Critical** | 1     | Index bug (`[9]`)                                                                 |
| **High**     | 5     | `isinstance` bug, unused param, undefined var, partition inconsistency, test fixture |
| **Moderate** | 9     | Missing/incomplete docstrings, wrong admonition syntax, Sphinx directives         |
| **Moderate** | 9     | Side effects on import, hardcoded values, long functions, `os.path`, missing `wraps` |
| **Low**      | 8     | Early returns, test portability, coverage gaps, empty module                      |

---

## Recommended Priority

1. Fix the **critical bug** (#1) and the `isinstance` bug (#2) immediately — these will crash at
   runtime.
2. Address the undefined variable (#4) and unused parameter (#3) — these indicate logic errors.
3. Extract the long `get_origin_destination_parquet` function and move module-level side effects
   (#20–23).
4. Add missing docstrings and fix admonition syntax for documentation generation.
5. Expand test coverage, especially for pure-Python modules like `distance_decay` and `h3_arcgis`
   which don't require ArcPy fixtures.
