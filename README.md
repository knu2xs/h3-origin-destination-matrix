# h3-origin-destination-matrix

Create origin-destination (OD) cost matrices using H3 hexagonal indices and Esri Network Analyst for spatial analysis workflows.

## Overview

This project provides a Python toolkit for generating origin-destination matrices at scale using [H3 hexagonal spatial indexing](https://h3geo.org/) combined with Esri's Network Analyst extension (`arcpy.nax`). It's designed to streamline proximity analysis and accessibility modeling by computing distance or travel time relationships between H3 cell centroids across an area of interest.

### Key Features

- **H3-Based Analysis**: Leverage Uber's H3 hexagonal grid system for consistent, multi-resolution spatial analysis
- **Network Analysis Integration**: Utilize Esri Network Analyst (`arcpy.nax`) to calculate real-world travel distances and times (walking, driving, etc.)
- **Scalable Processing**: Batch processing with configurable origin and output batch sizes for handling large areas with thousands of origin-destination pairs
- **Distance Decay Modeling**: Built-in sigmoid functions for modeling accessibility decay (e.g., bus stops, light rail stations)
- **Parquet Output**: Efficient storage and retrieval of OD matrices using Apache Parquet format via PyArrow
- **YAML Configuration**: Environment-aware configuration system supporting multiple analysis profiles (e.g., `dev`, `olympia_walk`, `olympia_drive`)
- **ArcGIS Pro Integration**: Includes ArcGIS Pro project (`.aprx`) and Python Toolbox (`.tbx`) for GUI-based workflows

### Use Cases

- Accessibility analysis (e.g., transit stop catchment areas)
- Spatial interaction modeling
- Service area delineation
- Urban mobility studies
- Retail trade area analysis

## Getting Started

### Prerequisites

- ArcGIS Pro 3.x with Network Analyst extension
- Conda (included with ArcGIS Pro)
- A network dataset for routing analysis (e.g., StreetMap Premium)

### Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/knu2xs/h3-origin-destination-matrix.git
    cd h3-origin-destination-matrix
    ```

2. Create the Conda environment (clones the ArcGIS Pro `arcgispro-py3` environment and adds project dependencies):

    ```bash
    make env
    ```

3. Start exploring:
    - **Python users**: Launch Jupyter Lab with `make jupyter` and explore `./notebooks/`
    - **GIS users**: Open `./arcgis/h3-origin-destination-matrix.aprx` in ArcGIS Pro

### Quick Start Example

```python
import h3_od

# Generate OD matrix for an area of interest
h3_od.proximity.get_aoi_h3_origin_destination_distance_parquet(
    area_of_interest="data/raw/raw.gdb/my_aoi_polygon",
    h3_resolution=9,
    parquet_path="./data/processed/od_matrix/parquet",
    network_dataset="path/to/network_dataset",
    travel_mode="Walking Distance",
    max_distance=5.0,         # miles
    search_distance=1.0,      # miles — snap distance to nearest routable edge
    origin_batch_size=250,
)
```

## Make Commands

Based on the pattern provided in the [Cookiecutter Data Science template by Driven Data](https://drivendata.github.io/cookiecutter-data-science/), this project streamlines common workflows using the `make` command pattern (via `make.cmd` on Windows).

- **`make env`** - Clones the ArcGIS Pro `arcgispro-py3` Conda environment into `./env/`, installs additional dependencies from `environment.yml`, and installs the local `h3_od` package in editable mode (`pip install -e .`).
- **`make add_dependencies`** - Updates the existing environment with dependencies from `environment.yml` and reinstalls the local package (useful after pulling new dependency changes).
- **`make data`** - Runs the data pipeline via `scripts/make_data.py`.
- **`make docs`** - Builds MkDocs documentation from `./docsrc/` and outputs to `./docs/` for publishing.
- **`make docserve`** - Runs a live MkDocs documentation server for local preview.
- **`make test`** - Runs all tests in `./testing/` using PyTest.
- **`make jupyter`** - Launches Jupyter Lab with remote connection support.
- **`make wheel`** - Builds a distributable wheel package.
- **`make black`** - Formats source code with Black.
- **`make pkg_dependencies`** - Collects package dependencies into `./dependencies/` for distribution.

## Project Structure

```
├── arcgis/                      # ArcGIS Pro project (.aprx), toolbox (.tbx), styles
├── config/
│   ├── config.yml               # Main project configuration (environment-aware)
│   └── secrets_template.yml     # Template for credentials (copy to secrets.yml)
├── data/
│   ├── raw/                     # Raw input data and AOI geometries
│   ├── interim/                 # Intermediate processing outputs
│   ├── processed/               # Final OD matrix Parquet datasets
│   └── external/                # External reference data
├── dependencies/                # Vendored dependencies (e.g., h3) for distribution
├── docsrc/                      # MkDocs documentation source
├── notebooks/                   # Jupyter notebooks for exploration and analysis
├── scripts/
│   ├── config.ini               # Script-level configuration (INI format)
│   ├── make_data.py             # Main data pipeline entry point
│   ├── make_data_for_aoi.py     # General AOI-based OD matrix generation
│   ├── make_data_olympia.py     # Olympia-specific example workflow
│   ├── lookup_h3_od.py          # H3 OD lookup utility
│   └── get_package_dependencies.py  # Dependency packaging script
├── src/
│   └── h3_od/                   # Main Python package
│       ├── config.py            # YAML-based configuration loader
│       ├── proximity.py         # OD matrix calculation functions
│       ├── distance_decay.py    # Sigmoid distance decay modeling
│       └── utils/               # Utility modules (H3-ArcPy, logging, geometry, PyArrow)
└── testing/                     # PyTest unit tests
```

## Configuration

### YAML Configuration (recommended)

The primary configuration lives in `config/config.yml` and supports multiple named environments. Set the active environment by changing the `ENVIRONMENT` constant in `src/h3_od/config.py` or by setting the `PROJECT_ENV` environment variable.

```yaml
environments:
  dev:
    logging:
      level: DEBUG
    h3:
      resolution: 8
    network:
      dataset: "path/to/network_dataset"
      travel_mode: "Rural Driving Distance"
      snap_distance: 0.5
      max_distance: 180.0
      origin_batch_size: 60
    data:
      aoi_polygon: "data/raw/raw.gdb/my_aoi"
      output_od_parquet: "data/processed/od_matrix/parquet"
```

Access configuration values in code:

```python
from h3_od.config import config, ENVIRONMENT

log_level = config.logging.level
resolution = config.h3.resolution
travel_mode = config.network.travel_mode
```

### Secrets

Copy `config/secrets_template.yml` to `config/secrets.yml` and fill in credentials. This file is not committed to version control.

### Script-Level Configuration (INI)

The `scripts/config.ini` file provides INI-style configuration with named sections for use with processing scripts such as `make_data_for_aoi.py`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.
