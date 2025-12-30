# h3-origin-destination-matrix

Create origin-destination (OD) matrices using H3 hexagonal indices and Esri Network Analyst for spatial analysis workflows.

## Overview

This project provides a Python toolkit for generating origin-destination matrices at scale using [H3 hexagonal spatial indexing](https://h3geo.org/) combined with Esri's Network Analyst extension. It's designed to streamline proximity analysis and accessibility modeling by computing distance or travel time relationships between H3 cell centroids across an area of interest.

### Key Features

- **H3-Based Analysis**: Leverage Uber's H3 hexagonal grid system for consistent, multi-resolution spatial analysis
- **Network Analysis Integration**: Utilize Esri Network Analyst to calculate real-world travel distances and times (walking, driving, etc.)
- **Scalable Processing**: Batch processing capabilities for handling large areas with thousands of origin-destination pairs
- **Distance Decay Modeling**: Built-in sigmoid functions for modeling accessibility decay (e.g., to transit stops)
- **Parquet Output**: Efficient storage and retrieval of OD matrices using Apache Parquet format
- **ArcGIS Pro Integration**: Includes ArcGIS Pro project (`.aprx`) and Python Toolbox for GUI-based workflows

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
- A network dataset for routing analysis

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/knu2xs/h3-origin-destination-matrix.git
   cd h3-origin-destination-matrix
   ```

2. Create the Conda environment:
   ```bash
   make env
   ```
   
   Or if using ArcGIS Pro's default environment:
   ```bash
   make env_clone
   ```

3. Start exploring:
   - **Python users**: Launch Jupyter Lab from the project root and explore `./notebooks`
   - **GIS users**: Open `./arcgis/h3-origin-destination-matrix.aprx` in ArcGIS Pro

### Quick Start Example

```python
import h3_od

# Generate OD matrix for an area of interest
h3_od.proximity.get_aoi_h3_origin_destination_distance_parquet(
    area_of_interest="path/to/aoi_polygon.shp",
    parquet_path="./data/processed/od_matrix",
    h3_resolution=9,
    network_dataset="path/to/network_dataset",
    travel_mode="Walking Time",
    max_distance=3.0,  # miles
    search_distance=100,  # feet
)
```


## Using Make - common commands

Based on the pattern provided in the [Cookiecutter Data Science template by Driven Data](https://drivendata.github.io/cookiecutter-data-science/), this project streamlines common workflows using the `make` command pattern.

- **`make env`** - Builds the Conda environment with all dependencies from `environment_dev.yml` and installs the local `h3_od` package in editable mode (`python -m pip install -e ./src`) for active development.

- **`make env_clone`** - Designed for ArcGIS Pro users. Clones the `arcgispro-py3` environment and installs dependencies from `environment_dev.yml` along with the local package, preserving compatibility with ArcGIS Pro's default environment.

- **`make docs`** - Builds Sphinx documentation from files in `./docsrc` and outputs to `./docs` for easy GitHub Pages publishing.

- **`make test`** - Activates the project environment and runs all tests in `./testing` using PyTest. Alternatively, use [TOX](https://tox.readthedocs.io) with the included `tox.ini` configuration (dependencies included in default requirements).

## Project Structure

```
├── arcgis/                  # ArcGIS Pro project and toolbox
├── config/                  # Configuration files
├── data/                    # Data directory (raw, interim, processed)
├── notebooks/               # Jupyter notebooks for exploration
├── scripts/                 # Processing scripts
│   ├── make_data_for_aoi.py    # General AOI-based processing
│   └── make_data_olympia.py    # Example workflow
├── src/
│   └── h3_od/              # Main Python package
│       ├── proximity.py        # OD matrix calculation functions
│       ├── distance_decay.py   # Distance decay modeling
│       └── utils/             # Utility functions
└── testing/                 # Unit tests
```

## Configuration

Edit `scripts/config.ini` to configure your analysis parameters:

```ini
[DEFAULT]
AOI_POLYGON = data/raw/aoi.shp
OUTPUT_OD_PARQUET = data/processed/od_matrix
NETWORK_DATASET = C:/Path/To/Network_ND
TRAVEL_MODE = Walking Time
SNAP_DISTANCE = 100
ORIGIN_BATCH_SIZE = 500
H3_RESOLUTION = 9
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
