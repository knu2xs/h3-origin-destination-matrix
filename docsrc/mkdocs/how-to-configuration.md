# How To: Project Configuration

This project uses YAML-based configuration files stored in the `config/` directory.
Settings are loaded through the `h3_od.config` module, which provides a singleton
pattern so files are parsed once and reused across all modules and scripts.

---

## File Layout

```text
config/
    config.yml              # project settings (committed to version control)
    secrets_template.yml    # template for credentials (committed)
    secrets.yml             # actual credentials (NOT committed — .gitignore)
```

| File | Purpose | Version Control |
|---|---|---|
| `config.yml` | Project and environment-specific settings | **Yes** |
| `secrets_template.yml` | Documents required credential keys | **Yes** |
| `secrets.yml` | Actual credential values | **No** |

---

## Quick Start

### 1. Set Up Secrets

Copy the template and fill in your values:

```powershell
Copy-Item config/secrets_template.yml config/secrets.yml
```

Edit `config/secrets.yml` with your credentials:

```yaml
esri:
  gis_url: "https://your-org.maps.arcgis.com"
  gis_username: "your_username"
  gis_password: "your_password"
```

!!! danger
    Never commit `secrets.yml` to version control. It is already listed in
    `.gitignore`.

### 2. Import Configuration in Python

```python
from h3_od.config import config, secrets, ENVIRONMENT
```

---

## Accessing Settings

The `config` object supports both **dot-notation** and **dict-style** access:

```python
from h3_od.config import config

# dot-notation
log_level = config.logging.level
travel_mode = config.network.travel_mode

# dict-style
aoi = config["data"]["aoi_polygon"]

# .get() with a default
h3_features = config.data.get("h3_features", None)
```

Secrets work the same way:

```python
from h3_od.config import secrets

gis_url = secrets.esri.gis_url
```

### Convenience Exports

The most commonly used values are also exported as module-level constants for
quick access:

```python
from h3_od.config import (
    LOG_LEVEL,          # str   — e.g. "DEBUG"
    H3_RESOLUTION,      # int   — e.g. 8
    NETWORK_DATASET,    # str   — path to the network dataset
    TRAVEL_MODE,        # str   — e.g. "Rural Driving Distance"
    SNAP_DISTANCE,      # float — e.g. 0.5
    MAX_DISTANCE,       # float — e.g. 180.0
    ORIGIN_BATCH_SIZE,  # int   — e.g. 60
    AOI_POLYGON,        # str   — path to the AOI polygon
    OUTPUT_OD_PARQUET,  # str   — path to the output parquet directory
)
```

---

## Environments

Settings are organized under the `environments` key in `config.yml`. Each
environment is a named block (e.g. `dev`, `olympia_walk`, `olympia_drive`) whose
values are deep-merged on top of the shared, top-level keys.

### Listing Available Environments

```python
from h3_od.config import get_available_environments

print(get_available_environments())
# ['dev', 'olympia_drive', 'olympia_walk']
```

### Switching Environments

There are three ways to select an environment, listed in order of precedence:

#### Option A — Environment Variable (recommended for scripts)

Set `PROJECT_ENV` **before** importing `h3_od`:

```python
import os
os.environ["PROJECT_ENV"] = "olympia_walk"

# now import — config will load the olympia_walk environment
from h3_od.config import config
```

Or from the command line:

```powershell
$env:PROJECT_ENV = "olympia_walk"
python scripts/make_data_olympia.py
```

#### Option B — Change the Default

Edit the `ENVIRONMENT` constant in `src/h3_od/config.py`:

```python
ENVIRONMENT: str = os.environ.get("PROJECT_ENV", "dev")  # change "dev" to your default
```

#### Option C — Load Programmatically

Use `load_config()` directly when you need a non-default environment without
changing any global state:

```python
from h3_od.config import load_config

cfg = load_config(environment="olympia_drive")
print(cfg.network.travel_mode)  # "Rural Driving Time"
```

---

## Adding a New Environment

Add a new block under `environments` in `config/config.yml`:

```yaml
environments:

  # ...existing environments...

  my_new_env:
    logging:
      level: INFO
    h3:
      resolution: 10
    network:
      dataset: "path/to/network_dataset"
      travel_mode: "Walking Distance"
      snap_distance: 0.25
      max_distance: 5.0
      origin_batch_size: 200
    data:
      aoi_polygon: "data/raw/raw.gdb/my_aoi"
      output_od_parquet: "data/processed/my_output/parquet"
```

No Python code changes are required — the new environment is immediately
available via `PROJECT_ENV=my_new_env` or `load_config(environment="my_new_env")`.

---

## Configuration Structure Reference

Below is the full schema of `config.yml`:

```yaml
project:
  name: "h3-origin-destination-matrix"
  title: "H3 Origin-Destination Matrix"
  description: "Project description."

environments:
  <environment_name>:
    logging:
      level: DEBUG | INFO | WARNING | ERROR | CRITICAL
    h3:
      resolution: <int>          # H3 resolution level (0–15)
    network:
      dataset: <str>             # path to ArcGIS network dataset
      travel_mode: <str>         # network analyst travel mode name
      snap_distance: <float>     # search tolerance for snapping to network
      max_distance: <float>      # maximum travel cost cutoff
      origin_batch_size: <int>   # origins per solve batch
    data:
      aoi_polygon: <str>         # path to area-of-interest feature class
      output_od_parquet: <str>   # output directory for parquet files
      h3_features: <str>         # (optional) pre-built H3 features
```

---

## Using Config in Scripts

Scripts set the environment **before** importing `h3_od`, then pull values from
the config module. Here is the typical pattern:

```python
"""Example script using project configuration."""
import datetime
import os
from pathlib import Path
import importlib.util
import sys

# select environment before importing h3_od
if "PROJECT_ENV" not in os.environ:
    os.environ["PROJECT_ENV"] = "dev"

# ensure h3_od is importable
dir_prj = Path(__file__).parent.parent

if importlib.util.find_spec("h3_od") is None:
    sys.path.insert(0, str(dir_prj / "src"))

import h3_od
from h3_od.utils import get_logger
from h3_od.config import (
    LOG_LEVEL,
    H3_RESOLUTION,
    AOI_POLYGON,
    OUTPUT_OD_PARQUET,
    NETWORK_DATASET,
    TRAVEL_MODE,
    MAX_DISTANCE,
    SNAP_DISTANCE,
    ORIGIN_BATCH_SIZE,
)

if __name__ == "__main__":

    aoi_features = Path(AOI_POLYGON)
    od_parquet = Path(OUTPUT_OD_PARQUET)
    network_dataset = Path(NETWORK_DATASET)

    dt_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
    log_pth = od_parquet.parent / f"{Path(__file__).stem}_{dt_str}.log"

    logger = get_logger(
        logger_name=Path(__file__).stem,
        level=LOG_LEVEL,
        logfile_path=log_pth,
    )

    logger.info(f"Running with environment: {h3_od.config.ENVIRONMENT}")

    h3_od.proximity.get_aoi_h3_origin_destination_distance_parquet(
        area_of_interest=aoi_features,
        parquet_path=od_parquet,
        h3_resolution=int(H3_RESOLUTION),
        network_dataset=network_dataset,
        travel_mode=TRAVEL_MODE,
        max_distance=MAX_DISTANCE,
        search_distance=SNAP_DISTANCE,
        origin_batch_size=ORIGIN_BATCH_SIZE,
    )
```

!!! tip
    Keep all processing logic inside `if __name__ == "__main__":` so the script
    can be safely imported without triggering side effects.
