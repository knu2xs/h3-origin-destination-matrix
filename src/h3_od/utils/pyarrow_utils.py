from pathlib import Path

import pyarrow.dataset as ds

__all__ = ["get_pyarrow_dataset_from_parquet"]


# load PyArrow dataset
def get_pyarrow_dataset_from_parquet(dataset_path: Path) -> ds.Dataset:
    """Simple wrapper to quickly load a parquet dataset into a PyArrow Dataset."""
    dataset = ds.dataset(
        dataset_path,
        format="parquet",
        partitioning="hive",
    )
    return dataset
