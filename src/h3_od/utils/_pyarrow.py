from pathlib import Path

import pyarrow.dataset as ds

__all__ = ["get_pyarrow_dataset_from_parquet"]


# load PyArrow dataset
def get_pyarrow_dataset_from_parquet(dataset_path: Path) -> ds.Dataset:
    """
    Load a Parquet dataset from disk into a PyArrow Dataset.

    Convenience wrapper around ``pyarrow.dataset.dataset`` that applies Hive-style
    partitioning, which is the partitioning scheme used by the origin-destination
    Parquet outputs in this project.

    Args:
        dataset_path: Path to the Parquet file or directory containing the
            partitioned Parquet dataset.

    Returns:
        A ``pyarrow.dataset.Dataset`` backed by the Parquet data at the given path.
    """
    dataset = ds.dataset(
        dataset_path,
        format="parquet",
        partitioning="hive",
    )
    return dataset
