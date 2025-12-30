#!/usr/bin/env python
# coding: utf-8
"""
Copy package dependencies listed in pyproject.toml to ./dependencies directory.

This script reads the dependencies from pyproject.toml, locates their installed
locations, and copies them to a local dependencies directory for archival or
offline use.
"""
import shutil
import sys
import tomllib
from pathlib import Path
from importlib.metadata import PackageNotFoundError
from importlib.util import find_spec

# Add src/ to sys.path so h3_od can be imported
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Ensure h3_od package is importable
if find_spec("h3_od") is None:
    raise ImportError("h3_od package is required but not found. Please install it before running this script.")

# constants
LOGGING_LEVEL = "DEBUG"


# configure logging
from h3_od.utils import get_logger
logger = get_logger(Path(__file__).stem, level=LOGGING_LEVEL, add_stream_handler=True)


def get_package_path(package_name: str) -> Path:
    """
    Get the installation location of a package.
    
    Args:
        package_name: Name of the package (e.g., 'h3', 'arcgis')
    
    Returns:
        Path to the package installation directory
    
    Raises:
        PackageNotFoundError: If the package is not installed
    """
    # Remove version specifiers and extras (e.g., "dask[dataframe]" -> "dask")
    clean_name = package_name.split('[')[0].split('=')[0].split('>')[0].split('<')[0].strip()

    # locate the package
    spec = find_spec(clean_name)

    # get package path
    if spec and spec.origin:
        pkg_pth = Path(spec.origin).parent

    else:
        raise PackageNotFoundError(f"Package '{clean_name}' is not installed")

    return pkg_pth


def parse_dependencies_from_pyproject(pyproject_path: Path) -> list[str]:
    """
    Parse dependencies from pyproject.toml file.
    
    Args:
        pyproject_path: Path to pyproject.toml file
    
    Returns:
        List of dependency package names
    """
    # Read the pyproject.toml file
    with open(pyproject_path, 'rb') as f:

        # Load TOML data from pyproject.toml
        data = tomllib.load(f)
    
    # Extract dependencies
    dependencies = data.get('project', {}).get('dependencies', [])

    return dependencies


def copy_package_to_dependencies(package_name: str, dest_dir: Path) -> None:
    """
    Copy a package from its installed location to the dependencies directory.
    
    Args:
        package_name: Name of the package to copy
        dest_dir: Destination directory for the package
    """
    # Clean package name
    clean_name = package_name.split('[')[0].split('=')[0].split('>')[0].split('<')[0].strip()
    
    logger.debug(f"Processing package: {clean_name}")
    
    try:
        pkg_location = get_package_path(clean_name)
        
        # Create destination path
        dest_path = dest_dir / clean_name
        
        # Check if package directory exists
        if pkg_location.exists():

            # Remove existing if present
            if dest_path.exists():
                logger.debug(f"Removing existing copy at {dest_path}")
                shutil.rmtree(dest_path)
            
            # Copy the package
            logger.debug(f"Copying from {pkg_location} to {dest_path}")
            shutil.copytree(pkg_location, dest_path, symlinks=False, 
                          ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
            logger.info(f"Successfully copied {clean_name}")
        else:
            logger.warning(f"⚠ Warning: Package location not found at {pkg_location}")
            
    except PackageNotFoundError as e:
        logger.error(f"✗ Error: {e}")
    except Exception as e:
        logger.error(f"✗ Error copying {clean_name}: {e}")


def main():
    """Main function to copy all dependencies."""
    # Get paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    pyproject_path = project_dir / 'pyproject.toml'
    dependencies_dir = project_dir / 'dependencies'
    
    # Check if pyproject.toml exists
    if not pyproject_path.exists():
        logger.error(f"Error: pyproject.toml not found at {pyproject_path}")
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")
    
    # Create dependencies directory if it doesn't exist
    dependencies_dir.mkdir(exist_ok=True)
    logger.info(f"Dependencies will be copied to: {dependencies_dir}\n")
    
    # Parse dependencies from pyproject.toml
    try:
        # get list of dependencies
        dependencies = parse_dependencies_from_pyproject(pyproject_path)

        logger.info(f"Found {len(dependencies)} dependencies in pyproject.toml: {dependencies}")
    
    except Exception as e:
        logger.error(f"Error parsing pyproject.toml: {e}")
        raise e
    
    # Copy each dependency
    success_count = 0
    
    for package in dependencies:
        try:
            copy_package_to_dependencies(package, dependencies_dir)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to copy {package}: {e}")
    
    # Summary
    logger.info(f"Summary: {success_count}/{len(dependencies)} packages copied successfully to {dependencies_dir}")


if __name__ == '__main__':
    main()
