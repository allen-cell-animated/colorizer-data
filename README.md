# colorizer-data

_**Python utilities to prepare data for the [Timelapse Feature Explorer](https://github.com/allen-cell-animated/timelapse-colorizer)**_

[Timelapse Feature Explorer](https://github.com/allen-cell-animated/timelapse-colorizer) is a browser-based web app for viewing tracked segmented data. This package provides utilities to convert time-series data to a format that can be read by the viewer.

**To start converting your own data, [follow our tutorial (`GETTING_STARTED.ipynb`)](./documentation/getting_started_guide/GETTING_STARTED.ipynb)!**

You can read more about the data format specification here: [`DATA_FORMAT.md`](./documentation/DATA_FORMAT.md)

Example scripts are also included in this repository, based on some of our internal projects. You can edit these scripts to work with your datasets as part of a processing pipeline.

If using these example scripts, you'll need to run these commands in an environment with read access to the datasets and write access to the output directory. **The scripts may have their own external dependencies which must be installed separately**, as they are not direct dependencies of this package.

## Installation

```cmd
# pip
pip install git+https://github.com/allen-cell-animated/colorizer-data.git@v1.6.2

# requirements.txt
colorizer_data @ git+https://github.com/allen-cell-animated/colorizer-data.git@v1.6.2
```

To install a different version, replace the end of the URL with a specific version or branch, like `@vX.X.X` or `@{branch-name}`.

## Example Usage

See the [Getting Started tutorial (`GETTING_STARTED.ipynb`)](./documentation/getting_started_guide/GETTING_STARTED.ipynb) for a detailed walkthrough on how to get your datasets
into the correct format for the Timelapse Feature Explorer.

```python
import pandas as pd
from io import StringIO
from colorizer_data import convert_colorizer_data

from pathlib import Path

# Open an example CSV dataset:
csv = """ID,Track,Time,X,Y,Continuous Feature,Discrete Feature,Categorical Feature,Outlier,Segmentation Image Path
0,1,0,50,50,0.5,0,A,0,frame_0.tiff
1,1,1,55,60,0.6,1,B,0,frame_1.tiff
2,2,0,60,70,0.7,2,C,0,frame_0.tiff
3,2,1,65,75,0.8,3,A,1,frame_1.tiff
"""
# Relative paths will be evaluated relative to the `source_dir` directory.
# So `frame_0.tiff` becomes `some/source/directory/frame_0.tiff`.
source_dir = Path("some/source/directory")
data: pd.DataFrame = pd.read_csv(StringIO(csv))
output_dir = Path("some/directory/my-dataset")

convert_colorizer_data(
    data,
    output_dir,
    source_dir=source_dir,
    object_id_column="ID",
    track_column="Track",
    times_column="Time",
    centroid_x_column="X",
    centroid_y_column="Y",
    image_column="Segmentation File Path",
    # Columns that aren't specified are automatically parsed as features,
    #  e.g. `Continuous Feature`, `Discrete Feature` and `Categorical Feature`.
)
```

A file structure like the following will be created in the specified output directory. Once uploaded to an HTTPS-accessible location, the Timelapse Feature Explorer can be pointed to the `manifest.json` file to load the dataset.

```txt
📂 some/directory/my-dataset/
  - 📄 manifest.json
  - 📄 outliers.parquet
  - 📄 tracks.parquet
  - 📄 times.parquet
  - 📄 centroids.parquet
  - 📄 bounds.parquet
  - 📕 feature_0.parquet  // Continuous Feature
  - 📗 feature_1.parquet  // Discrete Feature
  - 📘 feature_2.parquet  // Categorical Feature
  - 📷 frame_0.png
  - 📷 frame_1.png
```

Instructions on loading local files and advanced conversion configuration is described in the [Getting Started tutorial (`GETTING_STARTED.ipynb`)](./documentation/getting_started_guide/GETTING_STARTED.ipynb).

## Developers

After cloning the repository, you can install the project in **editable mode** with **dev dependencies** by running the following `pip` command:

```cmd
pip install -e '.[dev]'
```

### Versioning

This package uses [semantic versioning](https://semver.org). All versions are tagged in this repository with `vX.Y.Z`, where `X`, `Y`, and `Z` correspond with major, minor, and patch version numbers. The API will be backwards-compatible within the same major version.

Contributing developers can update the version number using the [`bump-my-version`](https://github.com/callowayproject/bump-my-version) Python tool, which will automatically tag commits. You can learn more about [git tagging here.](https://git-scm.com/book/en/v2/Git-Basics-Tagging)

#### Basic Usage

```txt
# Do a dry run and check the output before updating!
bump-my-version bump --dry-run -v [type]

bump-my-version bump -v [type]
git push origin [new tag]
```

The `type` should be either `major`, `minor`, or `patch`.

#### Example

If the current version is `v0.0.0`, bumping major versions will create the tag tag `v1.0.0`.

```cmd
bump-my-version bump --tag -v major
git push origin v1.0.0
```
