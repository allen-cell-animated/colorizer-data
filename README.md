# colorizer-data

### Python utilities to prepare data for the [Timelapse Feature Explorer](https://github.com/allen-cell-animated/timelapse-colorizer)

[Timelapse Feature Explorer](https://github.com/allen-cell-animated/timelapse-colorizer) is a browser-based web app for viewing tracked segmented data.

Utilities are included in this repository to convert time-series data to the Timelapse Feature Explorer's format. You can read more about the data format specification here: [`DATA_FORMAT.md`](./documentation/DATA_FORMAT.md)

Example scripts are also included in this repository, based on some of our internal projects. You can edit these scripts to work with your datasets as part of a processing pipeline.

If using these example scripts, you'll need to run these commands in an environment with read access to the datasets and write access to the output directory. **The scripts also have their own external dependencies which must be installed separately**, as they are not direct dependencies of this package.

## Installation

```
# pip
pip install git+https://github.com/allen-cell-animated/colorizer-data.git@v1.3.0

# requirements.txt
colorizer_data @ git+https://github.com/allen-cell-animated/colorizer-data.git@v1.3.0
```

To install a different version, replace the end of the URL with a specific version or branch, like `@vX.X.X` or `@{branch-name}`.

## Example Usage

This is a simplified example. See our [example scripts](./colorizer_data/bin/example_scripts/) for complete working code!

```python
import pandas as pd
from colorizer_data.writer import (
    ColorizerDatasetWriter,
    ColorizerMetadata,
    FeatureInfo,
    FeatureType,
)
from colorizer_data.utils import (
    configureLogging,
    generate_frame_paths,
    remap_segmented_image,
    make_bounding_box_array,
)

# Open a dataset
data: pd.DataFrame = pd.read_csv("dataset.csv")
output_dir = "./data"

configureLogging(output_dir)
writer = ColorizerDatasetWriter(output_dir, data)

# Write data and features from the dataset
writer.write_data(
    tracks=data["tracks"],
    times=data["times"],
    outliers=data["outliers"],
    centroids_x=data["centroids x"],
    centroids_y=data["centroids y"]
)
writer.write_feature(
    data["feature1"].to_numpy(),
    FeatureInfo(label="My Feature", type=FeatureType.CONTINUOUS)
)
writer.write_feature(
    data["feature2"].to_numpy(),
    FeatureInfo(label="My Other Feature", type=FeatureType.CATEGORICAL, categories=["A", "B", "C"])
)

# Write frames and bounding boxes
grouped_data = data.groupby("time")
bounds_arr = make_bounding_box_array(grouped_data)
for group_name, frame in grouped_data:
    row = frame.iloc[0]
    frame_number = row["time"]
    # Get segmentations as a 2D array
    seg2d: np.ndarray = row["2d-segmentation-image"]

    # Remap the 2D array and write to the dataset as an image
    seg_remapped, lut = remap_segmented_image(
        seg2d,
        frame,
        "object-ids",
    )
    writer.write_image(seg_remapped, frame_number)
    update_bounding_box_data(bounds_arr, seg_remapped)

writer.write_data(bounds=bound_arr)
writer.set_frame_paths(generate_frame_paths(len(grouped_data)))

# Write manifest and finish
metadata = ColorizerMetadata(
    frame_width=400,
    frame_height=300,
    frame_units="µm"
    frame_duration_sec=5,
)
writer.write_manifest(metadata=metadata)
```

## Developers

After cloning the repository, you can install the project in **editable mode** with **dev dependencies** by running the following `pip` command:

```cmd
pip install -e '.[dev]'
```

### Versioning

This package uses [semantic versioning](https://semver.org). All versions are tagged in this repository with `vX.Y.Z`, where `X`, `Y`, and `Z` correspond with major, minor, and patch version numbers. The API will be backwards-compatible within the same major version.

Contributing developers can update the version number using the [`bump-my-version`](https://github.com/callowayproject/bump-my-version) Python tool, which will automatically tag commits. You can learn more about [git tagging here.](https://git-scm.com/book/en/v2/Git-Basics-Tagging)

#### Basic Usage

```
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
