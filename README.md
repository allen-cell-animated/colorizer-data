# colorizer-data

#### A Python package of utilities for preparing data for the Timelapse Colorizer app.

Utilities are included in this repository to convert time-series data to the Timelapse Colorizer's format. [You can read more about the data format specification here.](./documentation/DATA_FORMAT.md)

We provide utilities for writing data to our specification and example scripts for some AICS internal projects' data sets. Each project (AICS nuclear morphogenesis, AICS EMT, etc) will need their own data conversion scripts.

For loading of datasets to work correctly, you'll need to run these commands from a device that has access to Allen Institute's on-premises data storage. If running off of shared resources, remember to initialize your virtual environment first! This may look like `conda activate {my_env}`.

## Installation

To install the package, you can either install it via `pip` or in a `requirements.txt` file.

```
# pip
pip install git+https://github.com/allen-cell-animated/colorizer-data.git

# requirements.txt
colorizer-data @ git+https://github.com/allen-cell-animated/colorizer-data.git
```

To install a specific tagged version or branch, add `@{version tag/branch name}` to the end of the git link.

```
pip install git+https://github.com/allen-cell-animated/colorizer-data.git@my-branch-name
pip install git+https://github.com/allen-cell-animated/colorizer-data.git@1.0.0
```

## Example Usage

See our example data scripts in [`timelapse-colorizer-data`](./timelapse-colorizer-data/) for complete working examples!

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
    outliers=data["outliers"]
)
writer.write_feature(
    data["feature1"].to_numpy(),
    FeatureInfo(label="My Feature", type=FeatureType.CONTINUOUS)
)
writer.write_feature(
    data["feature2"].to_numpy(),
    FeatureInfo(label="My Other Feature", type=FeatureType.CATEGORICAL, categories=["A", "B", "C"])
)

# Write frames
grouped_data = data.groupby("time")
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

writer.set_frame_paths(generate_frame_paths(len(grouped_data)))

# Write manifest and finish
writer.write_manifest()
```
