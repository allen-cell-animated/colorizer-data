# Tutorial: How to process data for Timelapse Feature Explorer

The Timelapse Feature Explorer is a web-based application designed for the interactive visualization and analysis of segmented time-series microscopy data! Data needs to be processed into a specific format to be loaded into the viewer. This tutorial will guide you through the process of preparing your data for the Timelapse Feature Explorer.

## Prerequisites

From a command terminal, clone this repository and install the dependencies for this tutorial. This will install the necessary dependencies for the example scripts and the latest release of `colorizer-data`.(You may want to do this from a virtual environment.)

```bash
git clone https://github.com/allen-cell-animated/colorizer-data.git
cd colorizer-data/documentation/getting_started_guide

pip install -r ./requirements.txt
```

## Processing your data

Timelapse Feature Explorer reads data in the format specified by the [`DATA_FORMAT`](./documentation/DATA_FORMAT.md) document. `colorizer-data` provides utilities for working with the data format.

### Raw data format

For this tutorial, there is a sample dataset included in the `getting_started_guide` directory. This dataset is a simplified example of raw, pre-processed segmentation data. The dataset is a CSV file with columns for track IDs, times, centroids, features (volume/height), and the paths of segmentation files.

Your files may be in a different format, but the general best-practices are to:

1. Make every object a row in the table.
2. Include a column for the object's track ID, time, centroid, and any features you want to visualize.

**Example:**
| object_id | track_id | time | centroid_x | centroid_y | volume | height | segmentation_path |
| ----------- | ---------- | ------ | ------------ | ------------ | -------- | -------- | ------------------- |
| 0 | 0 | 0 | 54 | 23 | 102.3 | 5 | ./raw_dataset/0.ome.tiff |
| 1 | 0 | 1 | 57 | 25 | 104.8 | 6 | ./raw_dataset/1.ome.tiff |
| 2 | 0 | 2 | 60 | 24 | 109.9 | 8 | ./raw_dataset/2.ome.tiff |
| 3 | 1 | 1 | 10 | 34 | 34.23 | 3 | ./raw_dataset/1.ome.tiff |
| 4 | 1 | 2 | 12 | 34 | 35.60 | 5 | ./raw_dataset/2.ome.tiff |
| 5 | 2 | 0 | 78 | 79 | 78.65 | 5 | ./raw_dataset/0.ome.tiff |
| ... | ... | ... | ... | ... | ... | ... | ... |

> **_NOTE:_** Note that one segmentation path exists for each timepoint, and object IDs start at zero for each frame/timepoint. Timelapse Feature Explorer requires that every object ID be unique across all timepoints, so we will need to remap the object IDs later.
>
> Also, one segmentation path exists for each timepoint.


### Processing script

Create a Python file named `process_data.py` and open it in your favorite text editor. Paste the following steps into the file.

1. Import dependencies and load the dataset into a pandas DataFrame.
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

data: pd.DataFrame = pd.read_csv("raw_dataset.csv")
output_dir = "./data"
```

2. Format the 

> **_NOTE:_** If you are working with 3D volume segmentations rather than 2D segmentations, you will need to flatten your segmentations into 2D images. A maximum intensity projection is common.

## Viewing the dataset

Now that the dataset is processed, we can view it in the Timelapse Feature Explorer!

### Hosting datasets

Timelapse Feature Explorer is designed to load datasets hosted in a cloud storage service or web server. This data will be fetched by the client's web browser, so it must be accessible via URL from the browser.

> **_NOTE:_** To use a dataset with our public build of Timelapse Feature Explorer, the dataset must be accessible using the HTTPS protocol (e.g., `https://example.com/your-dataset/`). If you need to use HTTP, you can run a local instance of the viewer. See the [Timelapse Feature Explorer documentation](https://github.com/allen-cell-animated/timelapse-colorizer#installation) for instructions on downloading and running the viewer.

For this tutorial, we've hosted a copy of the processed example dataset on GitHub. You can access it at this URL: [https://raw.githubusercontent.com/allen-cell-animated/colorizer-data/main/documentation/getting_started_guide/processed_dataset](https://raw.githubusercontent.com/allen-cell-animated/colorizer-data/main/documentation/getting_started_guide/processed_dataset)

### Opening your dataset

Open Timelapse Feature Explorer at [https://timelapse.allencell.org](https://timelapse.allencell.org).

![The Load button on the Timelapse Feature Explorer header, next to the Help dropdown.](./documentation/getting_started_guide/assets/load-button.png)

Click the **Load** in the header and paste in the following URL:

```
https://raw.githubusercontent.com/allen-cell-animated/colorizer-data/main/documentation/getting_started_guide/processed_dataset
```

> **_NOTE:_** You can either provide the URL of the directory containing a `manifest.json` or the full URL path of a `.json` file that follows the [manifest specification](./documentation/DATA_FORMAT.md#1-metadata). We recommend specifying the full URL path that includes the `manifest.json`.

Click **Load** in the popup menu to load the dataset. The viewer should appear with the dataset loaded!

## What's next?

