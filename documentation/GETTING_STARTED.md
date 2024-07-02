# Tutorial: How to process data for Timelapse Feature Explorer

The Timelapse Feature Explorer is a web-based application designed for the interactive visualization and analysis of segmented time-series microscopy data! Data needs to be processed into a specific format to be loaded into the viewer. 

In this tutorial, you'll learn how to prepare your data for the Timelapse Feature Explorer.

## Terms

A few key terms:

- **Dataset**: A dataset is a single time-series, and can have any number of tracked objects and features.
- **Raw dataset**: The raw data that you have collected or generated, before processing into the TFE format.
- **Collection**: An arbitrary grouping of datasets.
- **Object ID**: An ID associated with a single segmentation at a single timepoint. In the TFE-accepted format, object IDs must be sequential, starting from 0, and be unique across the whole dataset.
- **Track ID**: An identifier for a unique set of objects, linking their object IDs across timepoints.

## Prerequisites

From a command terminal, clone this repository and install the dependencies for this tutorial. This will install the necessary dependencies for the example scripts and the latest release of `colorizer-data`. (You may want to do this from a virtual Python environment-- see [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/en/latest/) for more information.)

```bash
git clone https://github.com/allen-cell-animated/colorizer-data.git
cd colorizer-data/documentation/getting_started_guide

pip install -r ./requirements.txt
```

## Working with raw data

For this tutorial, we'll be working with sample data included in the [`getting_started_guide/raw_dataset`](./getting_started_guide/raw_dataset/) directory.

This dataset is a simplified example of raw, pre-processed segmentation data. The data was generated using the [`generate_raw_data.py` script](./getting_started_guide/generate_raw_data.py), which generates a **CSV file** with columns for object IDs, track IDs, times, centroids, features (volume/height), and paths to segmentation files. The **segmentation files** are 2D images in OME-TIFF format.

Your files may be in a different format or need to be transformed. Generally, we recommend:

1. Saving your data as a CSV or other format that can be read into a pandas `DataFrame`.
2. Make every segmented object its row in the table.
3. Include a column for the object's track ID, time, centroid, and any features you want to visualize.

### Example from the raw dataset

| object_id | track_id | time | centroid_x | centroid_y | area | height | segmentation_path |
| ----------- | ---------- | ------ | ------------ | ------------ | -------- | -------- | ------------------- |
| 0 | 0 | 0 | 17 | 47 | 113.1 | 47 | frame_0.tiff |
| 1 | 1 | 0 | 33 | 48 | 113.1 | 48 | frame_0.tiff |
| 2 | 2 | 0 | 50 | 52 | 201.1 | 52 | frame_0.tiff |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 5 | 0 | 1 | 17 | 49 | 50.3 | 49 | frame_1.tiff |
| 6 | 1 | 1 | 33 | 48 | 78.5 | 48 | frame_1.tiff |
| 7 | 2 | 1 | 50 | 50 | 254.5 | 50 | frame_1.tiff |
| ... | ... | ... | ... | ... | ... | ... | ... |

> **_NOTE:_** Note that one segmentation path exists for each timepoint, and object IDs start at zero for each frame/timepoint. Timelapse Feature Explorer requires that every object ID be unique across all timepoints, so we will need to remap the object IDs later.

## Processing data

Timelapse Feature Explorer reads data in the format specified by the [`DATA_FORMAT`](./documentation/DATA_FORMAT.md) document. `colorizer-data` provides utilities for working with the data format.

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

data: pd.DataFrame = pd.read_csv("data.csv")
output_dir = "./processed_dataset"
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

![The Load button on the Timelapse Feature Explorer header, next to the Help dropdown.](./getting_started_guide/assets/load-button.png)

Click the **Load** in the header and paste in the following URL:

```
https://raw.githubusercontent.com/allen-cell-animated/colorizer-data/main/documentation/getting_started_guide/processed_dataset
```

> **_NOTE:_** You can either provide the URL of the directory containing a `manifest.json` or the full URL path of a `.json` file that follows the [manifest specification](./documentation/DATA_FORMAT.md#1-metadata). We recommend specifying the full URL path that includes the `manifest.json`.

Click **Load** in the popup menu to load the dataset. The viewer should appear with the dataset loaded!

## What's next?

