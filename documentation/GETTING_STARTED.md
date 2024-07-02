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

This dataset is a simplified example of raw, pre-processed segmentation data. The data was generated using the [`generate_raw_data.py` script](./getting_started_guide/generate_raw_data.py), which generates a **CSV file** with columns for object IDs, track IDs, times, centroids, features (volume/height), and paths to segmentation images. The **segmentation images** are 2D images in the OME-TIFF format.

Your files may be in a different format or have 3D segmentation images, in which case it will need to be transformed. Generally, we recommend:

1. Saving your data as a CSV or other format that can be read into a pandas `DataFrame`,
2. making every segmented object its own row in the table,
3. and including columns for the object's track ID, time, centroid, and any features you want to visualize.

### What does the example dataset look like?

Here's a preview of the raw dataset, `data.csv`:

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

Is this comment outdated? ⬇
> **_NOTE:_** Timelapse Feature Explorer requires that every object ID be unique across all timepoints. In this case, our object IDs are all unique (and roughly correspond to rows), so we don't have to do anything unique, but for other datasets you may need to remap object IDs.

Each of the segmentation images is an OME-TIFF image containing the IDs of the segmented objects.

![Frame 6 of the example dataset. The background is black, labeled `id=0`. There are five circles of various diameters and positions, with IDs starting at 30 and increasing to 34.](./getting_started_guide/assets/sample-segmentation.png)

_Frame 6 of the example dataset, as viewed in FIJI._

> **_NOTE:_** Note that `ID=0` represents the background in the segmentation images. Object ID are expected to start at 0, so values in the segmentation images are incremented by 1 to avoid conflicts with the background ID, so the actual object IDs will be one less than what is shown here.

## Processing data

Timelapse Feature Explorer reads data in the format specified by the [`DATA_FORMAT`](./documentation/DATA_FORMAT.md) document. We'll use the utilities provided by `colorizer-data` to convert to this format.

### Processing script

Start an interactive Python session.

```bash
python
```

Paste the following steps into the interactive terminal. (Alternatively, you can also create a Python script. The full script is available in the [`process_data.py` script](./getting_started_guide/process_data.py).)

#### 1. Import dependencies and load the dataset into a pandas DataFrame

```python
import pandas as pd
from bioio import BioImage
from colorizer_data.writer import (
    ColorizerDatasetWriter,
    ColorizerMetadata,
    FeatureInfo,
    FeatureType,
)
from colorizer_data.utils import (
    # configureLogging,
    generate_frame_paths,
    remap_segmented_image,
    make_bounding_box_array,
)

# Load the dataset
data: pd.DataFrame = pd.read_csv("raw_dataset/data.csv")

# Define column names
OBJECT_ID_COLUMN = "object_id"
TRACK_ID_COLUMN = "track_id"
TIMES_COLUMN = "time"
SEGMENTED_IMAGE_COLUMN = "OutputMask (H2B)"
CENTROIDS_X_COLUMN = "centroid_x"
CENTROIDS_Y_COLUMN = "centroid_y"
AREA_COLUMN = "area"
HEIGHT_COLUMN = "height"

# Make the writer
output_dir = "."
dataset_name = "processed_dataset"
writer = ColorizerDatasetWriter(output_dir, dataset_name)
```

#### 2. Create the writer and metadata objects, then write out the feature data

```python
# Turn each column into a numpy array, to be saved by the writer.
tracks = data[TRACK_ID_COLUMN].to_numpy()
times = dataset[TIMES_COLUMN].to_numpy()
centroids_x = dataset[CENTROIDS_X_COLUMN].to_numpy()
centroids_y = dataset[CENTROIDS_Y_COLUMN].to_numpy()
areas = dataset[AREA_COLUMN].to_numpy()
heights = dataset[HEIGHT_COLUMN].to_numpy()

writer.write_data(
    tracks=tracks,
    times=times,
    centroids_x=centroids_x,
    centroids_y=centroids_y,
)

# Additional metadata can be provided for each feature, which will be shown
# when interacting with it in the viewer.
area_info = FeatureInfo(
    label="Area",
    key="area",
    type=FeatureType.CONTINUOUS,
    units="px²",
)
height_info = FeatureInfo(
    label="Height",
    key="height",
    type=FeatureType.CONTINUOUS,
    units="nm",
)
writer.write_feature(areas, area_info)
writer.write_feature(heights, height_info)
```

#### 3. Write the images

```python
# Group data by the timestamp
data_grouped_by_time = data.groupby(TIMES_COLUMN)
frame_paths = []

for frame_num, frame_data in data_grouped_by_time:
    # Get the path to the image and load it.
    frame_path = frame_data.iloc[0][SEGMENTED_IMAGE_COLUMN]

    segmentation_image = bioio.BioImage(frame_path).get_image_data("YX", S=0, T=0, C=0)

    # NOTE: For datasets with 3D segmentations, you may need to flatten the data into 2D images. If so, replace the above line with the following:
    # segmentation_image = bioio.BioImage(frame_path).get_image_data("ZYX", S=0, T=0, C=0)
    # segmentation_image = segmentation_image.max(axis=0)

    # Remap the segmented so object IDs are unique across all timepoints.
    (remapped_segmentations) = remap_segmented_image(segmentation_image, frame_data, OBJECT_ID_COLUMN, INDEX_COLUMN)

    # Write the remapped segmentation image.
    frame_prefix = "frame_"
    frame_suffix = ".png"
    writer.write_image(seg_remapped, frame_num, frame_prefix, frame_suffix)
    frame_paths.append(frame_prefix + frame_num + frame_suffix)

writer.set_frame_paths(frame_paths)
```

#### 4. Write the dataset and any additional metadata

```python
# Define the metadata for this dataset.
metadata = ColorizerMetadata(
    name="Example dataset",
    description="An example dataset for the Timelapse Feature Explorer.",

    # The width and height of the frames in the original units.
    # Our images are 100x100, but let's say the original microscopy
    # area was 200x200 nanometers.
    frame_width=200,
    frame_height=200,
    frame_units="nm",

    # How long each frame lasts in seconds.
    frame_duration_seconds = 1,
)

# Write the dataset.
writer.write_manifest(metadata=metadata)
```

That's it! The dataset should now be processed and found in the `processed_dataset` directory.

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

