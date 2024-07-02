from bioio import BioImage
from datetime import datetime, timezone
import pandas as pd

# TODO: also install bioio-ome-tiff

from colorizer_data.types import DATETIME_FORMAT
from colorizer_data.utils import (
    remap_segmented_image,
)
from colorizer_data.writer import (
    ColorizerDatasetWriter,
    ColorizerMetadata,
    FeatureInfo,
    FeatureType,
)

# Load the dataset
data: pd.DataFrame = pd.read_csv("raw_dataset/data.csv")

# Define column names
OBJECT_ID_COLUMN = "object_id"
TRACK_ID_COLUMN = "track_id"
TIMES_COLUMN = "time"
SEGMENTED_IMAGE_COLUMN = "segmentation_path"
CENTROIDS_X_COLUMN = "centroid_x"
CENTROIDS_Y_COLUMN = "centroid_y"
AREA_COLUMN = "area"
HEIGHT_COLUMN = "height"

# Add in a column to act as an index for the dataset.
# This preserves row numbers even when the dataframe is grouped by
# time later.
INDEX_COLUMN = "index"
data = data.reset_index(drop=True)
data[INDEX_COLUMN] = data.index.values

# Create the writer, which we'll be using to save dataset-related files.
output_dir = "."
dataset_name = "processed_dataset"
writer = ColorizerDatasetWriter(output_dir, dataset_name)

# Turn each column into a numpy array, to be saved by the writer.
tracks = data[TRACK_ID_COLUMN].to_numpy()
times = data[TIMES_COLUMN].to_numpy()
centroids_x = data[CENTROIDS_X_COLUMN].to_numpy()
centroids_y = data[CENTROIDS_Y_COLUMN].to_numpy()
areas = data[AREA_COLUMN].to_numpy()
heights = data[HEIGHT_COLUMN].to_numpy()

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
    unit="px²",
)
height_info = FeatureInfo(
    label="Height",
    key="height",
    type=FeatureType.CONTINUOUS,
    unit="nm",
)
writer.write_feature(areas, area_info)
writer.write_feature(heights, height_info)

# Group data by the timestamp
data_grouped_by_time = data.groupby(TIMES_COLUMN)
frame_paths = []

for frame_num, frame_data in data_grouped_by_time:
    # Get the path to the image and load it.
    frame_path = frame_data.iloc[0][SEGMENTED_IMAGE_COLUMN]

    segmentation_image = BioImage("raw_dataset/" + frame_path).get_image_data(
        "YX", S=0, T=0, C=0
    )

    # NOTE: For datasets with 3D segmentations, you may need to flatten the data into 2D images. If so, replace the above line with the following:
    # segmentation_image = bioio.BioImage(frame_path).get_image_data("ZYX", S=0, T=0, C=0)
    # segmentation_image = segmentation_image.max(axis=0)

    # Remap the segmented so object IDs are unique across all timepoints.
    (remapped_segmentations, _lut) = remap_segmented_image(
        segmentation_image, frame_data, OBJECT_ID_COLUMN, INDEX_COLUMN
    )

    # Write the new segmentation image.
    frame_prefix = "frame_"
    frame_suffix = ".png"
    writer.write_image(remapped_segmentations, frame_num, frame_prefix, frame_suffix)
    frame_paths.append(frame_prefix + str(frame_num) + frame_suffix)

writer.set_frame_paths(frame_paths)

# Define the metadata for this dataset.
metadata = ColorizerMetadata(
    name="Example dataset",
    description="An example dataset for the Timelapse Feature Explorer.",
    author="Jane Doe et al.",
    dataset_version="v1.0",
    date_created=datetime.now(timezone.utc).strftime(DATETIME_FORMAT),
    last_modified=datetime.now(timezone.utc).strftime(DATETIME_FORMAT),
    # The width and height of the original segmentations, in any arbitrary units.
    # This will control the scale bar in the viewer.
    frame_width=100,
    frame_height=100,
    frame_units="nm",
    # Time elapsed between each frame capture, in seconds.
    frame_duration_sec=1,
)


# Write the final dataset
writer.write_manifest(metadata=metadata)
