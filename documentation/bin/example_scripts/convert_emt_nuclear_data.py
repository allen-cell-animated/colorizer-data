"""
A utility script for converting nuclear segmentation data from the EMT project. Original dataset
provided by Leigh Harris!

Note that this dataset does not have track IDs, so each unique object ID is treated as its own track.

To export the default datasets, you can run the following commands from the root directory:
```
python documentation/bin/example_scripts/convert_emt_nuclear_data.py --scale 1.0 --output_dir=/allen/aics/animated-cell/Dan/fileserver/colorizer/EMT_nuclear
```
"""

from aicsimageio import AICSImage
import argparse
import json
import logging
import multiprocessing
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
import time
from typing import List, Sequence

from colorizer_data.writer import ColorizerDatasetWriter
from colorizer_data.writer import (
    ColorizerMetadata,
    FeatureInfo,
    FeatureType,
)
from colorizer_data.utils import (
    INITIAL_INDEX_COLUMN,
    configureLogging,
    generate_frame_paths,
    get_total_objects,
    sanitize_path_by_platform,
    scale_image,
    remap_segmented_image,
    update_bounding_box_data,
    update_collection,
)

# DATASET SPEC: See DATA_FORMAT.md for more details on the dataset format!
# You can find the most updated version on GitHub here:
# https://github.com/allen-cell-animated/colorizer-data/blob/main/documentation/DATA_FORMAT.md

# OVERWRITE THESE!! These values should change based on your dataset. These are
# relabeled as constants here for clarity/intent of the column name.
OBJECT_ID_COLUMN = "Label"
"""Column of object IDs (or unique row number)."""
# Track ID column purposefully removed here, as it does not exist in this dataset.
TIMES_COLUMN = "Frame"
"""Column of frame number that the object ID appears in."""
SEGMENTED_IMAGE_COLUMN = "Filepath"
"""Column of path to the segmented image data or z stack for the frame."""
CENTROIDS_X_COLUMN = "x"
"""Column of X centroid coordinates, in pixels of original image data."""
CENTROIDS_Y_COLUMN = "y"
"""Column of Y centroid coordinates, in pixels of original image data."""

FEATURE_INFO: List[FeatureInfo] = [
    FeatureInfo(
        label="Slice",
        column_name="Slice",
        unit="",
        type=FeatureType.CONTINUOUS,
    ),
    FeatureInfo(
        label="Area",
        column_name="Area",
        unit="px²",
        type=FeatureType.CONTINUOUS,
    ),
    FeatureInfo(
        label="Orientation",
        column_name="Orientation",
        unit="",
        type=FeatureType.CONTINUOUS,
    ),
    FeatureInfo(
        label="Aspect Ratio",
        column_name="Aspect_Ratio",
        unit="",
        type=FeatureType.CONTINUOUS,
    ),
    FeatureInfo(
        label="Circularity",
        column_name="Circularity",
        unit="",
        type=FeatureType.CONTINUOUS,
    ),
    FeatureInfo(
        label="Mean Fluorescence",
        column_name="Mean_Fluor",
        unit="AU",
        type=FeatureType.CONTINUOUS,
    ),
]
"""List of features to save to the dataset, with additional information about the label, unit, and feature type."""

PHYSICAL_PIXEL_SIZE_XY = 0.271
PHYSICAL_PIXEL_UNIT_XY = "µm"


def get_image_from_row(row: pd.DataFrame) -> AICSImage:
    zstackpath = row[SEGMENTED_IMAGE_COLUMN]
    zstackpath = zstackpath.strip('"')
    zstackpath = sanitize_path_by_platform(zstackpath)
    return AICSImage(zstackpath)


def make_frame(
    grouped_frames,
    group_name: int,
    frame: pd.DataFrame,
    scale: float,
    bounds_arr: Sequence[int],
    writer: ColorizerDatasetWriter,
):
    start_time = time.time()

    # Get the path to the segmented zstack image frame from the first row (should be the same for
    # all rows in this group, since they are all on the same frame).
    row = frame.iloc[0]
    frame_number = row[TIMES_COLUMN]
    # Flatten the z-stack to a 2D image.
    aics_image = get_image_from_row(row)
    zstack = aics_image.get_image_data("ZYX", S=0, T=0, C=0)
    # Do a min projection instead of a max projection to prioritize objects which have lower IDs (which for this dataset,
    # indicates lower z-indices). This is due to the nature of the data, where lower cell nuclei have greater confidence,
    # and should be visualized when overlapping instead of higher nuclei.
    # Do a min operation but ignore zero values. Without this, doing `min(0, id)` will always return 0 which results
    # in black images. We use `np.ma.masked_equal` to mask out 0 values and have them be ignored,
    # then replace masked values with 0 again (`filled(0)`) to get our final projected image.
    masked = np.ma.masked_equal(zstack, 0, copy=False)
    seg2d = masked.min(axis=0).filled(0)

    # Scale the image and format as integers.
    seg2d = scale_image(seg2d, scale)
    seg2d = seg2d.astype(np.uint32)

    # Remap the frame image so the IDs are unique across the whole dataset.
    seg_remapped, lut = remap_segmented_image(
        seg2d,
        frame,
        OBJECT_ID_COLUMN,
    )

    writer.write_image(seg_remapped, frame_number)
    update_bounding_box_data(bounds_arr, seg_remapped)

    time_elapsed = time.time() - start_time
    logging.info(
        "Frame {} finished in {:5.2f} seconds.".format(int(frame_number), time_elapsed)
    )


def make_frames_parallel(
    grouped_frames: DataFrameGroupBy,
    scale: float,
    writer: ColorizerDatasetWriter,
):
    """
    Generate the images and bounding boxes for each time step in the dataset.
    """
    nframes = len(grouped_frames)
    total_objects = get_total_objects(grouped_frames)
    logging.info("Making {} frames...".format(nframes))

    with multiprocessing.Manager() as manager:
        bounds_arr = manager.Array("i", [0] * int(total_objects * 4))
        with multiprocessing.Pool() as pool:
            pool.starmap(
                make_frame,
                [
                    (grouped_frames, group_name, frame, scale, bounds_arr, writer)
                    for group_name, frame in grouped_frames
                ],
            )
        writer.write_data(bounds=np.array(bounds_arr, dtype=np.uint32))


def make_features(
    dataset: pd.DataFrame,
    writer: ColorizerDatasetWriter,
):
    """
    Generate the outlier, track, time, centroid, and feature data files.
    """
    # Collect array data from the dataframe for writing.

    # For now in this dataset there are no outliers. Just generate a list of falses.
    outliers = np.array([False for i in range(len(dataset.index))])
    times = dataset[TIMES_COLUMN].to_numpy()
    centroids_x = dataset[CENTROIDS_X_COLUMN].to_numpy()
    centroids_y = dataset[CENTROIDS_Y_COLUMN].to_numpy()

    # This dataset does not have tracks, so we just generate a list of indices, one for each
    # object. This will be a very simple numpy table, where tracks[i] = i.
    shape = dataset.shape
    tracks = np.array([*range(shape[0])])

    writer.write_data(
        tracks=tracks,
        times=times,
        centroids_x=centroids_x,
        centroids_y=centroids_y,
        outliers=outliers,
    )

    for info in FEATURE_INFO:
        data = dataset[info.column_name].to_numpy()
        writer.write_feature(data, info)


def get_dataset_dimensions(grouped_frames: DataFrameGroupBy) -> (float, float, str):
    """Get the dimensions of the dataset from the first frame, in units.
    Returns (width, height, units)."""
    row = grouped_frames.get_group(0).iloc[0]
    aics_image = get_image_from_row(row)
    dims = aics_image.dims
    # TODO: This conversion is hardcoded for now but should be updated with a LUT.
    # This value will change based on microscope objective and scope.
    return (
        dims.X * PHYSICAL_PIXEL_SIZE_XY,
        dims.Y * PHYSICAL_PIXEL_SIZE_XY,
        PHYSICAL_PIXEL_UNIT_XY,
    )


def make_dataset(
    data: pd.DataFrame,
    output_dir="./data/",
    dataset="3500005820_3",
    do_frames=True,
    scale=1,
):
    """Make a new dataset from the given data, and write the complete dataset
    files to the given output directory.
    """
    writer = ColorizerDatasetWriter(output_dir, dataset, scale=scale)
    full_dataset = data
    logging.info("Loaded dataset '" + str(dataset) + "'.")

    # Make a reduced dataframe grouped by time (frame number).
    columns = [
        TIMES_COLUMN,
        SEGMENTED_IMAGE_COLUMN,
        OBJECT_ID_COLUMN,
    ]
    reduced_dataset = full_dataset[columns]
    reduced_dataset = reduced_dataset.reset_index(drop=True)
    reduced_dataset[INITIAL_INDEX_COLUMN] = reduced_dataset.index.values
    grouped_frames = reduced_dataset.groupby(TIMES_COLUMN)

    dims = get_dataset_dimensions(grouped_frames)
    metadata = ColorizerMetadata(
        frame_width=dims[0], frame_height=dims[1], frame_units=dims[2]
    )

    # Make the features, frame data, and manifest.
    nframes = len(grouped_frames)
    writer.set_frame_paths(generate_frame_paths(nframes))

    make_features(full_dataset, writer)
    if do_frames:
        make_frames_parallel(grouped_frames, scale, writer)
    writer.write_manifest(metadata=metadata)


# This is stuff scientists are responsible for!!
def make_collection(output_dir="./data/", do_frames=True, scale=1, dataset=""):
    if dataset != "":
        # convert just the described dataset.
        readPath = f"/allen/aics/assay-dev/computational/data/EMT_deliverable_processing/LH_Analysis/Version2_ForPlotting/{dataset}.csv"
        data = pd.read_csv(readPath)
        logging.info("Making dataset '" + dataset + "'.")
        make_dataset(data, output_dir, dataset, do_frames, scale)

        # Update the collections file if it already exists
        collection_filepath = output_dir + "/collection.json"
        update_collection(collection_filepath, dataset, dataset)
    else:
        # For every condition, make a dataset.
        conditions = [
            "LOW_Matrigel_lumenoid",
            "High_Matrigel_lumenoid",
            "2D_Matrigel",
            "2D_PLF",
        ]
        collection = []

        for condition in conditions:
            # Read in each of the conditions as a dataset
            collection.append({"name": condition, "path": condition})
            readPath = f"/allen/aics/assay-dev/computational/data/EMT_deliverable_processing/LH_Analysis/Version2_ForPlotting/{condition}.csv"
            data = pd.read_csv(readPath)
            logging.info("Making dataset '" + condition + "'.")
            make_dataset(data, output_dir + "/" + condition, dataset, do_frames, scale)
        # write the collection.json file
        with open(output_dir + "/collection.json", "w") as f:
            json.dump(collection, f)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir",
    type=str,
    default="./data/",
    help="Parent directory to output to. Data will be written to a subdirectory named after the dataset parameter.",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="",
    help="Compatible named FMS dataset or FMS id to load. Will be loaded from hardcoded csv.",
)
parser.add_argument(
    "--noframes",
    action="store_true",
    help="If included, generates only the feature data, centroids, track data, and manifest, skipping the frame and bounding box generation.",
)
parser.add_argument(
    "--scale",
    type=float,
    default=1.0,
    help="Uniform scale factor that original image dimensions will be scaled by. 1 is original size, 0.5 is half-size in both X and Y.",
)

args = parser.parse_args()


def main():
    configureLogging(args.output_dir)
    logging.info("Starting...")

    make_collection(
        output_dir=args.output_dir,
        dataset=args.dataset,
        do_frames=not args.noframes,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
