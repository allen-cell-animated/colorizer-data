from typing import List, Sequence
from aicsimageio import AICSImage
import argparse
import json
import logging
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
import time
import csv
import chardet
import multiprocessing

from data_writer_utils import (
    INITIAL_INDEX_COLUMN,
    ColorizerDatasetWriter,
    ColorizerMetadata,
    FeatureInfo,
    FeatureType,
    configureLogging,
    get_total_objects,
    sanitize_path_by_platform,
    scale_image,
    remap_segmented_image,
    update_bounding_box_data,
)

# DATASET SPEC: See DATA_FORMAT.md for more details on the dataset format!
# You can find the most updated version on GitHub here:
# https://github.com/allen-cell-animated/colorizer-data/blob/main/documentation/DATA_FORMAT.md

# R0Nuclei_AreaShape_Center_X
# R0Nuclei_AreaShape_Center_Y
# R0Cell_AreaShape_Center_X
# R0Cell_AreaShape_Center_Y

# OVERWRITE THESE!! These values should change based on your dataset. These are
# relabeled as constants here for clarity/intent of the column name.
OBJECT_ID_COLUMN = "R0Nuclei_Number_Object_Number"
"""Column of object IDs (or unique row number)."""
TRACK_ID_COLUMN = "R0Nuclei_Number_Object_Number"
"""Column of track ID for each object."""
TIMES_COLUMN = "Image_Metadata_Timepoint"
"""Column of frame number that the object ID appears in."""
SEGMENTED_IMAGE_COLUMN = "OutputMask ((labelfreeCAAX))"
"""Column of path to the segmented image data or z stack for the frame."""
CENTROIDS_X_COLUMN = "R0Nuclei_AreaShape_Center_X"
"""Column of X centroid coordinates, in pixels of original image data."""
CENTROIDS_Y_COLUMN = "R0Nuclei_AreaShape_Center_Y"
"""Column of Y centroid coordinates, in pixels of original image data."""
FEATURE_COLUMNS = [
    "R0Cell_Neighbors_NumberOfNeighbors_Adjacent",
    "R0Nuclei_AreaShape_Area",
    "R0Cell_AreaShape_Area",
    "Migratory Cell (colony mask)",
    "Edge cell (colony mask)",
    "Colony cell (colony mask)",
    "Radial distance from colony centroid (um)",
    "Avg(Distance to Neighbor Nuclei)",
]
"""Columns of feature data to include in the dataset. Each column will be its own feature file."""
FEATURE_INFO: List[FeatureInfo] = [
    FeatureInfo(
        label="Adjacent neighbors",
        column_name="R0Cell_Neighbors_NumberOfNeighbors_Adjacent",
        unit="",
        type=FeatureType.DISCRETE,
    ),
    FeatureInfo(
        label="Nuclei area",
        column_name="R0Nuclei_AreaShape_Area",
        unit="",
        type=FeatureType.CONTINUOUS,
    ),
    FeatureInfo(
        label="Cell area",
        column_name="R0Cell_AreaShape_Area",
        unit="",
        type=FeatureType.CONTINUOUS,
    ),
    FeatureInfo(
        label="Radial distance from colony center",
        column_name="Radial distance from colony centroid (um)",
        unit="",
        type=FeatureType.CONTINUOUS,
    ),
    FeatureInfo(
        label="Avg. distance to neighbor nuclei",
        column_name="Avg(Distance to Neighbor Nuclei)",
        unit="",
        type=FeatureType.CONTINUOUS,
    ),
    FeatureInfo(
        label="Migratory cell mask",
        column_name="Migratory Cell (colony mask)",
        unit="",
        type=FeatureType.DISCRETE,
    ),
    FeatureInfo(
        label="Edge cell mask",
        column_name="Edge cell (colony mask)",
        unit="",
        type=FeatureType.DISCRETE,
    ),
    FeatureInfo(
        label="Colony cell mask",
        column_name="Colony cell (colony mask)",
        unit="",
        type=FeatureType.DISCRETE,
    ),
]

PHYSICAL_PIXEL_SIZE_XY = 0.271
PHYSICAL_PIXEL_UNIT_XY = "µm"


def detect_encoding(file_path, sample_size=1024):
    """
    Detect the encoding of a given file by sampling a portion of it.

    Args:
        file_path (str): The path of the file whose encoding needs to be detected.
        sample_size (int): The number of bytes to sample for encoding detection.

    Returns:
        str: The detected encoding.
    """
    with open(file_path, "rb") as f:
        sample = f.read(sample_size)
    return chardet.detect(sample)["encoding"]


def read_csv_generator(file_path):
    """
    Yield rows from a CSV file with automatic encoding detection.

    Args:
        file_path (str): The path of the CSV file to read.

    Yields:
        list: A row from the CSV file.
    """
    encoding = detect_encoding(file_path)

    with open(file_path, mode="r", encoding=encoding) as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            yield row


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
    seg2d = aics_image.get_image_data("YX", S=0, T=0, C=0)

    # Scale the image and format as integers.
    seg2d = scale_image(seg2d, scale)
    seg2d = seg2d.astype(np.uint32)

    # Remap the frame image so the IDs are unique across the whole dataset.
    seg_remapped, lut = remap_segmented_image(
        seg2d,
        frame,
        OBJECT_ID_COLUMN,
    )

    # Write the image and bounds data
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


def make_frames(
    grouped_frames: DataFrameGroupBy,
    scale: float,
    writer: ColorizerDatasetWriter,
):
    """
    Generate the images and bounding boxes for each time step in the dataset.
    """
    nframes = len(grouped_frames)
    logging.info("Making {} frames...".format(nframes))

    for group_name, frame in grouped_frames:
        start_time = time.time()

        # Get the path to the segmented zstack image frame from the first row (should be the same for
        # all rows in this group, since they are all on the same frame).
        row = frame.iloc[0]
        frame_number = row[TIMES_COLUMN]
        # Flatten the z-stack to a 2D image.
        aics_image = get_image_from_row(row)
        seg2d = aics_image.get_image_data("YX", S=0, T=0, C=0)

        # Scale the image and format as integers.
        seg2d = scale_image(seg2d, scale)
        seg2d = seg2d.astype(np.uint32)

        # Remap the frame image so the IDs are unique across the whole dataset.
        seg_remapped, lut = remap_segmented_image(
            seg2d,
            frame,
            OBJECT_ID_COLUMN,
        )

        writer.write_image_and_bounds_data(
            seg_remapped, grouped_frames, frame_number, lut
        )

        time_elapsed = time.time() - start_time
        logging.info(
            "Frame {} finished in {:5.2f} seconds.".format(
                int(frame_number), time_elapsed
            )
        )


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

    tracks = dataset[TRACK_ID_COLUMN].to_numpy()
    shape = dataset.shape
    # tracks = np.array([*range(shape[0])])

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

    # Custom: Write a categorical feature based on cell type by aggregating the masks
    # 0 = colony cell, 1 = edge cell, 2 = migratory cell
    migratory_mask = dataset["Migratory Cell (colony mask)"].to_numpy()
    edge_mask = dataset["Edge cell (colony mask)"].to_numpy()
    cell_types = np.zeros(shape=edge_mask.shape)
    cell_types += edge_mask
    cell_types += migratory_mask * 2
    writer.write_feature(
        cell_types,
        FeatureInfo(
            label="Cell type",
            type=FeatureType.CATEGORICAL,
            categories=["Colony Cell", "Edge Cell", "Migratory Cell"],
        ),
    )


def get_dataset_dimensions(grouped_frames: DataFrameGroupBy) -> (float, float, str):
    """Get the dimensions of the dataset from the first frame, in units.
    Returns (width, height, unit)."""
    row = grouped_frames.get_group(0).iloc[0]
    aics_image = get_image_from_row(row)
    dims = aics_image.dims
    # return (
    #     dims.X * aics_image.physical_pixel_sizes.X,
    #     dims.Y * aics_image.physical_pixel_sizes.Y,
    #     "µm"
    # )
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
        TRACK_ID_COLUMN,  # add this back in if data is tracked
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
    make_features(full_dataset, writer)
    if do_frames:
        make_frames_parallel(grouped_frames, scale, writer)

    writer.write_manifest(nframes, metadata=metadata)


# TODO: Make top-level function
# This is stuff scientists are responsible for!!
def make_collection(output_dir="./data/", do_frames=True, scale=1, dataset=""):
    # example dataset name : 3500005820_3
    # use pandas to load data
    # a is the full collection!
    file_path = "//allen/aics/microscopy/ClusterOutput/H2B_LabelFree_Deliverable_MIP_Zrange/H2BLabelFree_Deliverable/EMT_Deliverable_ColorizerVisualizationTable_AddedMeanNeighborDistanceAndCentroids.csv"

    encoding = detect_encoding(file_path)
    a = pd.read_csv(file_path, encoding=encoding)

    if dataset != "":
        # convert just the described dataset.
        plate = dataset.split("_")[0]
        position = dataset.split("_")[1]
        c = a.loc[a["Image_Metadata_Plate"] == int(plate)]
        c = c.loc[c["Image_Metadata_Position"] == int(position)]
        make_dataset(c, output_dir, dataset, do_frames, scale)
    else:
        # for every combination of plate and position, make a dataset
        b = a.groupby(["Image_Metadata_Plate", "Image_Metadata_Position"])
        collection = []
        for name, group in b:
            dataset = str(name[0]) + "_" + str(name[1])
            logging.info("Making dataset '" + dataset + "'.")
            collection.append({"name": dataset, "path": dataset})
            c = a.loc[a["Image_Metadata_Plate"] == name[0]]
            c = c.loc[c["Image_Metadata_Position"] == name[1]]
            make_dataset(c, output_dir, dataset, do_frames, scale)
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
if __name__ == "__main__":
    configureLogging(args.output_dir)
    logging.info("Starting...")

    make_collection(
        output_dir=args.output_dir,
        dataset=args.dataset,
        do_frames=not args.noframes,
        scale=args.scale,
    )
