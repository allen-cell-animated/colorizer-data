import json
import logging
import multiprocessing
import os
import pathlib
import time
from typing import Dict, List, Sequence, TypedDict, Union

from aicsimageio import AICSImage
from pandas import DataFrame
from pandas.core.groupby.generic import DataFrameGroupBy
import pandas as pd
import numpy as np


from colorizer_data.types import BackdropMetadata, ColorizerMetadata, FeatureInfo
from colorizer_data.utils import (
    get_total_objects,
    remap_segmented_image,
    sanitize_path_by_platform,
    scale_image,
    update_bounding_box_data,
)
from colorizer_data.writer import ColorizerDatasetWriter

"""
from colorizer_data.types import ColorizerMetadata, FeatureInfo, BackdropMetadata
from colorizer_data.utils import update_collection, convert_colorizer_data

data = pd.read_csv('colorizer_data.csv')

dataset_path = convert_colorizer_data(
    data: DataFrame,
    output_dir: str | pathlib.Path
    *,
    metadata: ColorizerMetadata = {},
    object_id_column: str = "Label",  # <- does this make sense? Is there a convention?
    times_column: str = "Frame",
    track_column: str = "Track",
    image_column: str = "File Path",
    centroid_x_column: str,    # <- is there a convention for these too?
    centroid_y_column: str,
    outliers_column: str,
    backdrop_columns: Dict[str, BackdropMetadata] = None,
    feature_column_names: str[] = None,          # Array of feature columns.
                                                 # If set, ONLY these columns will be parsed.
    feature_info: Dict[str, FeatureInfo] = None, # Map from string column name to FeatureInfo. 
                                                 # Metadata will be applied to these features.

    force_frame_generation = False,      # If false, frames will be regenerated only as needed.
)

# If the number of objects is mismatched, this will throw an error
append_feature(
    output_dir: str | pathLib.Path
    feature_data: np.ndarray,
    feature_info: FeatureInfo,
)

update_collection(
    collection_filepath: str | pathlib.Path,
    dataset_name: str,
    dataset_path: str | pathlib.Path,
)
"""

"""
Notes:
- If times is non-contiguous, throw a warning (since backdrops)
"""


class ConverterConfig(TypedDict):
    object_id_column: str
    times_column: str
    track_column: str
    image_column: str
    centroid_x_column: str
    centroid_y_column: str
    outlier_column: str
    backdrop_columns: Dict[str, BackdropMetadata]
    feature_column_names: List[str]
    feature_info: Dict[str, FeatureInfo]


def get_image_from_row(row: pd.DataFrame, config: ConverterConfig) -> AICSImage:
    zstackpath = row[config.image_column]
    zstackpath = sanitize_path_by_platform(zstackpath)
    return AICSImage(zstackpath)


def make_frame(
    frame: pd.DataFrame,
    scale: float,
    bounds_arr: Sequence[int],
    writer: ColorizerDatasetWriter,
    config: ConverterConfig,
):
    start_time = time.time()

    # Get the path to the segmented zstack image frame from the first row (should be the same for
    # all rows in this group, since they are all on the same frame).
    row = frame.iloc[0]
    frame_number = row[config.times_column]
    # Flatten the z-stack to a 2D image.
    aics_image = get_image_from_row(row)
    zstack = aics_image.get_image_data("ZYX", S=0, T=0, C=0)
    seg2d = zstack.max(axis=0)

    # Scale the image and format as integers.
    seg2d = scale_image(seg2d, scale)
    seg2d = seg2d.astype(np.uint32)

    # Remap the frame image so the IDs are unique across the whole dataset.
    seg_remapped, lut = remap_segmented_image(
        seg2d,
        frame,
        config.object_id_column,
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
    config: ConverterConfig,
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
                    (frame, scale, bounds_arr, writer, config)
                    for group_name, frame in grouped_frames
                ],
            )
        writer.write_data(bounds=np.array(bounds_arr, dtype=np.uint32))


def write_data(
    dataset: pd.DataFrame,
    writer: ColorizerDatasetWriter,
    config: ConverterConfig,
):
    outliers = dataset[config["outlier_column"]].to_numpy()
    tracks = dataset[config["track_column"]].to_numpy()
    times = dataset[config["times_column"]].to_numpy()
    # TODO: Check for this column's existence, return None if unavailable
    centroids_x = dataset[config["centroid_x_column"]].to_numpy()
    centroids_y = dataset[config["centroid_y_column"]].to_numpy()

    writer.write_data(
        tracks=tracks,
        times=times,
        centroids_x=centroids_x,
        centroids_y=centroids_y,
        outliers=outliers,
    )


def write_features(
    dataset: pd.DataFrame,
    dataset_name: str,
    writer: ColorizerDatasetWriter,
    config: ConverterConfig,
):
    pass


def should_regenerate_frames(writer: ColorizerDatasetWriter, data: DataFrame) -> bool:
    # get object count and regenerate frames if it has changed
    num_objects = data["Label"].nunique(False)

    if writer.manifest["frames"] is None:
        logging.info("No frames found in dataset manifest. Regenerating all frames.")
        return True
    else:
        # Check that all frames exist. If any are missing, regenerate all.
        for frame in writer.manifest["frames"]:
            if not os.path.exists(writer.outpath / frame):
                logging.info(f"Frame {frame} is missing. Regenerating all frames.")
                return True

    if writer.manifest["times"] is not None:
        # parse existing times to get object count and compare to new data
        # parse either JSON or Parquet!!!
        times_path = writer.outpath / writer.manifest["times"]
        _times_filename, times_extension = os.path.splitext(times_path)
        with open(times_path, "r") as f:
            times_objects = 0
            if times_extension == ".json":
                times_json_content = json.load(f)
                times_objects = len(times_json_content["data"])
            elif times_extension == ".parquet":
                times_df = pd.read_parquet(times_path)
                times_objects = len(times_df["data"])
        if times_objects != num_objects:
            logging.info(
                f"Number of objects has changed. Regenerating all frames. Old: {times_objects}, New: {num_objects}"
            )
            return True

    return False


def convert_colorizer_data(
    data: DataFrame,
    output_dir: Union[str, pathlib.Path],
    *,
    metadata: ColorizerMetadata = {},
    object_id_column: str = "Label",
    times_column: str = "Frame",
    track_column: str = "Track",
    image_column: str = "File Path",
    centroid_x_column: str = "Centroid X",
    centroid_y_column: str = "Centroid Y",
    outlier_column: str = "Outlier",
    backdrop_columns: Dict[str, BackdropMetadata] = {},
    feature_column_names: List[str] = None,  # Array of feature columns.
    # If set, ONLY these columns will be parsed.
    feature_info: Dict[
        str, FeatureInfo
    ] = None,  # Map from string column name to FeatureInfo.
    # Metadata will be applied to these features.
    force_frame_generation=False,  # If false, frames will be regenerated only as needed.
):
    """ """
    config = {
        "object_id_column": object_id_column,
        "times_column": times_column,
        "track_column": track_column,
        "image_column": image_column,
        "centroid_x_column": centroid_x_column,
        "centroid_y_column": centroid_y_column,
        "outlier_column": outlier_column,
        "backdrop_columns": backdrop_columns,
        "feature_column_names": feature_column_names,
        "feature_info": feature_info,
    }

    parent_directory = pathlib.Path(output_dir).parent
    dataset_name = pathlib.Path(output_dir).name

    writer = ColorizerDatasetWriter(parent_directory, dataset_name)

    write_data(data, writer, config)
    write_features(data, writer, config)

    if force_frame_generation or should_regenerate_frames(writer, data):
        # Group the data by time.
        grouped_frames = data.groupby(config["times_column"])
        make_frames_parallel(grouped_frames, 1.0, writer, config)

    # TODO: get count of frames
    writer.write_manifest(1, metadata)

    pass
