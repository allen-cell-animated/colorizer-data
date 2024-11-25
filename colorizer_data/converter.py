import json
import logging
import math
import multiprocessing
import os
import pathlib
import time
from typing import Dict, List, Optional, Sequence, TypedDict, Union

from bioio import BioImage
from dataclasses import dataclass
from pandas import DataFrame
from pandas.core.groupby.generic import DataFrameGroupBy
import pandas as pd
import numpy as np


from colorizer_data.types import (
    BackdropMetadata,
    ColorizerMetadata,
    DataFileType,
    FeatureInfo,
)
from colorizer_data.utils import (
    INITIAL_INDEX_COLUMN,
    generate_frame_paths,
    get_total_objects,
    read_data_array_file,
    remap_segmented_image,
    sanitize_key_name,
    sanitize_path_by_platform,
    scale_image,
    update_bounding_box_data,
)
from colorizer_data.writer import ColorizerDatasetWriter


@dataclass
class ConverterConfig:
    object_id_column: str
    times_column: str
    track_column: str
    image_column: str
    centroid_x_column: str
    centroid_y_column: str
    outlier_column: str
    backdrop_columns: Optional[List[str]] = None
    backdrop_info: Optional[Dict[str, BackdropMetadata]] = None
    feature_column_names: Optional[List[str]] = None
    feature_info: Optional[Dict[str, FeatureInfo]] = None
    output_format: DataFileType = DataFileType.PARQUET


def _get_image_from_row(row: pd.DataFrame, config: ConverterConfig) -> BioImage:
    zstackpath = row[config.image_column]
    zstackpath = sanitize_path_by_platform(zstackpath)
    return BioImage(zstackpath)


def _make_frame(
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
    segmentation_image = _get_image_from_row(row, config)
    zstack = segmentation_image.get_image_data("YX", S=0, T=0, C=0)
    seg2d = zstack
    # seg2d = zstack.max(axis=0)

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


def _make_frames_parallel(
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

    if math.isnan(total_objects) or (total_objects < 1):
        raise ValueError(
            "No objects found in dataset (e.g., no rows were provided). At least one object is required."
        )

    logging.info("Making {} frames...".format(nframes))

    with multiprocessing.Manager() as manager:
        bounds_arr = manager.Array("i", [0] * int(total_objects * 4))
        with multiprocessing.Pool() as pool:
            pool.starmap(
                _make_frame,
                [
                    (frame, scale, bounds_arr, writer, config)
                    for _group_name, frame in grouped_frames
                ],
            )
        writer.write_data(
            bounds=np.array(bounds_arr, dtype=np.uint32),
            write_json=config.output_format == DataFileType.JSON,
        )


def _get_data_or_none(
    dataset: pd.DataFrame, column_name: str
) -> Union[np.ndarray, None]:
    if column_name in dataset.columns:
        return dataset[column_name].to_numpy()
    else:
        return None


def _write_data(
    dataset: pd.DataFrame,
    writer: ColorizerDatasetWriter,
    config: ConverterConfig,
):
    outliers_data = _get_data_or_none(dataset, config.outlier_column)
    if outliers_data is not None:
        outliers_data = outliers_data.astype(bool)
        if outliers_data.all():
            raise ValueError(
                "All objects are marked as outliers. At least one object must not be an outlier."
            )

    tracks_data = _get_data_or_none(dataset, config.track_column)
    if tracks_data is None:
        tracks_data = _get_data_or_none(dataset, config.object_id_column)

    writer.write_data(
        tracks=tracks_data,
        times=_get_data_or_none(dataset, config.times_column),
        centroids_x=_get_data_or_none(dataset, config.centroid_x_column),
        centroids_y=_get_data_or_none(dataset, config.centroid_y_column),
        outliers=outliers_data,
        write_json=config.output_format == DataFileType.JSON,
    )


def _write_backdrops(
    dataset: pd.DataFrame,
    writer: ColorizerDatasetWriter,
    config: ConverterConfig,
):
    pass


def _get_reserved_column_names(config: ConverterConfig) -> List[str]:
    reserved_columns = [
        config.object_id_column,
        config.times_column,
        config.track_column,
        config.image_column,
        config.centroid_x_column,
        config.centroid_y_column,
        config.outlier_column,
    ]
    # TODO: Revisit this when backdrop handling is implemented
    if config.backdrop_columns is not None:
        reserved_columns += config.backdrop_columns
    elif config.backdrop_info is not None:
        reserved_columns += list(config.backdrop_info.keys())
    return reserved_columns


def _write_features(
    dataset: pd.DataFrame,
    writer: ColorizerDatasetWriter,
    config: ConverterConfig,
):
    # Detect all features
    feature_columns = config.feature_column_names
    if config.feature_column_names is None:
        reserved_columns = _get_reserved_column_names(config)
        feature_columns = [
            col for col in dataset.columns if col not in reserved_columns
        ]
        logging.info(
            f"No feature columns specified. The following columns will be used as features: {feature_columns}"
        )

    outliers = _get_data_or_none(dataset, config.outlier_column)
    if outliers is not None:
        outliers = outliers.astype(bool)

    for feature_column in feature_columns:
        # Get the feature data
        feature_data = dataset[feature_column].to_numpy()
        # Get feature info if it's provided
        feature_info = FeatureInfo(
            label=feature_column,
            key=sanitize_key_name(feature_column),
        )
        if (config.feature_info is not None) and (
            feature_column in config.feature_info
        ):
            feature_info = config.feature_info.get(feature_column)
        writer.write_feature(
            feature_data,
            feature_info,
            outliers=outliers,
            write_json=config.output_format == DataFileType.JSON,
        )


def _should_regenerate_frames(
    writer: ColorizerDatasetWriter, data: DataFrame, config: ConverterConfig
) -> bool:

    if "frames" not in writer.manifest:
        logging.info("No frames found in dataset manifest. Regenerating all frames.")
        return True
    else:
        # Check that all frames exist. If any are missing, frames should be regenerated.
        for frame in writer.manifest["frames"]:
            if not os.path.exists(writer.outpath + "/" + frame):
                logging.info(f"Frame {frame} is missing. Regenerating all frames.")
                return True
    # get object count and regenerate frames if it has changed
    num_objects = data[config.object_id_column].nunique()
    if writer.manifest["times"] is not None:
        # parse existing times to get object count and compare to new data
        times_path = writer.outpath + "/" + writer.manifest["times"]

        try:
            times_data = read_data_array_file(times_path)
            if times_data is None:
                logging.info(
                    "Existing times data file could not be parsed. Regenerating all frames."
                )
                return True
            elif len(times_data) != num_objects:
                logging.info(
                    f"Number of objects has changed (old: {len(times_data)}, new: {num_objects}). Regenerating all frames."
                )
                return True
        except Exception as e:
            logging.info(
                f"The existing times data file could not be read, which may indicate a corrupted dataset: {e}. Regenerating all frames."
            )
            return True

    return False


def _validate_manifest(writer: ColorizerDatasetWriter):
    if len(writer.features) == 0:
        raise ValueError(
            "No features found in dataset. At least one feature is required."
        )


def convert_colorizer_data(
    data: DataFrame,
    output_dir: Union[str, pathlib.Path],
    *,
    metadata: Optional[ColorizerMetadata] = None,
    object_id_column: str = "ID",
    times_column: str = "Frame",
    track_column: str = "Track",
    image_column: str = "File Path",
    centroid_x_column: str = "Centroid X",
    centroid_y_column: str = "Centroid Y",
    outlier_column: str = "Outlier",
    # backdrop_columns: Optional[
    #     List[str]
    # ] = None,  # use this if backdrops are column -> paths to images
    # backdrop_info: Optional[
    #     Dict[str, BackdropMetadata]
    # ] = None,  # use this if backdrops are already stored somewhere
    feature_column_names: Union[List[str], None] = None,
    feature_info: Optional[Dict[str, FeatureInfo]] = None,
    force_frame_generation=False,
    output_format=DataFileType.PARQUET,
):
    """
    Converts a pandas DataFrame into a Timelapse Feature Explorer dataset.

    Args:
        data (pd.DataFrame): The DataFrame to parse. Each row should represent a single object
            at a single frame number. Certain columns are required, such as object IDs ("ID"),
            frame number ("Frame"), and tracks ("Track"). These default column names can be
            overridden using the keyword arguments below.
        output_dir (str | pathlib.Path): The directory to write the dataset to. All dataset files
            will be placed directly within this directory.

        metadata (ColorizerMetadata | None): Metadata to include in the dataset's manifest, such
            as the dataset name, author, dataset description, frame resolution, and time units.
            See `ColorizerMetadata` for more information. Note that some information will be
            written automatically, such as a timestamp and revision number.
        object_id_column (str): The name of the column containing object IDs. Defaults to "ID."
        times_column (str): The name of the column containing time steps. Defaults to "Frame."
        track_column (str): The name of the column containing track IDs. Defaults to "Track."
        image_column (str): The name of the column containing filepaths to the segmentation images.
            Defaults to "File Path." Images will be copied and remapped. If they are 3D, they will
            be flattened along the Z-axis using a max projection.
        centroid_x_column (str): The name of the column containing x-coordinates of object
            centroids, in pixels relative to the frame image, where 0 is the left edge of the
            image. Defaults to "Centroid X."
        centroid_y_column (str): The name of the column containing y-coordinates of object
            centroids, in pixels relative to the frame image, where 0 is the top edge of the image.
            Defaults to "Centroid Y.""
        outlier_column (str): The name of the column containing outlier flags. 0/False indicates a
            normal object, while 1/True indicates an outlier. Outliers are excluded from min/max
            calculation for features. Defaults to "Outlier."
        backdrop_columns (List[str] | None): A list of column names containing file paths to
            backdrop images. If set, these images will be copied and included in the dataset as
            backdrops that can be toggled. Defaults to `None`.
        backdrop_info (Dict[str, BackdropMetadata] | None): A dictionary mapping column names to
            `BackdropMetadata` metadata. This includes the backdrop's name, description, and
            file paths. If the files do not exist in the dataset directory, the files
            will be copied to it and the paths updated to be relative to the manifest. Defaults to
            `None`.
        feature_column_names (List[str] | None): An array of feature column names. If a value is
            provided, ONLY the provided column names will be parsed as features; otherwise, ALL
            columns that aren't specified as a backdrop or a data column (e.g. object ID, time,
            track, etc.) will be treated as a feature. Defaults to `None`.
        feature_info (Dict[str, FeatureInfo] | None): A dictionary mapping column names to
            `FeatureInfo` metadata. This includes the feature's type (continuous, discrete, or
            categorical), units, min/max value overrides, descriptions, and category order for
            categorical features. If a feature's column name does not exist in the `feature_info`
            (or the dictionary is `None`), the feature type and metadata will be inferred based
            on column values. Defaults to `None`.
        force_frame_generation (bool): If True, frames will be regenerated even if they already
            exist. If False (default), frames will be regenerated only when changes are detected.
        output_format (DataFileType): Enum value, either `DataFileType.PARQUET` or `DataFileType.JSON`.
            Determines the format of the output data files. Defaults to `DataFileType.PARQUET`.

    Example:
        ```python
            import pandas as pd
            from colorizer_data import convert_colorizer_data

            # 1. Assuming CSV data has default columns "ID", "Track",
            #    "Frame", and "File Path":
            data = pd.read_csv("some/path/data.csv")
            convert_colorizer_data(data, "dataset_dir/dataset_name")

            # 2. If not, you can specify the column names:
            convert_colorizer_data(
                data,
                "dataset_dir/dataset_name",
                object_id_column="my_id_column",
                times_column="my_frame_column",
                track_column="my_track_column",
                image_column="my_image_filepath_column",
            )

            # 3. You can also specify metadata for some or all features.
            from colorizer_data import FeatureInfo, FeatureType
            feature_info = {
                "my_feature_column": FeatureInfo(
                    label="Height",
                    unit="µm",
                    type=FeatureType.CONTINUOUS,
                    description="Measured from the upper edge of the cell relative "
                    + "to the top of the glass slide.",
                    min=0,
                    max=10, # overrides min/max values calculated from data
                ),
                "my_other_feature_column": FeatureInfo(
                    label="Cell Type",
                    type=FeatureType.CATEGORICAL,
                    categories=["Colony", "Edge", "Migratory"],
                ),
            }
            convert_colorizer_data(
                data,
                "dataset_dir/dataset_name",
                feature_info=feature_info,
            )
        ```
    """
    # TODO: Trim spaces from column names and data
    config = ConverterConfig(
        object_id_column=object_id_column,
        times_column=times_column,
        track_column=track_column,
        image_column=image_column,
        centroid_x_column=centroid_x_column,
        centroid_y_column=centroid_y_column,
        outlier_column=outlier_column,
        # backdrop_columns=backdrop_columns,
        # backdrop_info=backdrop_info,
        feature_column_names=feature_column_names,
        feature_info=feature_info,
        output_format=output_format,
    )

    parent_directory = pathlib.Path(output_dir).parent
    dataset_name = pathlib.Path(output_dir).name

    writer = ColorizerDatasetWriter(parent_directory, dataset_name)

    if force_frame_generation or _should_regenerate_frames(writer, data, config):
        # Group the data by time, then run frame generation in parallel.
        reduced_dataset = data[
            [config.times_column, config.image_column, config.object_id_column]
        ]
        reduced_dataset = reduced_dataset.reset_index(drop=True)
        reduced_dataset[INITIAL_INDEX_COLUMN] = reduced_dataset.index.values
        grouped_frames = reduced_dataset.groupby(config.times_column)
        # TODO: this should pass out the frame paths
        _make_frames_parallel(grouped_frames, 1.0, writer, config)

    _write_data(data, writer, config)
    _write_features(data, writer, config)
    _write_backdrops(data, writer, config)

    # TODO: get accurate count of frames
    # TODO: throw error/warning if times are non-contiguous
    max_frame = data[config.times_column].max()
    writer.set_frame_paths(generate_frame_paths(max_frame + 1))

    _validate_manifest(writer)
    writer.write_manifest(metadata=metadata)
