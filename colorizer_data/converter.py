import logging
import math
import multiprocessing
import os
import pathlib
import shutil
import time
from typing import Dict, List, Optional, Sequence, Union

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
    configureLogging,
    generate_frame_paths,
    get_total_objects,
    merge_dictionaries,
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
    seg_id_column: str
    times_column: str
    track_column: str
    centroid_x_column: str
    centroid_y_column: str
    centroid_z_column: Optional[str]
    outlier_column: str
    backdrop_column_names: Optional[List[str]] = None
    backdrop_info: Optional[Dict[str, BackdropMetadata]] = None
    feature_column_names: Optional[List[str]] = None
    feature_info: Optional[Dict[str, FeatureInfo]] = None
    output_format: DataFileType = DataFileType.PARQUET

    image_column: Optional[str] = None

    allow_copy_frames_3d: bool = False
    frames_3d_path: Optional[Union[str, List[str]]] = None
    frames_3d_url: Optional[Union[str, List[str]]] = None
    frames_3d_seg_channel: Optional[int] = None
    
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
    zstack = segmentation_image.get_image_data("ZYX", S=0, T=0, C=0)
    seg2d = zstack.max(axis=0)

    # Scale the image and format as integers.
    seg2d = scale_image(seg2d, scale)
    seg2d = seg2d.astype(np.uint32)

    # Remap the frame image so the IDs are unique across the whole dataset.
    seg_remapped, lut = remap_segmented_image(
        seg2d,
        frame,
        config.seg_id_column,
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
    total_objects_in_dataset = get_total_objects(grouped_frames)

    if math.isnan(total_objects_in_dataset) or (total_objects_in_dataset < 1):
        raise ValueError(
            "No objects found in dataset (e.g., no rows were provided). At least one object is required."
        )

    logging.info("Making {} frames...".format(nframes))

    with multiprocessing.Manager() as manager:
        bounds_arr = manager.Array("i", [0] * int(total_objects_in_dataset * 4))
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
                f"All objects are marked as outliers in column '{config.outlier_column}'. At least one object must not be an outlier."
            )

    tracks_data = _get_data_or_none(dataset, config.track_column)
    if tracks_data is None:
        logging.warning(
            f"No track data found in the dataset for column name '{config.track_column}'. Object IDs will be used as a fallback instead."
        )
        tracks_data = _get_data_or_none(dataset, config.seg_id_column)

    times_data = _get_data_or_none(dataset, config.times_column)
    if times_data is None:
        raise ValueError(
            f"No times data found in the dataset for column name '{config.times_column}'. The time/frame number must be provided as a column."
        )

    writer.write_data(
        tracks=tracks_data,
        times=times_data,
        centroids_x=_get_data_or_none(dataset, config.centroid_x_column),
        centroids_y=_get_data_or_none(dataset, config.centroid_y_column),
        outliers=outliers_data,
        write_json=config.output_format == DataFileType.JSON,
    )


def _get_raw_backdrop_paths(
    grouped_frames: DataFrameGroupBy, column_name: str
) -> List[str | None]:
    # Get the backdrop paths for each frame.
    backdrop_paths = []
    for _group_name, frame in grouped_frames:
        row = frame.iloc[0]
        if column_name not in row or row[column_name] is None:
            backdrop_paths.append(None)
        else:
            backdrop_paths.append(row[column_name])
    return backdrop_paths


def _write_backdrop_from_column(
    backdrop_column: str,
    grouped_frames: DataFrameGroupBy,
    writer: ColorizerDatasetWriter,
    config: ConverterConfig,
):
    backdrop_metadata = BackdropMetadata(
        frames=_get_raw_backdrop_paths(grouped_frames, backdrop_column),
        name=backdrop_column,
        key=sanitize_key_name(backdrop_column),
    )

    # Override metadata if provided
    if config.backdrop_info is not None and backdrop_column in config.backdrop_info:
        override_metadata = config.backdrop_info.get(backdrop_column)
        if override_metadata is not None:
            backdrop_metadata = merge_dictionaries(backdrop_metadata, override_metadata)
            backdrop_metadata["key"] = sanitize_key_name(backdrop_metadata["key"])

    # Iterate over all the backdrop paths and rewrite them to be relative to
    # the dataset directory, copying the file into the dataset dir if necessary.
    updated_frame_paths = []
    for frame_number, raw_backdrop_image_path in enumerate(backdrop_metadata["frames"]):
        if raw_backdrop_image_path is None:
            updated_frame_paths.append(None)
            continue
        backdrop_image_path = pathlib.Path(
            sanitize_path_by_platform(raw_backdrop_image_path)
        )
        if not os.path.exists(backdrop_image_path):
            raise FileNotFoundError(
                f"Backdrop image '{backdrop_image_path}' does not exist. Please check the path and try again."
            )
        if writer.outpath in backdrop_image_path.parents:
            # Path exists and is already in the dataset directory.
            relative_path = backdrop_image_path.relative_to(writer.outpath)
            updated_frame_paths.append(relative_path.as_posix())
        else:
            # Path exists but is not in the dataset directory. Copy it.
            folder = writer.outpath / backdrop_metadata["key"]
            folder.mkdir(parents=True, exist_ok=True)
            dst_path = folder / f"image_{frame_number}.png"
            shutil.copy(backdrop_image_path, dst_path)
            relative_path = dst_path.relative_to(writer.outpath)
            updated_frame_paths.append(relative_path.as_posix())

    # Write the list of backdrop paths to the dataset
    writer.add_backdrops(
        backdrop_metadata["name"], updated_frame_paths, backdrop_metadata["key"]
    )


def _write_backdrops(
    dataset: pd.DataFrame,
    writer: ColorizerDatasetWriter,
    config: ConverterConfig,
):
    grouped_frames = dataset.groupby(config.times_column)

    # Get all backdrop column names. (Don't use set here
    # to preserve ordering)
    all_backdrop_names = config.backdrop_column_names or []
    if config.backdrop_info is not None:
        for backdrop_name in config.backdrop_info.keys():
            if backdrop_name not in all_backdrop_names:
                all_backdrop_names.append(backdrop_name)

    for backdrop_column in all_backdrop_names:
        _write_backdrop_from_column(backdrop_column, grouped_frames, writer, config)


def _get_reserved_column_names(config: ConverterConfig) -> List[str]:
    reserved_columns = [
        config.seg_id_column,
        config.times_column,
        config.track_column,
        config.image_column,
        config.centroid_x_column,
        config.centroid_y_column,
        config.outlier_column,
    ]
    if config.backdrop_column_names is not None:
        reserved_columns += config.backdrop_column_names
    if config.backdrop_info is not None:
        reserved_columns += list(config.backdrop_info.keys())
    return reserved_columns


def _get_reserved_column_names(config: ConverterConfig) -> List[str]:
    reserved_columns = [
        config.seg_id_column,
        config.times_column,
        config.track_column,
        config.centroid_x_column,
        config.centroid_y_column,
        config.outlier_column,
    ]
    if config.backdrop_column_names is not None:
        reserved_columns += config.backdrop_column_names
    elif config.backdrop_info is not None:
        reserved_columns += list(config.backdrop_info.keys())
    
    if config.image_column is not None:
        reserved_columns += config.image_column

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
            feature_info.key = sanitize_key_name(feature_info.key)
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
            if not os.path.exists(writer.outpath / frame):
                logging.info(f"Frame {frame} is missing. Regenerating all frames.")
                return True
    # Get object count and regenerate frames if it has changed
    
    # TODO: Need to check for changes to segmentation ID values or ordering here
    # for this to really be a robust check. Could also check for changes to
    # times.
    num_objects = data[config.seg_id_column].nunique()
    if writer.manifest["times"] is not None:
        # parse existing times to get object count and compare to new data
        times_path = writer.outpath / writer.manifest["times"]

        try:
            times_data = read_data_array_file(times_path)
            if times_data is None:
                logging.info(
                    "Existing times data file could not be parsed. Regenerating all 2D frames."
                )
                return True
            elif len(times_data) != num_objects:
                logging.info(
                    f"Number of objects has changed (old: {len(times_data)}, new: {num_objects}). Regenerating all 2D frames."
                )
                return True
        except Exception as e:
            logging.info(
                f"The existing times data file could not be read, which may indicate a corrupted dataset: {e}. Regenerating all 2D frames."
            )
            return True

    return False


def _validate_manifest(writer: ColorizerDatasetWriter):
    if len(writer.features) == 0:
        raise ValueError(
            "No features found in dataset. At least one feature is required."
        )

def _handle_3d_frames(writer: ColorizerDatasetWriter, config: ConverterConfig) -> None:
    # Check for 3D frame src (safe to assume Zarr?)
    # If 3D frame src is provided, go to 3D source (using bioio?) and 
    if config.frames_3d_path is not None:
        raise NotImplementedError("Paths are not implemented yet.")
    if config.frames_3d_url is not None:
        # Attempt to read the image to get info (such as length)
        img = BioImage(config.frames_3d_url)
        dims = img.shape()
        print(dims)
        # Assumes TCXYZ ordering of dimensions
        writer.set_3d_frame_src(config.frames_3d_url, dims[0], 0)

def convert_colorizer_data(
    data: DataFrame,
    output_dir: Union[str, pathlib.Path],
    *,
    source_dir: Optional[Union[str, pathlib.Path]] = None,
    metadata: Optional[ColorizerMetadata] = None,
    object_id_column: str = None,
    seg_id_column: str = "ID",
    times_column: str = "Frame",
    track_column: str = "Track",

    image_column: str = None,

    frames_3d_path: Optional[str] = None,
    frames_3d_url: Optional[str] = None,
    frames_3d_seg_channel: Optional[int] = None,
    allow_copy_frames_3d: bool = False,

    centroid_x_column: str = "Centroid X",
    centroid_y_column: str = "Centroid Y",
    centroid_z_column: Optional[str] = None,
    outlier_column: str = "Outlier",
    backdrop_column_names: Optional[List[str]] = None,
    backdrop_info: Optional[Dict[str, BackdropMetadata]] = None,
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

        source_dir (str | pathlib.Path | None): The directory containing the source data. Any
            relative paths in the `data` DataFrame will be resolved relative to this directory.
            Absolute paths will not be affected by this value. If `None`, the current working
            directory (`.`) will be used.
        metadata (ColorizerMetadata | None): Metadata to include in the dataset's manifest, such
            as the dataset name, author, dataset description, frame resolution, and time units.
            See `ColorizerMetadata` for more information. Note that some information will be
            written automatically, such as a timestamp and revision number.
        object_id_column (str): DEPRECATED. The name of the column containing object IDs. Defaults to "ID."
        seg_id_column (str): The name of the column containing segmentation IDs. Defaults to "ID."
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
            Defaults to "Centroid Y."
        centroid_z_column (str): The name of the column containing z-coordinates of object
            centroids, in pixels relative to the frame image.
        outlier_column (str): The name of the column containing outlier flags. 0/False indicates a
            normal object, while 1/True indicates an outlier. Outliers are excluded from min/max
            calculation for features. Defaults to "Outlier."
        backdrop_column_names (List[str] | None): A list of column names containing file paths to
            backdrop images. If the images are not already inside the output directory,
            they will be copied into it. Defaults to `None`.
        backdrop_info (Dict[str, BackdropMetadata] | None): A dictionary mapping backdrop names to
            `BackdropMetadata` overrides. This includes the backdrop's name, key, and optionally
            relative paths from `source_dir` to the frame images. Defaults to `None`.
            - If the name matches a backdrop column name in `data`, any defined fields will
            override the default metadata for that column (e.g. `key` or `name`).
            - If the name does not match a backdrop column name, a new backdrop will be added
            with the provided metadata.
            In either case, if the `frame` field is provided, those paths will be used instead of
            the original column values. Frame paths will be edited to be relative to the
            dataset directory, and copied into the directory if they are not already subfiles.
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
    if (object_id_column is not None):
        logging.warning("Argument `object_id_column` is being deprecated; please use `seg_id_column` to indicate the segmentation ID of objects in the image.")
        seg_id_column = object_id_column

    # TODO: Trim spaces from column names and data
    config = ConverterConfig(
        seg_id_column=seg_id_column,
        times_column=times_column,
        track_column=track_column,
        centroid_x_column=centroid_x_column,
        centroid_y_column=centroid_y_column,
        centroid_z_column=centroid_z_column,
        outlier_column=outlier_column,
        backdrop_column_names=backdrop_column_names,
        backdrop_info=backdrop_info,
        feature_column_names=feature_column_names,
        feature_info=feature_info,

        # Frame source
        image_column=image_column,
        frames_3d_path=frames_3d_path,
        frames_3d_url=frames_3d_url,
        frames_3d_seg_channel=frames_3d_seg_channel,
        allow_copy_frames_3d=allow_copy_frames_3d,
        
        # Additional config
        output_format=output_format,
    )

    parent_directory = pathlib.Path(output_dir).parent.absolute()
    dataset_name = pathlib.Path(output_dir).name
    if source_dir is None:
        source_dir = pathlib.Path.cwd()
    original_cwd = pathlib.Path.cwd()

    configureLogging(output_dir=output_dir, log_name="debug.log")

    configureLogging(output_dir=output_dir, log_name="debug.log")

    writer = ColorizerDatasetWriter(parent_directory, dataset_name)

    try:
        # Change source directory for evaluating relative paths
        os.chdir(source_dir)


        # Reorder rows by time, then by seg ID (local segmentation ID per frame)
        # to match the segmentation IDs in the ZARR data.
        data = data.sort_values([config.times_column, config.seg_id_column])

        if image_column is None:
            print("No image column provided, so 2D frame generation will be skipped.")
        elif force_frame_generation or _should_regenerate_frames(writer, data, config):
            # Group the data by time, then run frame generation in parallel.
            reduced_dataset = data[
                [config.times_column, config.image_column, config.seg_id_column]
            ]
            reduced_dataset = reduced_dataset.reset_index(drop=True)
            reduced_dataset[INITIAL_INDEX_COLUMN] = reduced_dataset.index.values
            grouped_frames = reduced_dataset.groupby(config.times_column)
            # TODO: this should pass out the frame paths
            _make_frames_parallel(grouped_frames, 1.0, writer, config)

            # TODO: get accurate count of frames
            # TODO: throw error/warning if times are non-contiguous
            max_frame = data[config.times_column].max()
            writer.set_frame_paths(generate_frame_paths(max_frame + 1))
        else:
            logging.info(
                "2D Frames already exist and no changes were detected. Skipping frame generation."
            )

        if frames_3d_path is not None or frames_3d_url is not None:
            logging.info("3D frame source provided.")
            _handle_3d_frames(writer, config)

        _write_data(data, writer, config)
        _write_features(data, writer, config)
        _write_backdrops(data, writer, config)

        # TODO: Add validation step to check for either frames or frames3d property
        _validate_manifest(writer)
        writer.write_manifest(metadata=metadata)
        logging.info("Dataset conversion completed successfully.")
    except Exception as e:
        raise e
    finally:
        # Restore working directory
        os.chdir(original_cwd)
