import collections
from datetime import datetime, timezone
import json
import logging
import os
import pathlib
import platform
import re
import shutil
from typing import Dict, List, Optional, Sequence, TypeVar, Union, Tuple

from bioio import BioImage
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import skimage

from colorizer_data.types import (
    CURRENT_VERSION,
    DATETIME_FORMAT,
    CollectionManifest,
    CollectionMetadata,
    ColorizerMetadata,
    FeatureInfo,
    FeatureType,
)

MAX_CATEGORIES = 12
INITIAL_INDEX_COLUMN = "initialIndex"
"""
Column added to reduced datasets, holding the original indices of each row.

example:
```
reduced_dataset = full_dataset[columns]
reduced_dataset = reduced_dataset.reset_index(drop=True)
reduced_dataset[INITIAL_INDEX_COLUMN] = reduced_dataset.index.values
```
"""
DEFAULT_FRAME_PREFIX = "frame_"
DEFAULT_FRAME_SUFFIX = ".png"
RESERVED_INDICES = 1
"""Reserved indices that cannot be used for cell data. 
0 is reserved for the background."""

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


class NumpyValuesEncoder(json.JSONEncoder):
    """Handles numpy numeric values (float32, double, float64, int16, int32, int64)."""

    def default(self, obj):
        if (
            isinstance(obj, float)
            or isinstance(obj, np.float32)
            or isinstance(obj, np.double)
            or isinstance(obj, np.float64)
        ):
            if np.isposinf(obj):
                return "Infinity"
            elif np.isneginf(obj):
                return "-Infinity"
            else:
                return float(obj)
        elif (
            isinstance(obj, int)
            or isinstance(obj, np.int16)
            or isinstance(obj, np.int32)
            or isinstance(obj, np.int64)
        ):
            return int(obj)
        elif obj is None:
            return None
        elif isinstance(obj, str):
            return obj
        return json.JSONEncoder.default(self, obj)


# Adapted from https://stackoverflow.com/a/56944256
class TextColorFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = LOG_FORMAT

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def configure_logging(output_dir: Union[str, pathlib.Path], log_name="debug.log"):
    # Set up logging so logs are written to a file in the output directory
    # Clear debug file if it exists
    output_dir_path = pathlib.Path(output_dir)
    os.makedirs(output_dir_path, exist_ok=True)
    debug_file = output_dir_path / log_name
    open(debug_file, "w").close()

    cliHandler = logging.StreamHandler()
    cliHandler.setFormatter(TextColorFormatter())
    cliHandler.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(debug_file)
    fileHandler.setFormatter(logging.Formatter(LOG_FORMAT))
    fileHandler.setLevel(logging.DEBUG)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[  # output to both log file and stdout stream
            fileHandler,
            cliHandler,
        ],
    )


def sanitize_path_by_platform(
    path: Union[str, pathlib.Path],
) -> str:
    """
    Sanitizes paths for specific platforms.

    All paths or strings are returned as POSIX path strings with forward
    slashes. Also, Windows UNC paths (paths starting with `\\` or `\\\\`) will
    be properly formatted with double slashes at the beginning.
    """
    if platform.system() == "Windows":
        # Sanitize UNC (universal naming convention) paths. See
        # https://learn.microsoft.com/en-us/dotnet/standard/io/file-path-formats#unc-paths.
        # "\\some\path" -> "//some/path"
        # "\some\path" -> "//some/path"
        # Cast to Windows path so backslashes are converted to forward slashes
        posix_path_str = pathlib.PureWindowsPath(path).as_posix()
        if posix_path_str.startswith("/") and not posix_path_str.startswith("//"):
            return "/" + posix_path_str
        return posix_path_str
    return pathlib.Path(path).as_posix()


def scale_image(seg2d: np.ndarray, scale: float) -> np.ndarray:
    """
    Scale an image by some scale factor.
    """
    if scale != 1.0:
        seg2d = skimage.transform.rescale(seg2d, scale, anti_aliasing=False, order=0)
    return seg2d


def extract_units_from_feature_name(feature_name: str) -> Tuple[str, Union[str, None]]:
    """
    Extracts units from the parentheses at the end of a feature name string, returning
    the feature name (without units) and units as a tuple. Returns None for the units
    if no units are found.

    ex: `"Feature Name (units)" -> ("Feature Name", "units")`
    """
    matches = [x for x in re.finditer(r"(\([^\(]*?\))", feature_name)]
    if len(matches) == 0:
        return (feature_name, None)
    match = matches[-1]  # Find last instance
    units = match.group()
    # Splice out the units
    feature_name = (
        feature_name[: match.start()].strip() + feature_name[match.end() :].strip()
    )
    return (feature_name, units)


def remap_segmented_image(
    seg2d: np.ndarray,
    frame: pd.DataFrame,
    object_id_column: str,
    absolute_id_column: str = INITIAL_INDEX_COLUMN,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remap the values in the segmented image 2d array so that each object has a
    unique ID across the whole dataset, accounting for reserved indices.

    Returns the remapped image and the lookup table (LUT) used to remap the IDs on
    this frame.
    """
    # Map values in segmented image to new unique indices for whole dataset
    max_object_id = max(np.nanmax(seg2d), int(np.nanmax(frame[object_id_column])))
    lut = np.zeros((max_object_id + 1), dtype=np.uint32)
    for row_index, row in frame.iterrows():
        # build our remapping LUT:
        object_id = 0
        if isinstance(row[object_id_column], pd.Series):
            # Catch malformed data
            object_id = int(row[object_id_column][0])
        else:
            object_id = int(row[object_id_column])
        # unique row ID for each object -> remap to unique index for whole dataset
        rowind = int(row[absolute_id_column])
        lut[object_id] = rowind + RESERVED_INDICES

    # remap indices of this frame.
    seg_remapped = lut[seg2d]
    return (seg_remapped, lut)


def update_metadata(
    metadata: Optional[Union[CollectionMetadata, ColorizerMetadata]],
    *,
    default_name: Optional[str] = None,
):
    """
    Updates the following fields in a dataset or collection metadata object:
    - date_created: Sets to current time if none exists.
    - last_modified: Sets to current time.
    - _revision: Sets to 0 if none exists; otherwise increments by 1.
    - _writer_version: Sets to `types.CURRENT_VERSION`.
    - name: Sets name if none exists and a `default_name` argument was provided.

    Args:
        metadata (CollectionMetadata | ColorizerMetadata): The metadata object to update.
        default_name (str): The name of the collection or dataset to use if the metadata has none.
    """
    current_time = datetime.now(timezone.utc).strftime(DATETIME_FORMAT)

    # Update creation date if missing
    if metadata.date_created is None:
        metadata.date_created = current_time
    # Update revision number
    revision = metadata._revision
    if revision is None:
        metadata._revision = 0
    else:
        metadata._revision = revision + 1
    # Update data version + modified timestamp
    metadata.last_modified = current_time
    metadata._writer_version = CURRENT_VERSION

    # Use default dataset name from writer constructor if no name was loaded
    # (will be overridden by `metadata.name` argument if provided)
    if metadata.name is None and default_name is not None:
        metadata.name = default_name


# TODO: Should collections have their own writer?
def update_collection(
    collection_path: str,
    dataset_name: str,
    dataset_path: str,
    *,
    metadata: Optional[CollectionMetadata] = None,
):
    """
    Adds a dataset to a collection file, creating the collection file if it doesn't already exist.
    If the dataset is already in the collection, the existing dataset path will be updated.

    Args:
        collection_path: The path of the collection file to create or update. If a directory is provided,
            a default `collection.json` file will be created or updated in that directory.
        dataset_name: The name of the dataset to add to the collection.
        dataset_path: The relative path to the dataset, from the root directory of the `collection_filepath`.
        metadata: Optional metadata to update the collection with. If not provided, the existing metadata will
            be used, and fields will be automatically updated. Define fields in the `metadata` argument to override
            this behavior.
    """
    collection: Optional[CollectionManifest] = None

    # Read in the existing collection, if it exists
    collection_filepath = pathlib.Path(sanitize_path_by_platform(collection_path))

    if os.path.exists(collection_filepath):
        if os.path.isdir(collection_filepath):
            # Write default collection.json
            collection_filepath = collection_filepath / "collection.json"
        else:
            try:
                with open(collection_filepath, "r") as f:
                    collection = json.load(f)
            except Exception as e:
                logging.warning(
                    "update_collection: Failed to read collection file '{}': {}".format(
                        collection_filepath, str(e)
                    )
                )
                collection = None
        logging.info("Updating collection file: {}".format(collection_filepath))
    else:
        if collection_filepath.suffix == "":
            # Directory, append default collection.json filename
            collection_filepath = collection_filepath / "collection.json"
        elif collection_filepath.suffix != ".json":
            logging.warning(
                "update_collection: Collection file '{}' should have a .json extension.".format(
                    collection_filepath
                )
            )
        os.makedirs(collection_filepath.parent, exist_ok=True)
        logging.info("Creating new collection file: {}".format(collection_filepath))

    # TODO: Check that the dataset path exists?
    dataset_path = sanitize_path_by_platform(dataset_path)

    if collection is None:
        collection: CollectionManifest = {
            "datasets": [],
            "metadata": CollectionMetadata(),
        }
    if isinstance(collection, list):
        # Nest into collection structure
        collection: CollectionManifest = {
            "datasets": collection,
            "metadata": CollectionMetadata(),
        }

    # Update the metadata fields
    old_metadata = CollectionMetadata.from_dict(collection["metadata"])
    update_metadata(old_metadata)

    if metadata is not None:
        collection["metadata"] = merge_dictionaries(
            old_metadata.to_dict(), metadata.to_dict()
        )
    else:
        collection["metadata"] = old_metadata.to_dict()

    # Update the collection
    in_collection = False
    for i in range(len(collection["datasets"])):
        dataset_item = collection["datasets"][i]
        if dataset_item["name"] == dataset_name:
            collection["datasets"][i]["path"] = dataset_path
            in_collection = True
    if not in_collection:
        collection["datasets"].append({"name": dataset_name, "path": dataset_path})

    # Update the collection file and write it out
    with open(collection_filepath, "w") as f:
        json.dump(collection, f)


def write_data_array(
    data: np.ndarray,
    outpath: pathlib.Path,
    filename: str,
    *,
    min: Union[float, int, None] = None,
    max: Union[float, int, None] = None,
    write_json: bool = False,
    parquet_compression: str = "brotli",
    parquet_use_dict: bool = False,
) -> str:
    """
    Writes a numpy array to a JSON or Parquet file, returning the filename of the written file.

    Args:
        data (`np.ndarray[int | float]`): The numpy array to write.
        outpath (`pathlib.Path`): The directory to write the file to.
        filename (`str`): The base filename to write to. The resulting file will be named `{filename}.parquet`
            or `{filename}.json`.

        min (`int | float | None`): The minimum value of the data array. Written only to JSON files. Defaults to None.
        max (`int | float | None`): The maximum value of the data array. Written only to JSON files. Defaults to None.
        write_json (`bool`): If True, writes the data as a JSON file instead of a Parquet file. False by default.
        parquet_compression (`str`): The compression algorithm to use for Parquet files. Defaults to 'brotli'.
            See https://arrow.apache.org/docs/python/parquet.html#compression-encoding-and-file-compatibility for more details.
        parquet_use_dict (`bool`): If True, uses dictionary encoding for parquet files; useful for large dtypes (like strings)
            with repeated values. Defaults to False.

    Returns:
        The `str` filename of the written file, ending in either `{filename}.parquet` or `{filename}.json`.
    """
    if write_json:
        data_json = {"data": data.tolist(), "min": min, "max": max}
        filename = "{}.json".format(filename)
        with open(outpath / filename, "w") as f:
            json.dump(data_json, f)
        return filename
    else:
        df = pd.DataFrame({"data": data})
        data_arrow = pa.Table.from_pandas(df, preserve_index=False)
        filename = "{}.parquet".format(filename)

        # In testing, brotli compression + no dictionary encoding gave the smallest overall size
        # (followed by GZIP + no dict). Because our data is largely numeric and not string-based,
        # dictionary encoding can double the file size for some float features with highly unique values.
        pq.write_table(
            data_arrow,
            outpath / filename,
            compression=parquet_compression,
            use_dictionary=parquet_use_dict,
        )
        return filename


def read_data_array_file(path: Union[str, pathlib.Path]) -> Optional[np.array]:
    """Reads a data array from a JSON or Parquet file, returning the data array or None if the file does not exist."""
    path = pathlib.Path(sanitize_path_by_platform(str(path)))
    if not path.exists():
        return None
    if path.suffix == ".json":
        with open(path, "r") as f:
            data_json = json.load(f)
            return np.array(data_json["data"])
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
        return np.array(df["data"].values)


def get_total_objects(dataframe: pd.DataFrame) -> int:
    """
    Get the total number of object IDs in the dataset.

    `dataframe` must have have a column matching the constant `INITIAL_INDEX_COLUMN`.
    See `INITIAL_INDEX_COLUMN` for usage.
    """
    # .max() gives the highest object ID, but not the total number of indices
    # (we have to add 1.)
    return dataframe[INITIAL_INDEX_COLUMN].max().max() + 1


def make_bounding_box_array(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Makes an appropriately-sized numpy array for bounding box data.

    `dataframe` must have have a column matching the constant `INITIAL_INDEX_COLUMN`.
    See `INITIAL_INDEX_COLUMN` for usage.
    """
    total_objects = get_total_objects(dataframe)
    return np.ndarray(shape=(total_objects * 4), dtype=np.uint32)


def update_bounding_box_data(
    bbox_data: Union[np.array, Sequence[int]],
    seg_remapped: np.ndarray,
):
    """
    Updates the tracked bounding box data array for all the indices in the provided
    segmented image.

    Args:
        bbox_data (np.array | Sequence[int]): The bounds data array to be updated.
        seg_remapped (np.ndarray): Segmentation image whose indices start at 1 and are are absolutely unique across the whole dataset,
            such as the results of `remap_segmented_image()`.

    [Documentation for bounds data format](https://github.com/allen-cell-animated/colorizer-data/blob/main/documentation/DATA_FORMAT.md#2-8-bounds-optional)
    """
    # Capture bounding boxes
    object_ids = np.unique(seg_remapped)
    for i in range(object_ids.size):
        curr_id = object_ids[i]
        # Optimize by skipping id=0, since it's used as the null background value in every frame.
        if curr_id == 0:
            continue
        # Boolean array that represents all pixels segmented with this index
        cell = np.argwhere(seg_remapped == object_ids[i])

        if cell.size > 0 and curr_id > 0:
            # Write bounds with 0-based indexing
            write_index = (curr_id - 1) * 4

            # Both min and max are in YX dimension order but we will write it to the array in XY order
            bbox_min = cell.min(0)
            bbox_max = cell.max(0)
            bbox_data[write_index] = bbox_min[1]
            bbox_data[write_index + 1] = bbox_min[0]
            bbox_data[write_index + 2] = bbox_max[1]
            bbox_data[write_index + 3] = bbox_max[0]


def sanitize_key_name(name: str) -> str:
    name = name.strip().replace(" ", "_").lower()
    # Remove all non-alphanumeric characters
    pattern = "[^0-9a-z_]+"
    return re.sub(pattern, "", name)


def generate_frame_paths(
    num_frames: int,
    start_frame: int = 0,
    file_prefix: str = DEFAULT_FRAME_PREFIX,
    file_suffix: str = DEFAULT_FRAME_SUFFIX,
) -> List[str]:
    """
    Returns a list of image file paths of length `num_frames`, with a configurable prefix, suffix,
    and starting frame number.

    By default, returns the list:
    - `frame_0.png`
    - `frame_1.png`
    - `frame_2.png`
    - `...`
    - `frame_{num_frames - 1}.png`
    """
    return [file_prefix + str(i + start_frame) + file_suffix for i in range(num_frames)]


def make_relative_image_paths(frame_paths: List[str], subdir_path: str) -> List[str]:
    """
    Preserves the base filename from a list of file paths or URLs, and maps them
    as an array of relative paths to new files in the subdirectory.

    If file names are duplicated, renames the files to prevent overwriting.
    (`img.png, img(0).png, img(1).png, etc.`)

    @example
    If the following four paths are provided as an array, and `subdir_path="./img/"`,
    they will be mapped to the following relative paths:

    | `index` | `frame_paths`                   | returned array        |
    | :-----: | :------------------------------ | :-------------------- |
    | 0       | /local/test/frame_0.png         | ./img/frame_0.png     |
    | 1       | /local/test/frame_1.png         | ./img/frame_1.png     |
    | 2       | `http://url.com/1/my_image.png` | ./img/my_image.png    |
    | 3       | `http://url.com/2/my_image.png` | ./img/my_image(0).png |
    """
    # TODO: Unit testing
    relative_paths = []
    # Handle case where files can be duplicated
    filename_count: Dict[str, int] = {}
    for path in frame_paths:
        # Get the filename from the paths
        # TODO: potentially unsafe for URLs if they have additional URL parameters?
        filename = path.rsplit("/", 1)[-1]
        basename, extension = os.path.splitext(filename)
        if filename in filename_count:
            count = filename_count[filename]
            new_filename = "{}({}){}".format(basename, count, extension)
            relative_paths.append(os.path.join(subdir_path, new_filename))
            filename_count[filename] = count + 1
        else:
            relative_paths.append(os.path.join(subdir_path, filename))
            filename_count[filename] = 0
    return relative_paths


def copy_remote_or_local_file(src_path: str, dst_path: str) -> None:
    """Copies a source file from a URL or filepath to destination filename."""
    if src_path.startswith("http"):
        # Download the image
        r = requests.get(src_path)
        if not r.ok:
            raise FileNotFoundError(
                f"Backdrop image '{src_path}' could not be downloaded."
            )
        with open(dst_path, "wb") as f:
            f.write(r.content)
    else:
        # Copy the image
        src_path = sanitize_path_by_platform(src_path)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Backdrop image '{src_path}' does not exist.")
        shutil.copyfile(src_path, dst_path)


def get_categories_from_feature_array(data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Gets the list of unique category strings in order of initial appearance from the data,
    which can be used with `info.categories`.

    Args:
        data: A numpy array that can be parsed as a string.

    Returns:
        The list of string categories, excluding `None` and `np.NaN`.

    """

    # Remove np.nan and/or None when getting unique values
    mask = pd.isnull(data)
    categories = np.unique(data[~mask].astype(str))
    return categories.tolist()


def get_unused_categories(data: np.ndarray, categories: List[str]) -> List[str]:
    """Returns any unique values of `data` that are not represented in `categories` as a list of strings."""
    inferred_categories = np.unique(data.astype(str))
    return np.setdiff1d(inferred_categories, categories, assume_unique=True)


def replace_out_of_bounds_values_with_nan(
    data: np.ndarray, min: float, max: float
) -> np.ndarray:
    """
    Replaces values in an array outside the min and max range (inclusive) with `np.nan`.
    """
    mask = (data < min) | (data > max)
    data[mask] = np.nan


def remap_categorical_feature_array(
    data: np.ndarray, categories: List[str]
) -> np.ndarray:
    """
    Turns an array of strings (or object data) into an array of (floating point) integers indexing
    into the provided `categories` array. Values in `data` that are not present in `categories`
    are replaced with `np.nan`.

    If you don't have a predetermined category array, use `get_categories_from_feature_array()` to find
    these values automatically.

    Example:
    ```
    categories = ["a", "b", "c"]
    data = np.array(["d", "b", "a", "a", "b", "c", "d", None])
    remapped_data = remap_categorical_feature_array(data, categories)
    # remapped_data: [np.nan, 1, 0, 0, 1, 2, np.nan, np.nan]
    ```
    """
    # Adapted from https://stackoverflow.com/a/8251757 "Numpy: For every element in one array, find the index in another array"
    data = data.astype(str)

    # index_to_sorted_index is a transform from the original category indices to their sorted order.
    # We use np.searchsorted to map the values in data to their indices in the `sorted_categories` array,
    # then invert the sorting transform to get the indexes relative to the original `categories` array.
    index_to_sorted_index = np.argsort(categories)
    sorted_categories = np.array(categories)[index_to_sorted_index]
    data_index_into_sorted_categories = np.searchsorted(sorted_categories, data)

    # Invert the sorting
    data_index_into_categories = np.take(
        index_to_sorted_index, data_index_into_sorted_categories, mode="clip"
    )

    # Mask out values that are not present in the categories array
    mask = np.array(categories)[data_index_into_categories] != data
    data_index_into_categories = data_index_into_categories.astype(float)
    data_index_into_categories[mask] = np.nan

    return data_index_into_categories


def infer_feature_type(data: np.ndarray, info: FeatureInfo) -> FeatureType:
    """
    Infer a concrete feature type from possibly unknown feature data and info types.

    Args:
        data (np.ndarray): The feature's data array.
        info (FeatureInfo): The feature's metadata.

    Returns:
        - If `info.type` is concrete (not `FeatureType.INDETERMINATE`), returns the type.
        - Otherwise, returns either `CATEGORICAL`, `DISCRETE`, or `CONTINUOUS` based on the type of the data array
        and other `info` metadata.
    """
    if info.type != FeatureType.INDETERMINATE:
        return info.type

    # See https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
    kind = data.dtype.kind
    if info.categories is not None and len(info.categories) > 0:
        # Category array is defined; assume categorical
        return FeatureType.CATEGORICAL
    elif kind in {"i", "u"}:
        # TODO: Check for floats that only have integer values
        return FeatureType.DISCRETE
    elif kind in {"f"}:
        return FeatureType.CONTINUOUS
    else:
        logging.warning(
            "Feature '{}' has non-numeric data, and will be assumed to be type CATEGORICAL.".format(
                info.get_name()
            )
        )
        return FeatureType.CATEGORICAL


def safely_cast_array_to_int(data: np.ndarray) -> np.ndarray:
    """
    Transforms an array of numeric values into integer values, truncating float values if present.

    (If the array contains NaN, the returned array will have dtype float, as NaN is not representable
    with integers.)
    """
    if np.isnan(data).any():
        # NaN values can't be represented as an integer (defaults to MIN_INT).
        # Keep data as truncated float values.
        return np.trunc(data).astype(float)
    return data.astype(int)


def cast_feature_to_info_type(
    data: np.ndarray, info: FeatureInfo
) -> Tuple[np.ndarray, FeatureInfo]:
    """
    Validates the type of a feature, casting the data values if needed.
    If the feature info has no type (`FeatureType.INDETERMINATE`), attempts to infer the feature's type
    and updates the feature info and data array.

    - For `FeatureType.DISCRETE` features, float values will be truncated and cast to int (if possible).
    - For `FeatureType.CONTINUOUS` features, values will be cast to float.
    - For `FeatureType.CATEGORICAL` features, data values converted to integer indexes into the
    `info.categories` array.
        - If `info.categories` is missing, categories will be automatically inferred.
        - If `info.categories` is present but data is non-numeric, maps data to the provided categories array.
        Values not in `info.categories` are replaced with `np.nan`.

    Args:
        data (np.ndarray): The feature's data array.
        info (FeatureInfo): The feature's metadata.

    Raises:
        RuntimeError if the feature type is CONTINUOUS or DISCRETE, but data is non-numeric.

    Returns:
        A tuple, containing an `np.ndarray` and a (possibly updated) copy of `info`.
    """
    info = info.clone()

    if info.type == FeatureType.INDETERMINATE:
        logging.warning(
            "Info type for feature '{}' is INDETERMINATE. Will attempt to infer feature type.".format(
                info.get_name()
            )
        )
        info.type = infer_feature_type(data, info)

    kind = data.dtype.kind
    if info.type == FeatureType.CONTINUOUS:
        if kind not in {"f", "u", "i"}:
            raise RuntimeError(
                "Feature '{}' has type set to CONTINUOUS, but has non-numeric data.".format(
                    info.get_name()
                )
            )
        return (data.astype(float), info)
    if info.type == FeatureType.DISCRETE:
        if kind not in {"f", "u", "i"}:
            raise RuntimeError(
                "Feature '{}' has type set to DISCRETE, but has non-numeric data.".format(
                    info.get_name()
                )
            )
        return (safely_cast_array_to_int(data), info)
    if info.type == FeatureType.CATEGORICAL:
        if info.categories is not None and kind in {"i", "u", "f"}:
            # Formatted correctly, return directly
            return (safely_cast_array_to_int(data), info)
        # Attempt to parse the data
        if info.categories is None:
            logging.warning(
                "Feature '{}' has type set to CATEGORICAL, but is missing a categories array.".format(
                    info.get_name()
                )
            )
            logging.warning(
                "Categories will be automatically inferred from the data. Set `FeatureInfo.categories` to override this behavior."
            )
            info.categories = get_categories_from_feature_array(data)
        else:
            # Feature has predefined categories, warn that we are mapping to preexisting categories.
            logging.warning(
                "CATEGORICAL feature '{}' has a categories array defined, but data type is not an int or float. Feature values will be mapped as integer indexes to categories.".format(
                    info.get_name()
                )
            )
        indexed_data = remap_categorical_feature_array(data, info.categories)
        dropped_categories = get_unused_categories(data, info.categories)
        if len(dropped_categories) > 0:
            logging.warning(
                "\tThe following values were not in the categories array and will be replaced with NaN (up to first 25): {}".format(
                    dropped_categories
                )
            )
        return (safely_cast_array_to_int(indexed_data), info)

    raise RuntimeError(
        "Unrecognized feature type '{}' on feature '{}'".format(
            info.type, info.get_name()
        )
    )


T = TypeVar("T", bound=Dict)


def merge_dictionaries(a: T, b: T) -> T:
    """Recursively merges key-value pairs of `b` into `a`, ignoring keys with `None` values."""
    # This is basically a replacement for `{...a, ...b}` in JavaScript
    if b is None or a is None:
        return a
    # Make shallow copy of a
    a = {**a}

    for key, value in b.items():
        if isinstance(value, dict):
            a[key] = merge_dictionaries(a.get(key), value)
        elif value is not None:
            a[key] = value
    return a


def get_duplicate_items(input: List[str]) -> List[str]:
    """
    Returns a list of any items in the input array that appear more than once.
    Duplicate items are returned in order of their first appearance.
    """
    # Copied from https://stackoverflow.com/questions/9835762/how-do-i-find-the-duplicates-in-a-list-and-create-another-list-with-them
    return [item for item, count in collections.Counter(input).items() if count > 1]


def _get_frame_count_from_3d_source(source: str) -> int:
    # Attempt to read the image to get info (such as length)
    img = BioImage(source)
    return int(img.dims.T)
