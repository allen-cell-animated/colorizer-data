import json
import logging
import os
import pathlib
import platform
import re
import shutil
from typing import Dict, List, Sequence, Union, Tuple

import numpy as np
import pandas as pd
import requests
import skimage

from colorizer_data.types import FeatureInfo, FeatureType

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


class NumpyValuesEncoder(json.JSONEncoder):
    """Handles float32 and int64 values."""

    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def configureLogging(output_dir: Union[str, pathlib.Path], log_name="debug.log"):
    # Set up logging so logs are written to a file in the output directory
    os.makedirs(output_dir, exist_ok=True)
    debug_file = output_dir + log_name
    open(debug_file, "w").close()  # clear debug file if it exists
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[  # output to both log file and stdout stream
            logging.FileHandler(debug_file),
            logging.StreamHandler(),
        ],
    )


def sanitize_path_by_platform(path: str) -> str:
    """Sanitizes paths for specific platforms."""
    if platform.system() == "Windows":
        if path.startswith("/") and not path.startswith("//"):
            return "/" + path
    return path


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


def update_collection(collection_filepath, dataset_name, dataset_path):
    """
    Adds a dataset to a collection file, creating the collection file if it doesn't already exist.
    If the dataset is already in the collection, the existing dataset path will be updated.
    """
    collection = []

    # Read in the existing collection, if it exists
    if os.path.exists(collection_filepath):
        try:
            with open(collection_filepath, "r") as f:
                collection = json.load(f)
        except:
            collection = []

    # Update the collection file and write it out
    with open(collection_filepath, "w") as f:
        in_collection = False
        # Check if the dataset already exists
        for i in range(len(collection)):
            dataset_item = collection[i]
            if dataset_item["name"] == dataset_name:
                # We found a matching dataset, so update the dataset path and exit
                collection[i]["path"] = dataset_path
                json.dump(collection, f)
                return
        # No matching dataset was found, so add it to the collection
        collection.append({"name": dataset_name, "path": dataset_path})
        json.dump(collection, f)


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

    [Documentation for bounds data format](https://github.com/allen-cell-animated/colorizer-data/blob/main/documentation/DATA_FORMAT.md#7-bounds-optional)
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


def convert_string_array_to_categorical_feature(
    data: np.ndarray, info: FeatureInfo
) -> Tuple[np.ndarray, FeatureInfo]:
    """
    Parses a string feature into the expected categorical feature format. Returns an updated copy of
    `info` where `info.categories` is a list of unique strings in order of initial appearance, and a new data
    integer array which indexes into `info.categories`.

    Example:
    ```
    info = FeatureInfo(type=FeatureType.CATEGORICAL, categories=[])
    data = np.array(["A", "B", "C", "A", "D"], dtype=str)

    new_data, new_info = convert_string_array_to_categorical_feature(data, info)
    print(new_data)  # [0, 1, 2, 0, 3]
    print(new_info.categories)  # ["A", "B", "C", "D"]
    ```
    """
    new_info = info.clone()
    categories, indexed_data = np.unique(data.astype(str), return_inverse=True)
    new_info.categories = categories.tolist()
    new_info.type = FeatureType.CATEGORICAL
    return (indexed_data, new_info)


def find_unused_categories(data: np.ndarray, categories: List[str]) -> List[str]:
    inferred_categories = np.unique(data.astype(str))
    return np.setdiff1d(inferred_categories, categories, assume_unique=True)


def remap_string_array_with_categories(
    data: np.ndarray, categories: List[str]
) -> np.ndarray:
    """
    Turns an array of strings (or object data) into an array of integers indexing into the `categories` array.
    Values in `data` that are not present in `categories` are replaced with `np.nan`.

    TODO: Example
    """
    # Adapted from https://stackoverflow.com/a/8251757 "Numpy: For every element in one array, find the index in another array"
    data = data.astype(str)
    index = np.argsort(categories)
    sorted_categories = np.array(categories)[index]
    sorted_index_data = np.searchsorted(sorted_categories, data)

    data_index = np.take(index, sorted_index_data, mode="clip")
    # Mask out values that are not present in the categories array
    mask = np.array(categories)[data_index] != data
    data_index = data_index.astype(float)
    data_index[mask] = np.nan
    return data_index


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
        # Potentially dangerous for floats/numbers stored as strings?
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
    `info.categories` array. If category information is missing or data is non-numeric, categories
    will be automatically inferred.

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
        if info.categories == None:
            logging.warning(
                "Feature '{}' has type set to CATEGORICAL, but is missing a categories array.".format(
                    info.get_name()
                )
            )
            logging.warning("Categories will be automatically inferred from the data.")
            logging.warning(
                "If the output looks incorrect, provide the categories as a string array and the data as an array of integers."
            )
            return convert_string_array_to_categorical_feature(data, info)
        else:
            # Feature has predefined categories. Warn that values will be remapped.
            logging.warning(
                "CATEGORICAL feature '{}' has a categories array defined, but data type is not an int or float. Feature values will be mapped as integer indexes to categories.".format(
                    info.get_name()
                )
            )
            indexed_data = remap_string_array_with_categories(data, info.categories)
            dropped_categories = find_unused_categories(data, info.categories)
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
