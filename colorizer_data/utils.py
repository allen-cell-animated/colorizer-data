import json
import logging
import os
import pathlib
import platform
import re
from typing import List, Sequence, Union

import numpy as np
import pandas as pd
import skimage

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


def extract_units_from_feature_name(feature_name: str) -> (str, Union[str, None]):
    """
    Extracts units from the parentheses at the end of a feature name string, returning
    the feature name (without units) and units as a tuple. Returns None for the units
    if no units are found.

    ex: `"Feature Name (units)" -> ("Feature Name", "units")`
    """
    match = re.search(r"\((.+)\)$", feature_name)
    if match is None:
        return (feature_name, None)
    units = match.group()
    units = units[1:-1]  # Remove parentheses
    feature_name = feature_name[: match.start()].strip()
    return (feature_name, units)


def remap_segmented_image(
    seg2d: np.ndarray,
    frame: pd.DataFrame,
    object_id_column: str,
    absolute_id_column: str = INITIAL_INDEX_COLUMN,
) -> (np.ndarray, np.ndarray):
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