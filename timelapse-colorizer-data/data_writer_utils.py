from dataclasses import dataclass
import json
import logging
import os
import pathlib
import platform
import re
import multiprocessing
from typing import List, Tuple, TypedDict, Union
from multiprocessing import shared_memory

import numpy as np
import pandas as pd
import skimage
from PIL import Image

INITIAL_INDEX_COLUMN = "initialIndex"
"""Column added to reduced datasets, holding the original indices of each row."""
RESERVED_INDICES = 1
"""Reserved indices that cannot be used for cell data. 
0 is reserved for the background."""


class FeatureMetadata(TypedDict):
    units: str


class NumpyValuesEncoder(json.JSONEncoder):
    """Handles float32 and int64 values."""

    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


class SharedArray:
    """
    A multiprocessing-safe array backed by shared memory.

    example usage:
    ```
    # -- main process --
    shared_array = SharedArray(100, np.int64)

    # -- subprocess --
    data, shared_mem = shared_array.get_array()
    # do something with the data
    data[0] = 299
    shared_mem.close()

    # -- main process --
    # do something with the data
    data, shared_mem = shared_array.get_array()
    print(data[0])
    # close the array when finished
    shared_array.close()
    ```
    """

    shared_memory_name: str
    dtype: np.dtype
    size: int

    def __init__(self, size: int, dtype: np.dtype):
        self.dtype = dtype
        self.size = size

        # Allocate space in memory for the array.
        shared_memory = shared_memory.SharedMemory(
            create=True, size=size * dtype.itemsize
        )
        self.shared_memory_name = shared_memory.name
        shared_memory.close()

    def get_array(self) -> Tuple[np.ndarray, shared_memory.SharedMemory]:
        """
        Returns a numpy array backed by shared memory, and the SharedMemory
        object used to create it.

        SharedMemory.close() must be called once the numpy array is no longer
        in use.
        """

        shared_memory = shared_memory.SharedMemory(name=self.shared_memory_name)
        return (
            np.ndarray(
                shape=(self.size),
                dtype=self.dtype,
                buffer=shared_memory.buf,
            ),
            lambda: shared_memory.close(),
        )

    def close(self):
        """
        Permanently cleans up the memory allocated by the SharedArray.
        This should only be done once, when all processes are finished using
        the SharedArray.
        """
        # Free the shared memory.
        try:
            shared_memory = shared_memory.SharedMemory(name=self.shared_memory_name)
            shared_memory.close()
            shared_memory.unlink()
        except:
            pass


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


def get_total_objects(grouped_frames: pd.DataFrame) -> int:
    """Get the total number of object IDs in the dataset."""
    # .max() gives the highest object ID, but not the total number of indices
    # (we have to add 1.)
    return grouped_frames[INITIAL_INDEX_COLUMN].max().max() + 1


def make_bounding_box_shared_array(grouped_frames: pd.DataFrame) -> SharedArray:
    """
    Makes an appropriately-sized SharedArray for bounding box data that can
    be used safely with multiprocessing.
    """
    total_objects = get_total_objects(grouped_frames)
    return SharedArray(size=total_objects * 4, dtype=np.uint32)


def update_bounding_box_data(
    bbox_data: np.ndarray,
    seg_remapped: np.ndarray,
    lut: np.ndarray,
):
    """
    Updates the tracked bounding box data for all the indices in the provided segmented image.

    [Documentation for bounds data format](https://github.com/allen-cell-animated/colorizer-data/blob/main/documentation/DATA_FORMAT.md#7-bounds-optional)
    """
    # Capture bounding boxes
    # Optimize by skipping i = 0, since it's used as a null value in every frame
    for i in range(1, lut.size):
        # Boolean array that represents all pixels segmented with this index
        cell = np.argwhere(seg_remapped == lut[i])

        if cell.size > 0 and lut[i] > 0:
            write_index = (lut[i] - 1) * 4

            # Reverse min and max so it is written in x, y order
            bbox_min = cell.min(0).tolist()
            bbox_max = cell.max(0).tolist()
            bbox_min.reverse()
            bbox_max.reverse()
            bbox_data[write_index : write_index + 2] = bbox_min
            bbox_data[write_index + 2 : write_index + 4] = bbox_max


@dataclass
class ColorizerMetadata:
    """Class representing metadata for a Colorizer dataset."""

    width_units: float
    height_units: float
    units: str

    def to_json(self):
        return {
            "frameDims": {
                "width": self.width_units,
                "height": self.height_units,
                "units": self.units,
            }
        }


class ColorizerDatasetWriter:
    """
    Writes provided data as Colorizer-compatible dataset files to the configured output directory.

    The output directory will contain a `manifest.json` and additional dataset files,
    following the data schema described in the project documentation. (See
    [DATA_FORMAT.md](https://github.com/allen-cell-animated/nucmorph-colorizer/blob/main/documentation/DATA_FORMAT.md)
    for more details.)
    """

    # TODO: Update all of the documentation links to point to the new repo

    outpath: Union[str, pathlib.Path]

    scale: float

    def __init__(
        self,
        output_dir: Union[str, pathlib.Path],
        dataset: str,
        scale: float = 1,
    ):
        self.outpath = os.path.join(output_dir, dataset)
        os.makedirs(self.outpath, exist_ok=True)
        self.scale = scale

    def write_feature_data(
        self,
        features: List[np.array] = None,
        tracks: np.array = None,
        times: np.array = None,
        centroids_x: np.array = None,
        centroids_y: np.array = None,
        outliers: np.array = None,
        bounds: np.array = None,
    ):
        """
        Writes feature, track, centroid, time, and outlier data to JSON files.
        Accepts numpy arrays for each file type and writes them to the configured
        output directory according to the data format.

        Features will be written to files in order of the `features` list,
        starting from 0 (e.g., `feature_0.json`, `feature_1.json`, ...)

        [documentation](https://github.com/allen-cell-animated/nucmorph-colorizer/blob/main/documentation/DATA_FORMAT.md#1-tracks)
        """
        # TODO check outlier and replace values with NaN or something!
        if outliers is not None:
            logging.info("Writing outliers.json...")
            ojs = {"data": outliers.tolist(), "min": False, "max": True}
            with open(self.outpath + "/outliers.json", "w") as f:
                json.dump(ojs, f)

        # Note these must be in same order as features and same row order as the dataframe.
        if tracks is not None:
            logging.info("Writing track.json...")
            trjs = {"data": tracks.tolist()}
            with open(self.outpath + "/tracks.json", "w") as f:
                json.dump(trjs, f)

        if times is not None:
            logging.info("Writing times.json...")
            tijs = {"data": times.tolist()}
            with open(self.outpath + "/times.json", "w") as f:
                json.dump(tijs, f)

        if centroids_x is not None or centroids_y is not None:
            if centroids_x is None or centroids_y is None:
                raise Exception(
                    "Both arguments centroids_x and centroids_y must be defined."
                )
            logging.info("Writing centroids.json...")
            centroids_stacked = np.ravel(np.dstack([centroids_x, centroids_y]))
            centroids_stacked = centroids_stacked * self.scale
            centroids_stacked = centroids_stacked.astype(int)
            centroids_json = {"data": centroids_stacked.tolist()}
            with open(self.outpath + "/centroids.json", "w") as f:
                json.dump(centroids_json, f)

        if bounds is not None:
            logging.info("Writing bounds.json...")
            bounds_json = {"data": bounds.tolist()}
            with open(self.outpath + "/bounds.json", "w") as f:
                json.dump(bounds_json, f)

        if features is not None:
            logging.info("Writing feature json...")
            for i in range(len(features)):
                f = features[i]
                fmin = np.nanmin(f)
                fmax = np.nanmax(f)
                # TODO normalize output range excluding outliers?
                js = {"data": f.tolist(), "min": fmin, "max": fmax}
                with open(self.outpath + "/feature_" + str(i) + ".json", "w") as f:
                    json.dump(js, f, cls=NumpyValuesEncoder)
            logging.info("Done writing features.")

    def write_manifest(
        self,
        num_frames: int,
        feature_names: List[str],
        # TODO: feature metadata should probably go in args of write_feature_data
        # and be tracked by the writer.
        feature_metadata: List[FeatureMetadata] = [],
        metadata: ColorizerMetadata = None,
    ):
        """
        Writes the final manifest file for the dataset in the configured output directory.

        [documentation](https://github.com/allen-cell-animated/nucmorph-colorizer/blob/main/documentation/DATA_FORMAT.md#Dataset)

        `manifest.json:`
        ```
          frames: [frame_0.png, frame_1.png, ...]

          # Names are given by `feature_names` parameter, in order.
          features: { "feature_0_name": "feature_0.json", "feature_1_name": "feature_1.json", ... }

          # per cell, same order as featureN.json files. true/false boolean values.
          outliers: "outliers.json"
          # per-cell track id, same format as featureN.json files
          tracks: "tracks.json"
          # per-cell frame index, same format as featureN.json files
          times: "times.json"
          # per-cell centroid. For each index i, the coordinates are (x: data[2i], y: data[2i + 1]).
          centroids: "centroids.json"
          # bounding boxes for each cell. For each index i, the minimum bounding box coordinates
          # (upper left corner) are given by (x: data[4i], y: data[4i + 1]),
          # and the maximum bounding box coordinates (lower right corner) are given by
          # (x: data[4i + 2], y: data[4i + 3]).
          bounds: "bounds.json"
        ```
        """
        # write manifest file
        featmap = {}
        output_json = {}

        for i in range(len(feature_names)):
            featmap[feature_names[i]] = "feature_" + str(i) + ".json"

        # TODO: Write these progressively to an internal map during feature writing
        # so we only write the files that are known?
        output_json = {
            "frames": ["frame_" + str(i) + ".png" for i in range(num_frames)],
            "features": featmap,
            "outliers": "outliers.json",
            "tracks": "tracks.json",
            "times": "times.json",
            "centroids": "centroids.json",
            "bounds": "bounds.json",
        }

        # Merge the feature metadata together and include it in the output if present
        if feature_metadata:
            if len(feature_metadata) == len(feature_names):
                combined_feature_metadata = {}
                for i in range(len(feature_metadata)):
                    combined_feature_metadata[feature_names[i]] = feature_metadata[i]
                output_json["featureMetadata"] = combined_feature_metadata
            else:
                logging.warn(
                    "Feature metadata length does not match number of features. Skipping metadata."
                )

        # Add the metadata
        if metadata:
            output_json["metadata"] = metadata.to_json()

        with open(self.outpath + "/manifest.json", "w") as f:
            json.dump(output_json, f)

        logging.info("Finished writing dataset.")

    def write_image(
        self,
        seg_remapped: np.ndarray,
        frame_num: int,
    ):
        """
        Writes the current segmented image to a PNG file in the output directory.
        The image will be saved as `frame_{frame_num}.png`.

        IDs for each pixel are stored in the RGBA channels of the image.

        [documentation](https://github.com/allen-cell-animated/nucmorph-colorizer/blob/main/documentation/DATA_FORMAT.md#3-frames)
        """
        seg_rgba = np.zeros(
            (seg_remapped.shape[0], seg_remapped.shape[1], 4), dtype=np.uint8
        )
        seg_rgba[:, :, 0] = (seg_remapped & 0x000000FF) >> 0
        seg_rgba[:, :, 1] = (seg_remapped & 0x0000FF00) >> 8
        seg_rgba[:, :, 2] = (seg_remapped & 0x00FF0000) >> 16
        seg_rgba[:, :, 3] = 255  # (seg2d & 0xFF000000) >> 24
        img = Image.fromarray(seg_rgba)  # new("RGBA", (xres, yres), seg2d)
        img.save(self.outpath + "/frame_" + str(frame_num) + ".png")
