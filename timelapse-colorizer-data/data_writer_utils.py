from dataclasses import dataclass
import json
import logging
import os
import pathlib
import platform
import re
from typing import Dict, List, NotRequired, Sequence, TypedDict, Union

import numpy as np
import pandas as pd
import skimage
from PIL import Image

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
RESERVED_INDICES = 1
"""Reserved indices that cannot be used for cell data. 
0 is reserved for the background."""


class FeatureMetadata(TypedDict):
    units: str


class FrameDimensions(TypedDict):
    units: str
    width: int
    height: int


class DatasetMetadata(TypedDict):
    frameDims: FrameDimensions
    frameDurationSeconds: float
    startTimeSeconds: float


@dataclass
class ColorizerMetadata:
    """Data class representation of metadata for a Colorizer dataset."""

    frame_width: float = 0
    frame_height: float = 0
    frame_units: str = ""
    frame_duration_sec: float = 0
    start_time_sec: float = 0

    def to_json(self) -> DatasetMetadata:
        return {
            "frameDims": {
                "width": self.frame_width,
                "height": self.frame_height,
                "units": self.frame_units,
            },
            "startTimeSeconds": self.start_time_sec,
            "frameDurationSeconds": self.frame_duration_sec,
        }


class DatasetManifest(TypedDict):
    frames: List[str]
    features: List[Dict[str, str]]
    outliers: str
    tracks: str
    centroids: str
    times: str
    bounds: str
    featureMetadata: Dict[str, FeatureMetadata]
    metadata: DatasetMetadata


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


class ColorizerDatasetWriter:
    """
    Writes provided data as Colorizer-compatible dataset files to the configured output directory.

    The output directory will contain a `manifest.json` and additional dataset files,
    following the data schema described in the project documentation. (See
    [DATA_FORMAT.md](https://github.com/allen-cell-animated/colorizer-data/blob/main/documentation/DATA_FORMAT.md)
    for more details.)
    """

    outpath: Union[str, pathlib.Path]
    manifest: DatasetManifest
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
        self.manifest = {}

    def write_data(
        self,
        features: Union[List[np.ndarray], None] = None,
        tracks: Union[np.ndarray, None] = None,
        times: Union[np.ndarray, None] = None,
        centroids_x: Union[np.ndarray, None] = None,
        centroids_y: Union[np.ndarray, None] = None,
        outliers: Union[np.ndarray, None] = None,
        bounds: Union[np.ndarray, None] = None,
    ):
        """
        Writes dataset data arrays (such as feature, track, time, centroid, outlier,
        and bounds data) to JSON files.
        Accepts numpy arrays for each file type and writes them to the configured
        output directory according to the data format.

        Features will be written to files in order of the `features` list,
        starting from 0 (e.g., `feature_0.json`, `feature_1.json`, ...)

        [documentation](https://github.com/allen-cell-animated/colorizer-data/blob/main/documentation/DATA_FORMAT.md#1-tracks)
        """
        # TODO check outlier and replace values with NaN or something!
        if outliers is not None:
            logging.info("Writing outliers.json...")
            ojs = {"data": outliers.tolist(), "min": False, "max": True}
            with open(self.outpath + "/outliers.json", "w") as f:
                json.dump(ojs, f)
            self.manifest["outliers"] = "outliers.json"

        # Note these must be in same order as features and same row order as the dataframe.
        if tracks is not None:
            logging.info("Writing track.json...")
            trjs = {"data": tracks.tolist()}
            with open(self.outpath + "/tracks.json", "w") as f:
                json.dump(trjs, f)
            self.manifest["tracks"] = "tracks.json"

        if times is not None:
            logging.info("Writing times.json...")
            tijs = {"data": times.tolist()}
            with open(self.outpath + "/times.json", "w") as f:
                json.dump(tijs, f)
            self.manifest["times"] = "times.json"

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
            self.manifest["centroids"] = "centroids.json"

        if bounds is not None:
            logging.info("Writing bounds.json...")
            bounds_json = {"data": bounds.tolist()}
            with open(self.outpath + "/bounds.json", "w") as f:
                json.dump(bounds_json, f)
            self.manifest["bounds"] = "bounds.json"

        if features is not None:
            # TODO: Write to self.manifest
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

        [documentation](https://github.com/allen-cell-animated/colorizer-data/blob/main/documentation/DATA_FORMAT.md#Dataset)

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
        }
        output_json.update(self.manifest)

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
            json.dump(output_json, f, indent=2)

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

        [documentation](https://github.com/allen-cell-animated/colorizer-data/blob/main/documentation/DATA_FORMAT.md#3-frames)
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
