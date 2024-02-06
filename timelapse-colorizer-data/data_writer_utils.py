from dataclasses import dataclass
from enum import Enum
import json
import logging
import os
import pathlib
import platform
import re
from typing import List, Sequence, TypedDict, Union

import numpy as np
import pandas as pd
import skimage
from PIL import Image

MAX_CATEGORIES = 12
INITIAL_INDEX_COLUMN = "initialIndex"
DEFAULT_FRAME_PREFIX = "frame_"
DEFAULT_FRAME_SUFFIX = ".png"
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


class FeatureType(str, Enum):
    CONTINUOUS = "continuous"
    """For continuous decimal values."""
    DISCRETE = "discrete"
    """For integer-only values."""
    CATEGORICAL = "categorical"
    """For category labels. Can include up to 12 string categories, stored in the feature's
    `categories` field. The feature data value for each object ID should be the
    integer index of the name in the `categories` field.
    """


@dataclass
class FeatureInfo:
    """Represents a feature's metadata.

    Args:
        - `label` (`str`): The human-readable name of the feature. Empty string (`""`) by default.
        - `column_name` (`str`): The column name, in the dataset, of the feature. Used for the feature's name
        if no label is provided. Empty string (`""`) by default.
        - `key` (`str`): The internal key name of the feature. Formats the feature label if no
        key is provided. Empty string (`""`) by default.
        - `unit` (`str`): Units this feature is measured in. Empty string (`""`) by default.
        - `type` (`FeatureType`): The type, either continuous, discrete, or categorical, of this feature.
        `FeatureType.CONTINUOUS` by default.
        - `categories` (`List`[str]): The ordered categories for categorical features. `None` by default.
    """

    label: str = ""
    key: str = ""
    column_name: str = ""
    unit: str = ""
    type: FeatureType = FeatureType.CONTINUOUS
    categories: Union[List[str], None] = None


class FeatureMetadata(TypedDict):
    """For data writer internal use. Represents the metadata that will be saved for each feature."""

    data: str
    key: str
    name: str
    """The relative path from the manifest to the feature JSON file."""
    unit: str
    type: FeatureType
    categories: List[str]


class FrameDimensions(TypedDict):
    """Dimensions of each frame, in physical units (not pixels)."""

    units: str
    width: float
    """Width of a frame in physical units (not pixels)."""
    height: float
    """Height of a frame in physical units (not pixels)."""


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
    start_frame_num: int = 0

    def to_json(self) -> DatasetMetadata:
        return {
            "frameDims": {
                "width": self.frame_width,
                "height": self.frame_height,
                "units": self.frame_units,
            },
            "startTimeSeconds": self.start_time_sec,
            "frameDurationSeconds": self.frame_duration_sec,
            "startingFrameNumber": self.start_frame_num,
        }


class DatasetManifest(TypedDict):
    features: List[FeatureMetadata]
    outliers: str
    tracks: str
    centroids: str
    times: str
    bounds: str
    metadata: DatasetMetadata
    frames: List[str]


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
        self.manifest = {"features": []}

    def write_feature(self, data: np.ndarray, info: FeatureInfo):
        """
        Writes feature data arrays and stores feature metadata to be written to the manifest.

        Args:
            data (`np.ndarray[int | float]`): The numeric numpy array for the feature, to be written to a JSON file.
            info (`FeatureInfo`): Metadata for the feature.

        Feature JSON files are suffixed by index, starting at 0, which increments
        for each call to `write_feature()`. The first feature will have `feature_0.json`,
        the second `feature_1.json`, and so on.

        If the feature type is `FeatureType.CATEGORICAL`, `categories` must be defined in `info`.

        See the [documentation on features](https://github.com/allen-cell-animated/colorizer-data/blob/main/documentation/DATA_FORMAT.md#6-features) for more details.
        """
        # Fetch feature data
        num_features = len(self.manifest["features"])
        fmin = np.nanmin(data)
        fmax = np.nanmax(data)
        filename = "feature_" + str(num_features) + ".json"
        file_path = self.outpath + "/" + filename

        key = info.key
        if key == "":
            # Use label, formatting as needed
            key = sanitize_key_name(info.label)

        # Create manifest from feature data
        metadata: FeatureMetadata = {
            "name": info.label,
            "data": filename,
            "unit": info.unit,
            "type": info.type,
            "key": key,
        }

        # Add categories to metadata only if feature is categorical; also do validation here
        if info.type == FeatureType.CATEGORICAL:
            if info.categories is None:
                raise RuntimeError(
                    "write_feature: Feature '{}' has type CATEGORICAL but no categories were provided.".format(
                        info.label
                    )
                )
            if len(info.categories) > MAX_CATEGORIES:
                raise RuntimeError(
                    "write_feature: Cannot exceed maximum number of categories ({} > {})".format(
                        len(info.categories), MAX_CATEGORIES
                    )
                )
            metadata["categories"] = info.categories
            # TODO cast to int, but handle NaN?

        # Write the feature JSON file
        logging.info("Writing {}...".format(filename))
        js = {"data": data.tolist(), "min": fmin, "max": fmax}
        with open(file_path, "w") as f:
            json.dump(js, f, cls=NumpyValuesEncoder)

        # Update the manifest with this feature data
        # Default to column name if no label is given; throw error if neither is present
        label = info.label or info.column_name
        if not label:
            raise RuntimeError(
                "write_feature: Provided FeatureInfo has no label or column name."
            )

        self.manifest["features"].append(metadata)

    def write_data(
        self,
        tracks: Union[np.ndarray, None] = None,
        times: Union[np.ndarray, None] = None,
        centroids_x: Union[np.ndarray, None] = None,
        centroids_y: Union[np.ndarray, None] = None,
        outliers: Union[np.ndarray, None] = None,
        bounds: Union[np.ndarray, None] = None,
    ):
        """
        Writes (non-feature) dataset data arrays (such as track, time, centroid, outlier,
        and bounds data) to JSON files.

        Accepts numpy arrays for each file type and writes them to the configured
        output directory according to the data format.

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

    def set_frame_paths(self, paths: List[str]) -> None:
        """
        Stores an ordered array of paths to image frames, to be written
        to the manifest. Paths should be are relative to the dataset directory.

        Use `generate_frame_paths()` if your frame numbers are contiguous (no gaps or skips).
        """
        self.manifest["frames"] = paths

    def write_manifest(
        self,
        num_frames: int = None,
        metadata: ColorizerMetadata = None,
    ):
        """
        Writes the final manifest file for the dataset in the configured output directory.

        Must be called **AFTER** all other data is written.

        [documentation](https://github.com/allen-cell-animated/colorizer-data/blob/main/documentation/DATA_FORMAT.md#Dataset)
        """

        if num_frames is not None and self.manifest["frames"] is None:
            logging.warn(
                "ColorizerDatasetWriter: The argument `num_frames` on `write_manifest` is deprecated and will be removed in the future! Please call `set_frame_paths(generate_frame_paths(num_frames))` instead."
            )
            self.set_frame_paths(generate_frame_paths(num_frames))

        # Add the metadata
        if metadata:
            self.manifest["metadata"] = metadata.to_json()

        self.validate_dataset()

        with open(self.outpath + "/manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=2)

        logging.info("Finished writing dataset.")

    def write_image(
        self,
        seg_remapped: np.ndarray,
        frame_num: int,
        frame_prefix: str = DEFAULT_FRAME_PREFIX,
        frame_suffix: str = DEFAULT_FRAME_SUFFIX,
    ):
        """
        Writes the current segmented image to a PNG file in the output directory.
        By default, the image will be saved as `frame_{frame_num}.png`.

        IDs for each pixel are stored in the RGBA channels of the image.

        Args:
          seg_remapped (np.ndarray[int]): A 2D numpy array of integers, where each value in the array is the object ID of the
          segmentation that occupies that pixel.
          frame_num (int): The frame number.

        Positional args:
          frame_prefix (str): The prefix of the file to be written. This can include subdirectory paths. By default, this is `frame_`.
          frame_suffix (str); The suffix of the file to be written. By default, this is `.png`.

        Effects:
          Writes the ID information to an RGB image file at the path `{frame_prefix}{frame_num}{frame_suffix}`. (By default, this looks
          like `frame_n.png`.)

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
        # TODO: Automatically create subdirectories if `frame_prefix` contains them.
        img.save(self.outpath + "/" + frame_prefix + str(frame_num) + frame_suffix)

    def validate_dataset(
        self,
    ):
        """
        Logs warnings to the console if any expected files are missing.
        """
        if self.manifest["times"] is None:
            logging.warn("No times JSON information provided!")
        if not os.path.isfile(self.outpath + "/" + self.manifest["times"]):
            logging.warn(
                "Times JSON file does not exist at expected path '{}'".format(
                    self.manifest["times"]
                )
            )

        # TODO: Add validation for other required data files

        if self.manifest["frames"] is None:
            logging.warn(
                "No frames are provided! Did you forget to call `set_frame_paths` on the writer?"
            )
        else:
            # Check that all the frame paths exist
            missing_frames = []
            for i in range(len(self.manifest["frames"])):
                path = self.manifest["frames"][i]
                if not os.path.isfile(self.outpath + "/" + path):
                    missing_frames.append([i, path])
            if len(missing_frames) > 0:
                logging.warn(
                    "{} image frame(s) missing from the dataset! The following files could not be found:".format(
                        len(missing_frames)
                    )
                )
                for i in range(len(missing_frames)):
                    index, path = missing_frames[i]
                    logging.warn("  {}: '{}'".format(index, path))
                logging.warn("For auto-generated frame numbers, you may need to ")
