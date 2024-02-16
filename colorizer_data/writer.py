from dataclasses import dataclass
from enum import Enum
import json
import logging
import os
import pathlib
from typing import List, TypedDict, Union

import numpy as np
from PIL import Image

from colorizer_data.utils import (
    DEFAULT_FRAME_PREFIX,
    DEFAULT_FRAME_SUFFIX,
    generate_frame_paths,
    sanitize_key_name,
    MAX_CATEGORIES,
    NumpyValuesEncoder,
)


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
                logging.warn(
                    "For auto-generated frame numbers, check that no frames are missing data in the original dataset,"
                    + " or add an offset if your frame numbers do not start at 0."
                    + " You may also need to generate the list of frames yourself if your dataset is skipping frames."
                )
