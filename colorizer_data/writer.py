import json
import logging
import multiprocessing
import os
import pathlib
import shutil
from typing import Dict, List, Union

import numpy as np
from PIL import Image
from colorizer_data.types import (
    BackdropMetadata,
    ColorizerMetadata,
    DatasetManifest,
    FeatureInfo,
    FeatureMetadata,
    FeatureType,
)

from colorizer_data.utils import (
    DEFAULT_FRAME_PREFIX,
    DEFAULT_FRAME_SUFFIX,
    copy_remote_or_local_file,
    generate_frame_paths,
    make_relative_image_paths,
    sanitize_key_name,
    MAX_CATEGORIES,
    NumpyValuesEncoder,
)


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
    backdrops: Dict[str, BackdropMetadata]
    scale: float

    def __init__(
        self,
        output_dir: Union[str, pathlib.Path],
        dataset: str,
        scale: float = 1,
        force_overwrite: bool = False,
    ):
        self.outpath = os.path.join(output_dir, dataset)
        os.makedirs(self.outpath, exist_ok=True)
        self.scale = scale

        # Check for existence of manifest file in expected path
        manifest_path = self.outpath + "/manifest.json"
        if os.path.exists(manifest_path) and not force_overwrite:
            # Load in the existing file
            try:
                with open(manifest_path, "r") as f:
                    self.manifest = json.load(f)
                logging.info(
                    "An existing manifest file was found in the output directory and will be updated."
                )
            except:
                logging.warning(
                    "A manifest file exists in this output directory but could not be loaded, and will be overwritten instead!"
                )
                self.manifest = {}
        else:
            self.manifest = {}

        # Clear features
        self.manifest["features"] = []

        # Load backdrops from existing manifest, if applicable
        self.backdrops = {}
        if "backdrops" in self.manifest:
            for backdrop_metadata in self.manifest["backdrops"]:
                self.backdrops[backdrop_metadata["key"]] = backdrop_metadata

    def write_categorical_feature(self, data: np.ndarray, info: FeatureInfo) -> None:
        """
        Writes a categorical feature data array and stores feature metadata to be written to the manifest. See
        `write_feature` for full description of file writing behavior and naming.

        Skips features that have more than 12 categories.

        Args:
            data (`np.ndarray`): An array with dtype string, with no more than 12 unique values. Categories will be ordered
            in order of appearance in `data`.
            info (`FeatureInfo`): Metadata for the feature. The `categories` array and `type` will be overridden.
        """
        categories, indexed_data = np.unique(data.astype(str), return_inverse=True)
        if len(categories) > MAX_CATEGORIES:
            logging.warning(
                "write_feature_categorical: Too many unique categories in provided data for feature column '{}' ({} > max {}).".format(
                    info.column_name, len(categories), MAX_CATEGORIES
                )
            )
            logging.warning("\tFEATURE WILL BE SKIPPED.")
            logging.warning("\tCategories provided: {}".format(categories))
            return
        info.categories = categories.tolist()
        info.type = FeatureType.CATEGORICAL
        return self.write_feature(indexed_data, info)

    def write_feature(self, data: np.ndarray, info: FeatureInfo) -> None:
        """
        Writes a feature data array and stores feature metadata to be written to the manifest.

        Args:
            data (`np.ndarray[int | float]`): The numeric numpy array for the feature, to be written to a JSON file.
            info (`FeatureInfo`): Metadata for the feature.

        Feature JSON files are suffixed by index, starting at 0, which increments
        for each call to `write_feature()`. The first feature will have `feature_0.json`,
        the second `feature_1.json`, and so on.

        If the feature type is `FeatureType.CATEGORICAL`, values will be interpreted as integer indices into a list of
        string `categories`, defined in `info`.

        See the [documentation on features](https://github.com/allen-cell-animated/colorizer-data/blob/main/documentation/DATA_FORMAT.md#6-features) for more details.
        """

        if info.type == FeatureType.CATEGORICAL and info.categories > MAX_CATEGORIES:
            logging.warning(
                "write_feature_categorical: Too many unique categories in provided data for feature column '{}' ({} > max {}).".format(
                    info.column_name, len(info.categories), MAX_CATEGORIES
                )
            )
            logging.warning("\tFEATURE WILL BE SKIPPED.")
            logging.warning("\tCategories provided: {}".format(info.categories))
            return

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

    def copy_and_add_backdrops(
        self,
        name: str,
        frame_paths: List[str],
        key=None,
        subdir_name: str = None,
        clear_subdir: bool = True,
    ) -> None:
        """
        Copies a set of backdrop images from the provided paths (either filepaths or URLs) to the
        dataset's output directory, then registers the backdrop image set in the dataset.

        Args:
            name (str): The name of the backdrop set.
            frame_paths (List[str]): The relative paths to the backdrop images.
            key (str): The key of the backdrop set. If not provided, a sanitized version of the name will be used.
            subdir_name (str): The subdirectory to save images to. If not provided, uses the key. The subdirectory will be
            created if it does not exist.
            clear_subdir (bool): Whether to delete the contents of the subdirectory before copying files. True by default.
        """
        # TODO: Scale images with the writer's set scale. Images will likely need to be saved first,
        # then opened, scaled, and saved out again.

        # Make sanitized version of name as key
        if key is None:
            key = sanitize_key_name(name)

        if subdir_name is None:
            subdir_name = key

        # Optionally clear subdirectory. Create it if it does not exist
        subdir_path = os.path.join(self.outpath, subdir_name)
        if clear_subdir and os.path.isdir(subdir_path):
            # Note: this will throw errors if there are read-only files inside the directory.
            shutil.rmtree(subdir_path)
        os.makedirs(subdir_path, exist_ok=True)

        frame_paths = list(map((lambda path: path.strip("'\" \t")), frame_paths))
        relative_paths = make_relative_image_paths(frame_paths, subdir_name)

        with multiprocessing.Pool() as pool:
            pool.starmap(
                copy_remote_or_local_file,
                [
                    (frame_paths[i], os.path.join(self.outpath, relative_paths[i]))
                    for i in range(len(frame_paths))
                ],
            )

        # Save the updated paths and then call add_backdrops
        self.add_backdrops(name, relative_paths, key)

    def add_backdrops(
        self,
        name: str,
        frame_paths: List[str],
        key=None,
    ):
        """
        Registers a backdrop image set to the dataset.

        Args:
            name (str): The name of the backdrop set.
            frame_paths (List[str]): The relative paths to the backdrop images.
            key (str): The key of the backdrop set. If not provided, a sanitized version of the name will be used.
        """
        # Make sanitized version of name as key if not provided
        if key is None:
            key = sanitize_key_name(name)
        if self.backdrops.get(key):
            logging.warning(
                f"Backdrop key '{key}' already exists in manifest. Overwriting..."
            )
        self.backdrops[key] = {"key": key, "name": name, "frames": frame_paths}

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

        self.manifest["backdrops"] = list(self.backdrops.values())

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
