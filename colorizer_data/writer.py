import json
import logging
import multiprocessing
import os
import pathlib
import shutil
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

from colorizer_data.types import (
    BackdropMetadata,
    ColorizerMetadata,
    DatasetManifest,
    FeatureInfo,
    FeatureMetadata,
    FeatureType,
    Frames3dMetadata,
)
from colorizer_data.utils import (
    DEFAULT_FRAME_PREFIX,
    DEFAULT_FRAME_SUFFIX,
    _get_frame_count_from_3d_source,
    cast_feature_to_info_type,
    copy_remote_or_local_file,
    generate_frame_paths,
    get_duplicate_items,
    make_relative_image_paths,
    merge_dictionaries,
    read_data_array_file,
    replace_out_of_bounds_values_with_nan,
    sanitize_key_name,
    MAX_CATEGORIES,
    NumpyValuesEncoder,
    update_metadata,
    write_data_array,
)

MAX_SEG_ID_GAP = 10


class ColorizerDatasetWriter:
    """
    Writes provided data as Colorizer-compatible dataset files to the configured output directory.

    Args:
      output_dir (`str | pathlib.Path`): The output directory to write the dataset to.
      dataset (`str`): The name of the dataset. The dataset will be written to a subdirectory with this name.

    Keyword Args:
        scale (`float`): The scale of the data. Defaults to 1.
        force_overwrite (`bool`): If False (default), updates fields in the manifest file if one already exists
        in the output directory. If True, overwrites the existing manifest file without
        reading existing data.

    The dataset subdirectory will contain a `manifest.json` and additional dataset files,
    following the data schema described in the project documentation. (See [DATA_FORMAT.md](https://github.com/allen-cell-animated/colorizer-data/blob/main/documentation/DATA_FORMAT.md) for
    more details.)
    """

    outpath: pathlib.Path
    default_dataset_name: str
    manifest: DatasetManifest
    metadata: ColorizerMetadata
    backdrops: Dict[str, BackdropMetadata]
    features: Dict[str, FeatureMetadata]
    scale: float

    def __init__(
        self,
        output_dir: Union[str, pathlib.Path],
        dataset: str,
        scale: float = 1,
        force_overwrite: bool = False,
    ):
        self.outpath = pathlib.Path(os.path.join(output_dir, dataset))
        os.makedirs(self.outpath, exist_ok=True)
        self.scale = scale

        # Check for existence of manifest file in expected path
        manifest_path = self.outpath / "manifest.json"
        if os.path.exists(manifest_path) and not force_overwrite:
            # Load in the existing file
            try:
                with open(manifest_path, "r") as f:
                    self.manifest = json.load(f)
                logging.info(
                    "An existing manifest file was found in the output directory and will be updated."
                )
            except Exception as e:
                logging.error(e)
                logging.warning(
                    "A manifest file exists in this output directory but could not be loaded, and will be overwritten instead!"
                )
                self.manifest = {}
        else:
            self.manifest = {}

        # Clear features
        self.manifest["features"] = []
        self.features = {}

        # Load backdrops from existing manifest, if applicable
        self.backdrops = {}
        if "backdrops" in self.manifest:
            for backdrop_metadata in self.manifest["backdrops"]:
                self.backdrops[backdrop_metadata["key"]] = backdrop_metadata

        self.default_dataset_name = dataset
        if "metadata" not in self.manifest:
            # New default metadata
            self.metadata = ColorizerMetadata()
        else:
            self.metadata = ColorizerMetadata.from_dict(self.manifest["metadata"])

    def write_categorical_feature(
        self,
        data: np.ndarray,
        info: FeatureInfo,
        *,
        write_json: bool = False,
    ) -> None:
        """
        Writes a categorical feature data array and stores feature metadata to be written to the manifest. See
        `write_feature` for full description of file writing behavior and naming.

        Skips features that have more than 12 categories.

        Args:
            data (`np.ndarray`): An array with dtype string, with no more than 12 unique values. Categories will be ordered
            in order of appearance in `data`.
            info (`FeatureInfo`): Metadata for the feature. The `categories` array and `type` will be overridden.
            write_json (`bool`): Whether to write the feature data as a `.json` file rather than the default Parquet format.
                Compatible with TFE viewer >= v1.1.0. Default is `False`.
        """
        categories, indexed_data = np.unique(data.astype(str), return_inverse=True)
        if len(categories) > MAX_CATEGORIES:
            logging.warning(
                "write_feature_categorical: Too many unique categories in provided data for feature column '{}' ({} > max {}).".format(
                    info.get_name(), len(categories), MAX_CATEGORIES
                )
            )
            logging.warning("\tFEATURE WILL BE SKIPPED.")
            logging.warning("\tCategories provided: {}".format(categories))
            return
        info.categories = categories.tolist()
        info.type = FeatureType.CATEGORICAL
        return self.write_feature(indexed_data, info, write_json=write_json)

    def write_feature(
        self,
        data: np.ndarray,
        info: FeatureInfo,
        *,
        outliers: Union[np.ndarray, None] = None,
        write_json: bool = False,
    ) -> None:
        """
        Writes a feature data array and stores feature metadata to be written to the manifest.

        Args:
            data (`np.ndarray[int | float]`): The numpy array for the feature, to be written to a JSON file.
            info (`FeatureInfo`): Metadata for the feature. See `FeatureInfo` for more details.
            outliers (`np.ndarray`): Optional boolean array, where an object `i` is an outlier if `outliers[i] == True`.
                Outliers will not count towards min/max calculation. Ignored if not provided.
            write_json (`bool`): Whether to write the feature data as a `.json` file rather than the default Parquet format.
                Compatible with TFE viewer >= v1.1.0. Default is `False`.

        Feature JSON files are suffixed by index, starting at 0, which increments
        for each call to `write_feature()`. The first feature will have `feature_0.json`,
        the second `feature_1.json`, and so on.

        Feature data will be parsed and cast to data types using `info.type`. If the type is
        `FeatureType.INDETERMINATE`, will attempt to infer the feature type from `data`. See
        `utils.cast_feature_to_info_type` for casting behavior. If types are mismatched or cannot
        be interpreted, skips writing the feature.

        If the feature type is `FeatureType.CATEGORICAL`, values will be interpreted as integer indices into a list of
        string `categories`, defined in `info`. Values that don't match indices in the list
        (e.g., `x < 0` or `x >= len(info.categories)`) will be replaced with `np.NaN`.

        See the [documentation on features](https://github.com/allen-cell-animated/colorizer-data/blob/main/documentation/DATA_FORMAT.md#2-6-features) for more details.
        """
        # TODO: Write feature files using the keys of the features instead

        try:
            data, info = cast_feature_to_info_type(data, info)
        except RuntimeError as error:
            logging.error("RuntimeError: {}".format(error))
            logging.warning(
                "Could not parse feature '{}'. FEATURE WILL BE SKIPPED.".format(
                    info.get_name()
                )
            )

        # Do additional validation on categorical data.
        if info.type == FeatureType.CATEGORICAL:
            if len(info.categories) > MAX_CATEGORIES:
                logging.warning(
                    "Feature '{}' has too many categories ({} > max {}).".format(
                        info.get_name(), len(info.categories), MAX_CATEGORIES
                    )
                )
                logging.warning("\tFEATURE WILL BE SKIPPED.")
                logging.warning(
                    "\tCategories provided (up to first 25): {}".format(
                        info.categories[:25]
                    )
                )
                return
            if np.min(data) < 0 or np.max(data) >= len(info.categories):
                logging.warning(
                    "Feature '{}' has values out of range of the defined categories.".format(
                        info.get_name()
                    )
                )
                logging.warning("\tBad values will be replaced with NaN.")
                replace_out_of_bounds_values_with_nan(data, 0, len(info.categories) - 1)

        num_features = len(self.features.keys())

        file_basename = "feature_" + str(num_features)

        # Calculate min/max
        filtered_data = data
        if outliers is not None:
            filtered_data = data[np.logical_not(outliers)]
        # Exclude NaN + Infinity values from min/max calculation
        filtered_data = filtered_data[np.isfinite(filtered_data)]
        encoder = NumpyValuesEncoder()
        fmin = encoder.default(info.min)
        fmax = encoder.default(info.max)
        if fmin is None:
            try:
                fmin = np.nanmin(filtered_data)
            except ValueError:
                raise ValueError(
                    "ColorizerDatasetWriter.write_feature: Feature '{}' had no finite, non-outlier values when calculating min/max bounds.".format(
                        info.get_name()
                    )
                    + " Provide a min and max property in FeatureInfo to override automatic bounds calculation."
                )
            fmin = encoder.default(fmin)
        if fmax is None:
            try:
                fmax = np.nanmax(filtered_data)
            except ValueError:
                raise ValueError(
                    "ColorizerDatasetWriter.write_feature: Feature '{}' has no finite, non-outlier values when calculating min/max bounds.".format(
                        info.get_name()
                    )
                    + " Provide a min and max property in FeatureInfo to override automatic bounds calculation."
                )
            fmax = encoder.default(fmax)

        # The viewer reads float data as float32, so cast it if needed.
        if data.dtype == np.float64 or data.dtype == np.double:
            data = data.astype(np.float32)

        # Write the feature JSON file
        logging.info("Writing {}...".format(file_basename))
        filename = write_data_array(
            data,
            self.outpath,
            file_basename,
            write_json=write_json,
            min=fmin,
            max=fmax,
        )

        # Write the metadata to the manifest
        key = info.key
        if key == "":
            # Use label, formatting as needed
            key = sanitize_key_name(info.label)

        metadata: FeatureMetadata = {
            "name": info.label,
            "data": filename,
            "unit": info.unit,
            "type": info.type,
            "key": key,
            "min": fmin,
            "max": fmax,
            "description": info.description,
        }
        # Add categories to metadata only if feature is categorical; also do validation here
        if info.type == FeatureType.CATEGORICAL:
            if info.categories is None:
                raise RuntimeError(
                    "write_feature: Feature '{}' has type CATEGORICAL but no categories were provided.".format(
                        info.get_name()
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

        # Update the manifest with this feature data
        # Default to column name if no label is given; throw error if neither is present
        label = info.label or info.column_name
        if not label:
            raise RuntimeError(
                "write_feature: Provided FeatureInfo '{}' has no label or column name.".format(
                    info.get_name()
                )
            )
        if key in self.features.keys():
            # Throw a warning that we are overwriting data
            old_feature_data = self.features[key]
            logging.warning(
                "Feature key '{}' already exists in manifest. Feature '{}' will overwrite existing feature '{}'. Overwriting...".format(
                    key,
                    label,
                    old_feature_data["name"],
                )
            )
        self.features[key] = metadata

    def write_data(
        self,
        tracks: Union[np.ndarray, None] = None,
        times: Union[np.ndarray, None] = None,
        seg_ids: Union[np.ndarray, None] = None,
        centroids_x: Union[np.ndarray, None] = None,
        centroids_y: Union[np.ndarray, None] = None,
        centroids_z: Union[np.ndarray, None] = None,
        outliers: Union[np.ndarray, None] = None,
        bounds: Union[np.ndarray, None] = None,
        *,
        write_json: bool = False,
    ):
        """
        Writes (non-feature) dataset data arrays (such as track, time, centroid, outlier,
        and bounds data) to JSON files.

        Accepts numpy arrays for each file type and writes them to the configured
        output directory according to the data format.

        Args:
            tracks (`np.ndarray`): A 1D numpy array of integer track numbers, where `tracks[i]` is the track number for the `i`th object.
            times (`np.ndarray`): A 1D numpy array of float timestamps, where `times[i]` is the time, in frames, at which the `i`th object
                is visible.
            centroids_x (`np.ndarray`): A 1D numpy array of float x-coordinates for object centroids.
            centroids_y (`np.ndarray`): A 1D numpy array of float y-coordinates for object centroids.
            centroids_z (`np.ndarray`): An optional 1D numpy array of float z-coordinates for object centroids.
            outliers (`np.ndarray`): An optional 1D numpy array of boolean values, where `outliers[i]` is `True` if the `i`th object is an outlier.
            bounds (`np.ndarray`): An optional 1D numpy array of float values. For the `i`th object, the coordinates of the upper left corner are
                `(x: bounds[4i], y: bounds[4i + 1])` and the lower right corner are `(x: bounds[4i + 2], y: bounds[4i + 3])`.
            write_json (`bool`): Whether to write the specified data as a JSON file rather than the default Parquet format.
                Parquet data is compatible with TFE viewer >= v1.1.0. Default is `False`.

        [documentation](https://github.com/allen-cell-animated/colorizer-data/blob/main/documentation/DATA_FORMAT.md#2-1-tracks)
        """
        # TODO check outlier and replace values with NaN or something!
        if outliers is not None:
            logging.info("Writing outliers data...")
            track_filename = write_data_array(
                outliers, self.outpath, "outliers", write_json=write_json
            )
            self.manifest["outliers"] = track_filename

        # Note these must be in same order as features and same row order as the dataframe.
        if tracks is not None:
            logging.info("Writing track data...")
            track_filename = write_data_array(
                tracks, self.outpath, "tracks", write_json=write_json
            )
            self.manifest["tracks"] = track_filename

        if times is not None:
            logging.info("Writing times data...")
            times_filename = write_data_array(
                times, self.outpath, "times", write_json=write_json
            )
            self.manifest["times"] = times_filename

        if seg_ids is not None:
            logging.info("Writing segmentation ID data...")
            seg_ids_filename = write_data_array(
                seg_ids, self.outpath, "seg_ids", write_json=write_json
            )
            self.manifest["segIds"] = seg_ids_filename

        if (
            centroids_x is not None
            or centroids_y is not None
            or centroids_z is not None
        ):
            if centroids_x is None or centroids_y is None:
                raise Exception(
                    f"Both arguments centroids_x and centroids_y must be defined if any centroid coordinates are provided. Got centroids_x: {centroids_x} and centroids_y: {centroids_y}."
                )
            logging.info("Writing centroids data...")
            centroids_stacked = None
            if centroids_z is None:
                centroids_stacked = np.ravel(np.dstack([centroids_x, centroids_y]))
            else:
                centroids_stacked = np.ravel(
                    np.dstack([centroids_x, centroids_y, centroids_z])
                )
            centroids_stacked = centroids_stacked * self.scale
            centroids_stacked = centroids_stacked.astype(int)
            centroids_filename = write_data_array(
                centroids_stacked,
                self.outpath,
                "centroids",
                write_json=write_json,
            )
            self.manifest["centroids"] = centroids_filename

        if bounds is not None:
            logging.info("Writing bounds data...")
            bounds_filename = write_data_array(
                bounds, self.outpath, "bounds", write_json=write_json
            )
            self.manifest["bounds"] = bounds_filename

    def copy_and_add_backdrops(
        self,
        name: str,
        frame_paths: List[str],
        key=None,
        subdir_name: Optional[str] = None,
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
        if key is None or key == "":
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
        if key is None or key == "":
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

    def set_3d_frame_data(self, data: Frames3dMetadata) -> None:
        if data.total_frames is None:
            logging.warning(
                "ColorizerDatasetWriter: The `total_frames` property of the Frames3dMetadata object is `None`. Will attempt to infer the number of frames from the provided data."
            )
            data.total_frames = _get_frame_count_from_3d_source(data.source)
        self.manifest["frames3d"] = data.to_dict()
        # TODO: when DatasetManifest is a dataclass, it can serialize Frames3dMetadata directly
        # instead of needing to call to_dict() here.

    def write_manifest(
        self,
        num_frames: int = None,
        metadata: ColorizerMetadata = None,
    ):
        """
        Writes the final manifest file for the dataset in the configured output directory.

        Must be called **AFTER** all other data is written.

        Args:
            num_frames (int): DEPRECATED. Define to generate the expected paths for frame images.
            metadata (ColorizerMetadata): Metadata to be written with the dataset. Leave fields blank to use existing default values.

        Note that some metadata fields (like `last_modified`, `_writer_version`, `_revision`, and `date_created`) will
        be automatically updated. Add definitions for these fields in the `metadata` argument to override this behavior.

        [documentation](https://github.com/allen-cell-animated/colorizer-data/blob/main/documentation/DATA_FORMAT.md#2-Dataset)
        """

        if num_frames is not None and "frames" not in self.manifest:
            logging.warning(
                "ColorizerDatasetWriter: The argument `num_frames` on `write_manifest` is deprecated and will be removed in the future! Please call `set_frame_paths(generate_frame_paths(num_frames))` instead."
            )
            self.set_frame_paths(generate_frame_paths(num_frames))

        update_metadata(self.metadata, default_name=self.default_dataset_name)

        # Optionally merge new metadata with old
        if metadata is not None:
            self.manifest["metadata"] = merge_dictionaries(
                self.metadata.to_dict(), metadata.to_dict()
            )
        else:
            self.manifest["metadata"] = self.metadata.to_dict()

        self.manifest["backdrops"] = list(self.backdrops.values())
        self.manifest["features"] = list(self.features.values())

        self.validate_dataset()

        with open(self.outpath / "manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=2)

        logging.info("Finished writing dataset.")

    def __check_for_duplicate_keys(self, keys: List[str], key_name: str):
        """Throws an error if duplicates are detected in the list of keys, and prints an error message to the console."""
        duplicate_keys = get_duplicate_items(keys)
        if len(duplicate_keys) > 0:
            logging.error(
                f"All {key_name} keys must be unique! The following duplicated {key_name} keys were detected:"
            )
            logging.error("   [" + ", ".join(duplicate_keys) + "]")
            logging.error("Dataset writing will now halt.")
            raise RuntimeError(
                f"Duplicate {key_name} keys detected in dataset manifest."
            )

    def write_image(
        self,
        seg_remapped: np.ndarray,
        frame_num: int,
        frame_prefix: str = DEFAULT_FRAME_PREFIX,
        frame_suffix: str = DEFAULT_FRAME_SUFFIX,
    ) -> str:
        """
        Writes the current segmented image to a PNG file in the output directory.
        By default, the image will be saved as `frame_{frame_num}.png`.

        IDs for each pixel are stored in the RGBA channels of the image.

        Args:
          seg_remapped (np.ndarray[int]): A 2D numpy array of integers, where each value in the array is the object ID of the
          segmentation that occupies that pixel.
          frame_num (int): The frame number.

        Keyword args:
          frame_prefix (str): The prefix of the file to be written. This can include subdirectory paths. By default, this is `frame_`.
          frame_suffix (str); The suffix of the file to be written. By default, this is `.png`.

        Effects:
          Writes the ID information to an RGB image file at the path `{frame_prefix}{frame_num}{frame_suffix}`. (By default, this looks
          like `frame_n.png`.)

        Returns:
          str: The relative path to the saved image file.

        [documentation](https://github.com/allen-cell-animated/colorizer-data/blob/main/documentation/DATA_FORMAT.md#2-3-frames)
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
        relative_image_path = frame_prefix + str(frame_num) + frame_suffix
        img.save(self.outpath / relative_image_path)
        return relative_image_path

    def validate_dataset(
        self,
    ):
        """
        Logs warnings to the console if any expected files are missing.
        """
        if self.manifest["times"] is None:
            logging.warning("No times JSON information provided!")
        if not os.path.isfile(self.outpath / self.manifest["times"]):
            logging.warning(
                "Times JSON file does not exist at expected path '{}'".format(
                    self.manifest["times"]
                )
            )

        # TODO: Add validation for other required data files

        # Check that segmentation IDs do not have large gaps between them.
        if "segIds" in self.manifest and "times" in self.manifest:
            seg_ids = read_data_array_file(
                os.path.join(self.outpath, self.manifest["segIds"])
            )
            times = read_data_array_file(
                os.path.join(self.outpath, self.manifest["times"])
            )
            # Bin segmentation IDs by time, then check for gaps in the IDs.
            if seg_ids is not None and times is not None:
                timeToSegIds = {}
                for i in range(len(times)):
                    time = times[i]
                    segId = seg_ids[i]
                    if time not in timeToSegIds:
                        timeToSegIds[time] = set()
                    timeToSegIds[time].add(segId)

            gaps: List[(int, int, int)] = []
            for time, segIds in timeToSegIds.items():
                segIds = list(segIds)
                segIds.sort()
                # Check for gaps in the IDs
                for i in range(len(segIds) - 1):
                    if segIds[i + 1] - segIds[i] > MAX_SEG_ID_GAP:
                        gaps.append((time, segIds[i], segIds[i + 1]))
            if len(gaps) > 0:
                logging.warning(
                    f"Segmentation IDs have {len(gaps)} gap(s) greater than {MAX_SEG_ID_GAP}. "
                    "This may cause performance or out-of-memory issues in the viewer when loading this dataset. "
                    "For best performance, segmentation IDs should be contiguous on each frame."
                )
                logging.warning(
                    "The following gap(s) were detected (showing up to 10 max):"
                )
                for i in range(min(10, len(gaps))):
                    time, segId1, segId2 = gaps[i]
                    logging.warning(
                        f"  Time {time}: Segmentation ID gap between {segId1} and {segId2}."
                    )

        # Check that all features + backdrops have unique keys. This should be guaranteed because
        # they are stored as dictionaries before writing.
        feature_keys = [feature["key"] for feature in self.manifest["features"]]
        self.__check_for_duplicate_keys(feature_keys, "feature")
        if "backdrops" in self.manifest.keys():
            backdrop_keys = [backdrop["key"] for backdrop in self.manifest["backdrops"]]
            self.__check_for_duplicate_keys(backdrop_keys, "backdrop")

        # Check for missing frames
        if "frames" not in self.manifest and "frames3d" not in self.manifest:
            logging.warning(
                "No frames are provided! Did you forget to call `set_frame_paths` on the writer?"
            )
        elif "frames" in self.manifest:
            # Check that all the 2D frame paths exist
            missing_frames = []
            for i in range(len(self.manifest["frames"])):
                path = self.manifest["frames"][i]
                if not os.path.isfile(self.outpath / path):
                    missing_frames.append([i, path])
            if len(missing_frames) > 0:
                logging.warning(
                    "{} image frame(s) missing from the dataset! The following files could not be found:".format(
                        len(missing_frames)
                    )
                )
                for i in range(len(missing_frames)):
                    index, path = missing_frames[i]
                    logging.warning("  {}: '{}'".format(index, path))
                logging.warning(
                    "For auto-generated frame numbers, check that no frames are missing data in the original dataset,"
                    + " or add an offset if your frame numbers do not start at 0."
                    + " You may also need to generate the list of frames yourself if your dataset is skipping frames."
                )
