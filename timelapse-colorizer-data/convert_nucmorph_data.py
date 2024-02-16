from dataclasses import dataclass
import multiprocessing
from typing import List, Sequence
from aicsimageio import AICSImage
import argparse
import logging
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
import time

# requirements
# "aicsfiles @ https://artifactory.corp.alleninstitute.org/artifactory/api/pypi/pypi-release-local/aicsfiles/5.1.0/aicsfiles-5.1.0.tar.gz",
# Requires us to use Python 3.8. See https://github.com/aics-int/nuc-morph-analysis/blob/main/docs/INSTALL.md#basic-installation-instructions-with-conda-and-pip.
# "nuc-morph-analysis @ git+https://github.com/aics-int/nuc-morph-analysis.git"
from nuc_morph_analysis.utilities.create_base_directories import create_base_directories
from nuc_morph_analysis.lib.preprocessing.load_data import (
    load_dataset,
    get_dataset_pixel_size,
)
from nuc_morph_analysis.lib.visualization.plotting_tools import (
    get_plot_labels_for_metric,
)
from nuc_morph_analysis.lib.visualization.add_features_for_colorizer import (
    add_growth_features
)
from data_writer_utils import (
    INITIAL_INDEX_COLUMN,
    ColorizerDatasetWriter,
    ColorizerMetadata,
    FeatureInfo,
    FeatureType,
    configureLogging,
    generate_frame_paths,
    get_total_objects,
    make_bounding_box_array,
    sanitize_path_by_platform,
    scale_image,
    remap_segmented_image,
    update_bounding_box_data,
)


@dataclass
class NucMorphFeatureSpec:
    column_name: str
    type: FeatureType = FeatureType.CONTINUOUS
    categories: List = None


# Example Commands:
# pip install https://artifactory.corp.alleninstitute.org/artifactory/api/pypi/pypi-release-local/aicsfiles/5.1.0/aicsfiles-5.1.0.tar.gz git+https://github.com/aics-int/nuc-morph-analysis.git
# python timelapse-colorizer-data/convert_nucmorph_data.py --output_dir /allen/aics/animated-cell/Dan/fileserver/colorizer/data --dataset mama_bear --scale 0.25
# python timelapse-colorizer-data/convert_nucmorph_data.py --output_dir /allen/aics/animated-cell/Dan/fileserver/colorizer/data --dataset baby_bear --scale 0.25
# python timelapse-colorizer-data/convert_nucmorph_data.py --output_dir /allen/aics/animated-cell/Dan/fileserver/colorizer/data --dataset goldilocks --scale 0.25

# NOTE: If you are regenerating the dataset but have NOT changed the segmentations/object IDs, add the option `--noframes` to skip the frame generation step!

# DATASET SPEC: See DATA_FORMAT.md for more details on the dataset format!
# You can find the most updated version on GitHub here:
# https://github.com/allen-cell-animated/colorizer-data/blob/main/documentation/DATA_FORMAT.md

# NUCMORPH DATA REFERENCE:
# dataset	string	In FMS manifest	Name of which dataset this row of data belongs to (baby_bear, goldilocks, or mama_bear)
# track_id	int	In FMS manifest	ID for a single nucleus in all frames for which it exists (single value per nucleus, consistent across multiple frames)
# CellID	hash	In FMS manifest	ID for a single instance/frame of a nucleus (every nucleus has a different value in every frame)
# index_sequence	int	In FMS manifest	frame number associated with the nucleus data in a given row, relative to the start of the movie
# colony_time	int	Needs calculated and added	Frame number staggered by a given amount per dataset, so that the frame numbers in all datasets are temporally algined relative to one another rather than all starting at 0
# raw_full_zstack_path	String	In FMS manifest	Path to zstack of raw image of entire colony in a single frame
# seg_full_zstack_path	String	In FMS manifest	Path to zstack of segmentation of entire colony in a single frame
# is_outlier	boolean	In FMS manifest	True if this nucleus in this frame is flagged as an outlier (a single nucleus may be an outlier in some frames but not others)
# edge_cell	boolean	In FMS manifest	True if this nucleus touches the edge of the FOV
# NUC_shape_volume_lcc	float	In FMS manifest	Volume of a single nucleus in pixels in a given frame
# NUC_position_depth	float	In FMS manifest	Height (in the z-direction) of the a single nucleus in pixels in a given frame
# NUC_PC1	float	Needs calculated and added	Value for shape mode 1 for a single nucleus in a given frame
# NUC_PC2	float	Needs calculated and added	Value for shape mode 2 for a single nucleus in a given frame
# NUC_PC3	float	Needs calculated and added	Value for shape mode 3 for a single nucleus in a given frame
# NUC_PC4	float	Needs calculated and added	Value for shape mode 4 for a single nucleus in a given frame
# NUC_PC5	float	Needs calculated and added	Value for shape mode 5 for a single nucleus in a given frame
# NUC_PC6	float	Needs calculated and added	Value for shape mode 6 for a single nucleus in a given frame
# NUC_PC7	float	Needs calculated and added	Value for shape mode 7 for a single nucleus in a given frame
# NUC_PC8	float	Needs calculated and added	Value for shape mode 8 for a single nucleus in a given frame

# OVERWRITE THESE!! These values should change based on your dataset. These are
# relabeled as constants here for clarity/intent of the column name.
OBJECT_ID_COLUMN = "label_img"
"""Column of object IDs (or unique row number)."""
TRACK_ID_COLUMN = "track_id"
"""Column of track ID for each object."""
TIMES_COLUMN = "index_sequence"
"""Column of frame number that the object ID appears in."""
SEGMENTED_IMAGE_COLUMN = "seg_full_zstack_path"
"""Column of path to the segmented image data or z stack for the frame."""
CENTROIDS_X_COLUMN = "centroid_x"
"""Column of X centroid coordinates, in pixels of original image data."""
CENTROIDS_Y_COLUMN = "centroid_y"
"""Column of Y centroid coordinates, in pixels of original image data."""
OUTLIERS_COLUMN = "is_outlier"
"""Column of outlier status for each object. (true/false)"""

"""Columns of feature data to include in the dataset. Each column will be its own feature file."""
FEATURE_COLUMNS = [
    NucMorphFeatureSpec("NUC_shape_volume_lcc"),
    NucMorphFeatureSpec("NUC_position_depth"),
    NucMorphFeatureSpec("NUC_position_height"),
    NucMorphFeatureSpec("NUC_position_width"),
    NucMorphFeatureSpec("xy_aspect"),
    # 0 - track terminates by dividing
    # 1 - track terminates by going off the edge of the FOV
    # 2 - track terminates by apoptosis
    NucMorphFeatureSpec("Volume_change_BC"),
    NucMorphFeatureSpec("Volume_foldchange_BC"),
    NucMorphFeatureSpec("Late_growth_rate_fitted"),
    NucMorphFeatureSpec("Late_growth_duration"),
    NucMorphFeatureSpec("colony_depth"),
    NucMorphFeatureSpec("density"),
    NucMorphFeatureSpec(
        "termination", FeatureType.CATEGORICAL, ["Division", "Leaves FOV", "Apoptosis"]
    ),
    NucMorphFeatureSpec("parent_id", FeatureType.DISCRETE),
    NucMorphFeatureSpec("family_id", FeatureType.DISCRETE),
    NucMorphFeatureSpec("is_outlier", FeatureType.CATEGORICAL, ["False", "True"]),
    NucMorphFeatureSpec("edge_cell", FeatureType.CATEGORICAL, ["False", "True"]),
]


def get_image_from_row(row: pd.DataFrame) -> AICSImage:
    zstackpath = row[SEGMENTED_IMAGE_COLUMN]
    zstackpath = sanitize_path_by_platform(zstackpath)
    return AICSImage(zstackpath)


def make_frame(
    grouped_frames: DataFrameGroupBy,
    group_name: int,
    frame: pd.DataFrame,
    scale: float,
    bounds_arr: Sequence[int],
    writer: ColorizerDatasetWriter,
):
    start_time = time.time()

    # Get the path to the segmented zstack image frame from the first row (should be the same for
    # all rows in this group, since they are all on the same frame).
    row = frame.iloc[0]
    frame_number = row[TIMES_COLUMN]
    # Flatten the z-stack to a 2D image.
    aics_image = get_image_from_row(row)
    zstack = aics_image.get_image_data("ZYX", S=0, T=0, C=0)
    seg2d = zstack.max(axis=0)

    # Scale the image and format as integers.
    seg2d = scale_image(seg2d, scale)
    seg2d = seg2d.astype(np.uint32)

    # Remap the frame image so the IDs are unique across the whole dataset.
    seg_remapped, lut = remap_segmented_image(
        seg2d,
        frame,
        OBJECT_ID_COLUMN,
    )

    writer.write_image(seg_remapped, frame_number)
    update_bounding_box_data(bounds_arr, seg_remapped)

    time_elapsed = time.time() - start_time
    logging.info(
        "Frame {} finished in {:5.2f} seconds.".format(int(frame_number), time_elapsed)
    )


def make_frames_parallel(
    grouped_frames: DataFrameGroupBy,
    scale: float,
    writer: ColorizerDatasetWriter,
):
    """
    Generate the images and bounding boxes for each time step in the dataset.
    """
    nframes = len(grouped_frames)
    total_objects = get_total_objects(grouped_frames)
    logging.info("Making {} frames...".format(nframes))

    with multiprocessing.Manager() as manager:
        bounds_arr = manager.Array("i", [0] * int(total_objects * 4))
        with multiprocessing.Pool() as pool:
            pool.starmap(
                make_frame,
                [
                    (grouped_frames, group_name, frame, scale, bounds_arr, writer)
                    for group_name, frame in grouped_frames
                ],
            )
        writer.write_data(bounds=np.array(bounds_arr, dtype=np.uint32))


def make_features(
    dataset: pd.DataFrame,
    features: List[NucMorphFeatureSpec],
    dataset_name: str,
    writer: ColorizerDatasetWriter,
):
    """
    Generate the outlier, track, time, centroid, and feature data files.
    """
    # Collect array data from the dataframe for writing.
    outliers = dataset[OUTLIERS_COLUMN].to_numpy()
    tracks = dataset[TRACK_ID_COLUMN].to_numpy()
    times = dataset[TIMES_COLUMN].to_numpy()
    centroids_x = dataset[CENTROIDS_X_COLUMN].to_numpy()
    centroids_y = dataset[CENTROIDS_Y_COLUMN].to_numpy()

    writer.write_data(
        tracks=tracks,
        times=times,
        centroids_x=centroids_x,
        centroids_y=centroids_y,
        outliers=outliers,
    )

    # Write the feature data
    formatted_units = {
        "": None,
        "($\mu m$)": "µm",
        "($\mu m^3$)": "µm³",
        "($\mu m^3$/hr)": "µm³/hr",
        "(min)": "min",
        "($\mu m^{-1}$)": "µm⁻¹",
    }
    for feature in features:
        if feature.column_name not in dataset.columns:
            logging.warning(
                "Feature '{}' not found in dataset. Skipping...".format(
                    feature.column_name
                )
            )
            continue

        (scale_factor, label, unit) = get_plot_labels_for_metric(
            feature.column_name, dataset=dataset_name
        )
        unit = formatted_units.get(unit)

        data = dataset[feature.column_name]

        # Get data and scale to use actual units
        if scale_factor is not None:
            data = data * scale_factor

        writer.write_feature(
            data,
            FeatureInfo(
                label=label, unit=unit, type=feature.type, categories=feature.categories
            ),
        )


def get_dataset_dimensions(
    grouped_frames: DataFrameGroupBy, pixsize: float
) -> (float, float, str):
    """Get the dimensions of the dataset from the first frame, in units.
    Returns (width, height, unit)."""
    row = grouped_frames.get_group(0).iloc[0]
    aics_image = get_image_from_row(row)
    dims = aics_image.dims
    return (dims.X * pixsize, dims.Y * pixsize, "µm")


def make_dataset(output_dir="./data/", dataset="baby_bear", do_frames=True, scale=1):
    """Make a new dataset from the given data, and write the complete dataset
    files to the given output directory.
    """
    writer = ColorizerDatasetWriter(output_dir, dataset, scale=scale)

    # use nucmorph to load data
    datadir, figdir = create_base_directories(dataset)
    pixsize = get_dataset_pixel_size(dataset)

    df_original = load_dataset(dataset, datadir=None)
    logging.info("Loaded dataset '" + str(dataset) + "'.")

    df_add_features = add_growth_features(df_original)
    full_dataset: pd.DataFrame = df_add_features
    logging.info("Calculated and added growth features.")

    # Make a reduced dataframe grouped by time (frame number).
    columns = [TRACK_ID_COLUMN, TIMES_COLUMN, SEGMENTED_IMAGE_COLUMN, OBJECT_ID_COLUMN]
    reduced_dataset = full_dataset[columns]
    reduced_dataset = reduced_dataset.reset_index(drop=True)
    reduced_dataset[INITIAL_INDEX_COLUMN] = reduced_dataset.index.values
    grouped_frames = reduced_dataset.groupby(TIMES_COLUMN)

    dims = get_dataset_dimensions(grouped_frames, pixsize)
    metadata = ColorizerMetadata(
        frame_width=dims[0], frame_height=dims[1], frame_units=dims[2]
    )

    # Make the features, frame data, and manifest.
    nframes = len(grouped_frames)
    writer.set_frame_paths(generate_frame_paths(nframes))

    make_features(full_dataset, FEATURE_COLUMNS, dataset, writer)
    if do_frames:
        make_frames_parallel(grouped_frames, scale, writer)
    writer.write_manifest(metadata=metadata)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir",
    type=str,
    default="./data/",
    help="Parent directory to output to. Data will be written to a subdirectory named after the dataset parameter.",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="baby_bear",
    help="Compatible named FMS dataset or FMS id to load. Will be loaded using `nuc_morph_analysis.preprocessing.load_data.load_dataset()`.",
)
parser.add_argument(
    "--noframes",
    action="store_true",
    help="If included, generates only the feature data, centroids, track data, and manifest, skipping the frame and bounding box generation.",
)
parser.add_argument(
    "--scale",
    type=float,
    default=1.0,
    help="Uniform scale factor that original image dimensions will be scaled by. 1 is original size, 0.5 is half-size in both X and Y.",
)

args = parser.parse_args()
if __name__ == "__main__":
    configureLogging(args.output_dir)
    logging.info("Starting...")

    make_dataset(
        output_dir=args.output_dir,
        dataset=args.dataset,
        do_frames=not args.noframes,
        scale=args.scale,
    )
