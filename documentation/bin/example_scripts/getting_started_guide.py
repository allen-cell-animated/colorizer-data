"""
This is a non-notebook version of the getting started guide that can be run as a
script. Feel free to use this as a template when converting your own dataset.

# From the root directory of the repository, run:
```
# 1. Python environment (optional):
python -m venv .venv
source .venv/bin/activate
# On Windows use `.venv\Scripts\activate`

# 2. Install dependencies:
cd documentation/getting_started_guide
pip install -r requirements.txt

# 3. Run the script:
python ../bin/example_scripts/getting_started_guide.py

# 4. Open the viewer:
python ../../colorizer_data/bin/tfe_open.py processed_dataset/collection.json

# If you install the project in editable mode (`pip install -e .` from the
# root directory), you can also run `tfe-open.py` from anywhere:
tfe-open processed_dataset/collection.json
```
"""

from pathlib import Path

from colorizer_data import (
    convert_colorizer_data,
    FeatureInfo,
    FeatureType,
    ColorizerMetadata,
    CollectionMetadata,
    update_collection,
)
import pandas as pd


source_directory = Path("raw_datasets")
dataset_directories = ["dataset_1", "dataset_2"]

output_directory = Path("processed_dataset")
collection_path = output_directory / "collection.json"

# Feature metadata: change this based on your dataset's features.
area_info = FeatureInfo(
    label="Area",
    key="area",
    type=FeatureType.CONTINUOUS,
    unit="px²",
    description="Area of object in square pixels, calculated from radius.",
)
radius_info = FeatureInfo(
    label="Radius",
    key="radius",
    # Discrete features are used for integers.
    type=FeatureType.DISCRETE,
    unit="px",
    description="Radius of object in pixels.",
)
location_info = FeatureInfo(
    label="Location",
    key="location",
    # Categorical features are used for string-based labels.
    type=FeatureType.CATEGORICAL,
    # Categories can be auto-detected from the data, or provided manually
    # if you want to preserve a specific order for the labels.
    categories=["top", "middle", "bottom"],
    description="Y position of object's centroid in the frame, as either 'top' (y < 40%), 'middle' (40% ≤ y ≤ 60%), or 'bottom' (y > 60%) of the frame.",
)
feature_info = {"area": area_info, "radius": radius_info, "location": location_info}

# Collection metadata
collection_metadata = CollectionMetadata(
    name="Example collection",
    description="An example collection of datasets for the Timelapse Feature Explorer!",
    author="Author name et al.",
    # Change this as needed.
    collection_version="v1.0",
)


def main():
    # Load the dataset
    for i in range(len(dataset_directories)):
        dataset_dir_name = dataset_directories[i]
        dataset_src_dir = source_directory / dataset_dir_name
        dataset_out_dir = output_directory / dataset_dir_name
        dataset_out_dir.mkdir(parents=True, exist_ok=True)

        # Dataset metadata
        metadata = ColorizerMetadata(
            name="Example dataset {}".format(i + 1),
            description="An example dataset for the Timelapse Feature Explorer.",
            author="Author name et al.",
            dataset_version="some.version.here",
            # The width and height of the original segmentations images, in
            # units defined by `frame_units`. This configures the scale bar in
            # the viewer.
            frame_width=100,
            frame_height=100,
            frame_units="nm",
            # Time elapsed between each frame capture, in seconds.
            frame_duration_sec=1,
        )

        # Convert the dataset
        data: pd.DataFrame = pd.read_csv(dataset_src_dir / "data.csv")
        convert_colorizer_data(
            data,
            dataset_out_dir,
            source_dir=dataset_src_dir,
            metadata=metadata,
            feature_info=feature_info,
            object_id_column="segmentation_id",
            times_column="time",
            track_column="track_id",
            image_column="segmentation_path",
            centroid_x_column="centroid_x",
            centroid_y_column="centroid_y",
        )

        # Add to collection
        path_relative_to_collection = str(dataset_out_dir.relative_to(output_directory))
        update_collection(
            collection_path,
            # Display name
            "Dataset {}".format(i + 1),
            # Path relative to collection file
            path_relative_to_collection,
            metadata=collection_metadata,
        )


if __name__ == "__main__":
    main()
