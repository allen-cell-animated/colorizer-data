from io import StringIO
import json
import os
import pathlib

from colorizer_data import convert_colorizer_data
from colorizer_data.types import FeatureMetadata
import pandas as pd
import pytest
from typing import Dict, List, Union


sample_csv_headers = "ID,Track,Frame,Centroid X,Centroid Y,Continuous Feature,Discrete Feature,Categorical Feature,Outlier,File Path"
sample_csv_headers_alternate = "object_id,track,frame,centroid_x,centroid_y,Continuous Feature,Discrete Feature,Categorical Feature,outlier,file_path"
sample_csv_data = """0,1,0,50,50,0.5,0,A,0,./colorizer_data/tests/assets/test_csv/frame_0.tiff
    1,1,1,55,60,0.6,1,B,0,./colorizer_data/tests/assets/test_csv/frame_1.tiff
    2,2,0,60,70,0.7,2,C,0,./colorizer_data/tests/assets/test_csv/frame_0.tiff
    3,2,1,65,75,0.8,3,A,1,./colorizer_data/tests/assets/test_csv/frame_1.tiff"""

# ///////////////////////// METHODS /////////////////////////


@pytest.fixture
def existing_dataset(tmp_path) -> pathlib.Path:
    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    # TODO: Should I just write the relevant data files out without going through the image
    # processing step?
    convert_colorizer_data(csv_data, tmp_path, use_json=True)
    return tmp_path


def feature_array_to_dict(
    feature_array: List[FeatureMetadata],
) -> Dict[str, FeatureMetadata]:
    return {feature.name: feature for feature in feature_array}


def validate_data(path: pathlib.Path, data: Union[List[int], List[float]]):
    loaded_data = []
    if path.suffix == ".json":
        with open(path, "r") as f:
            feature_data = json.load(f)
            loaded_data = feature_data["data"]
    else:
        feature_data = pd.read_parquet(path)
        loaded_data = feature_data["data"]

    assert len(loaded_data) == len(data)
    for i, d in enumerate(data):
        assert abs(loaded_data[i] - d) < 1e-6


def validate_default_dataset(dataset_dir: pathlib.Path, filetype="json"):
    expected_manifest = dataset_dir / "manifest.json"
    assert os.path.exists(expected_manifest)

    manifest = {}
    with open(expected_manifest, "r") as f:
        manifest = json.load(f)

    # Should not include data columns as features
    assert manifest["features"] == [
        {
            "name": "Continuous Feature",
            "key": "continuous_feature",
            "type": "continuous",
            "description": "",
            "unit": "",
            "min": 0.5,
            "max": 0.7,
            "data": f"feature_0.{filetype}",
        },
        {
            "name": "Discrete Feature",
            "key": "discrete_feature",
            "type": "discrete",
            "description": "",
            "unit": "",
            "min": 0,
            "max": 2,
            "data": f"feature_1.{filetype}",
        },
        {
            "name": "Categorical Feature",
            "key": "categorical_feature",
            "type": "categorical",
            "description": "",
            "unit": "",
            "min": 0,
            "max": 2,
            "categories": ["A", "B", "C"],
            "data": f"feature_2.{filetype}",
        },
    ]
    assert os.path.exists(dataset_dir / f"feature_0.{filetype}")
    assert os.path.exists(dataset_dir / f"feature_1.{filetype}")
    assert os.path.exists(dataset_dir / f"feature_2.{filetype}")
    validate_data(dataset_dir / f"feature_0.{filetype}", [0.5, 0.6, 0.7, 0.8])
    validate_data(dataset_dir / f"feature_1.{filetype}", [0, 1, 2, 3])
    validate_data(dataset_dir / f"feature_2.{filetype}", [0, 1, 2, 0])

    assert manifest["frames"] == ["frame_0.png", "frame_1.png"]
    assert os.path.exists(dataset_dir / "frame_0.png")
    assert os.path.exists(dataset_dir / "frame_1.png")

    assert manifest["tracks"] == f"tracks.{filetype}"
    assert os.path.exists(dataset_dir / f"tracks.{filetype}")
    validate_data(dataset_dir / f"tracks.{filetype}", [1, 1, 2, 2])

    assert manifest["centroids"] == f"centroids.{filetype}"
    assert os.path.exists(dataset_dir / f"centroids.{filetype}")
    validate_data(
        dataset_dir / f"centroids.{filetype}", [50, 50, 55, 60, 60, 70, 65, 75]
    )

    assert manifest["times"] == f"times.{filetype}"
    assert os.path.exists(dataset_dir / f"times.{filetype}")
    validate_data(dataset_dir / f"times.{filetype}", [0, 1, 0, 1])

    assert manifest["outliers"] == f"outliers.{filetype}"
    assert os.path.exists(dataset_dir / f"outliers.{filetype}")
    validate_data(dataset_dir / f"outliers.{filetype}", [0, 0, 0, 1])


# ///////////////////////// PARSING TESTS /////////////////////////


def test_handles_simple_csv(tmp_path):
    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    convert_colorizer_data(csv_data, tmp_path / "dataset", use_json=True)
    validate_default_dataset(tmp_path / "dataset")


def test_handles_renamed_columns(tmp_path):
    csv_content = f"{sample_csv_headers_alternate}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    convert_colorizer_data(
        csv_data,
        tmp_path / "dataset",
        object_id_column="object_id",
        track_column="track",
        times_column="frame",
        centroid_x_column="centroid_x",
        centroid_y_column="centroid_y",
        image_column="file_path",
        outlier_column="outlier",
        use_json=True,
    )
    validate_default_dataset(
        tmp_path / "dataset",
    )


def test_handles_default_csv_parquet(tmp_path):
    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    convert_colorizer_data(csv_data, tmp_path / "dataset", use_json=False)
    validate_default_dataset(tmp_path / "dataset", "parquet")


def test_fails_if_no_features_given(tmp_path):
    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    # Delete all the columns that have feature data
    csv_data = csv_data.drop(
        ["Continuous Feature", "Discrete Feature", "Categorical Feature"], axis=1
    )
    with pytest.raises(Exception):
        convert_colorizer_data(csv_data, tmp_path)


def test_fails_if_no_objects_exist(tmp_path):
    csv_content = f"{sample_csv_headers}"
    csv_data = pd.read_csv(StringIO(csv_content))

    with pytest.raises(Exception):
        convert_colorizer_data(csv_data, tmp_path)


def test_handles_missing_centroid_and_outlier_columns(tmp_path):
    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    csv_data = csv_data.drop(["Centroid X", "Centroid Y", "Outlier"], axis=1)
    convert_colorizer_data(csv_data, tmp_path, use_json=True)

    # Outliers and centroids should not be written
    assert not os.path.exists(tmp_path / "outliers.json")
    assert not os.path.exists(tmp_path / "centroids.json")
    # Data should not be in manifest
    manifest = {}
    with open(tmp_path / "manifest.json", "r") as f:
        manifest = json.load(f)
        assert "outliers" not in manifest
        assert "centroids" not in manifest


def test_throws_error_if_all_values_are_outliers(tmp_path):
    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    csv_data["Outlier"] = 1
    with pytest.raises(Exception):
        convert_colorizer_data(csv_data, tmp_path)


def test_uses_id_as_track_if_track_is_missing(tmp_path):
    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    csv_data = csv_data.drop(["Track"], axis=1)
    convert_colorizer_data(csv_data, tmp_path, use_json=True)

    manifest = {}
    with open(tmp_path / "manifest.json", "r") as f:
        manifest = json.load(f)
        print(manifest)
        assert manifest["tracks"] == "tracks.json"
        # Uses IDs as track IDs
        validate_data(tmp_path / "tracks.json", [0, 1, 2, 3])


"""
TODO: Test additional edge cases
- [x] Frame generation
  - [x] override switch works
  - [x] detects change in number of objects
  - [x] detects removal of frames
  - [x] does not regenerate frames if they already exist
  - [x] handles missing times data
- [x] Handles different data column names
- [ ] Handles missing centroid, outliers, or bounds data
- [x] Keeps bounds data during frame regeneration
- [ ] Handles backdrop images via column
- [ ] Handles backdrop images via dictionary
"""

# ///////////////////////// FRAME GENERATION TESTS /////////////////////////


def test_does_not_rewrite_existing_frames_or_bounds_data(existing_dataset):
    frame_0_time = os.path.getmtime(existing_dataset / "frame_0.png")
    frame_1_time = os.path.getmtime(existing_dataset / "frame_1.png")
    bounds_time = os.path.getmtime(existing_dataset / "bounds.json")

    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    convert_colorizer_data(csv_data, existing_dataset, force_frame_generation=False)

    # Frames + bbox data should not be modified
    assert os.path.getmtime(existing_dataset / "frame_0.png") == frame_0_time
    assert os.path.getmtime(existing_dataset / "frame_1.png") == frame_1_time
    assert os.path.getmtime(existing_dataset / "bounds.json") == bounds_time

    # Reference to frames and bounds data should still be present in the manifest
    manifest = {}
    with open(existing_dataset / "manifest.json", "r") as f:
        manifest = json.load(f)
        assert manifest["bounds"] == "bounds.json"
        assert manifest["frames"] == ["frame_0.png", "frame_1.png"]


def test_detects_missing_frames(existing_dataset):
    frame_0_time = os.path.getmtime(existing_dataset / "frame_0.png")
    frame_1_time = os.path.getmtime(existing_dataset / "frame_1.png")
    # Delete one of the frames
    os.remove(existing_dataset / "frame_0.png")

    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    convert_colorizer_data(csv_data, existing_dataset, force_frame_generation=False)

    # Frames should be regenerated
    assert os.path.exists(existing_dataset / "frame_0.png")
    assert os.path.exists(existing_dataset / "frame_1.png")

    # Frame 0 and 1 should both have newer write times
    assert os.path.getmtime(existing_dataset / "frame_0.png") > frame_0_time
    assert os.path.getmtime(existing_dataset / "frame_1.png") > frame_1_time


def test_force_image_generation_flag_works(existing_dataset):
    # Record write time of both frames
    frame_0_time = os.path.getmtime(existing_dataset / "frame_0.png")
    frame_1_time = os.path.getmtime(existing_dataset / "frame_1.png")

    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    convert_colorizer_data(csv_data, existing_dataset, force_frame_generation=True)

    # Frame 0 and 1 should both have newer write times
    assert os.path.getmtime(existing_dataset / "frame_0.png") > frame_0_time
    assert os.path.getmtime(existing_dataset / "frame_1.png") > frame_1_time


def test_rewrites_images_when_object_count_changes(existing_dataset):
    # Record write time of both frames
    frame_0_time = os.path.getmtime(existing_dataset / "frame_0.png")
    frame_1_time = os.path.getmtime(existing_dataset / "frame_1.png")

    csv_content = f"{sample_csv_headers}\n{sample_csv_data}\n4,3,1,70,80,0.9,4,D,1,./colorizer_data/tests/assets/test_csv/frame_1.tiff"
    csv_data = pd.read_csv(StringIO(csv_content))
    convert_colorizer_data(csv_data, existing_dataset, force_frame_generation=False)

    # Frame 0 and 1 should both have newer write times
    assert os.path.getmtime(existing_dataset / "frame_0.png") > frame_0_time
    assert os.path.getmtime(existing_dataset / "frame_1.png") > frame_1_time


def test_regenerates_frames_if_missing_times_file(existing_dataset):
    # Record write time of both frames
    frame_0_time = os.path.getmtime(existing_dataset / "frame_0.png")
    frame_1_time = os.path.getmtime(existing_dataset / "frame_1.png")

    # Delete times file.
    os.remove(existing_dataset / "times.json")

    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    convert_colorizer_data(csv_data, existing_dataset, force_frame_generation=False)

    # Frame 0 and 1 should both have newer write times
    assert os.path.getmtime(existing_dataset / "frame_0.png") > frame_0_time
    assert os.path.getmtime(existing_dataset / "frame_1.png") > frame_1_time
