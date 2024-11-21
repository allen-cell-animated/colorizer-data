from io import StringIO
import json
import pathlib

import pytest
from colorizer_data import convert_colorizer_data
from colorizer_data.types import FeatureMetadata
import os
import pandas as pd
from typing import Dict, List, Union

from colorizer_data.utils import configureLogging

# TODO: Make CSV tiffs super tiny to speed up image processing in tests
sample_csv_headers = "ID,Track,Frame,Centroid X,Centroid Y,Continuous Feature,Discrete Feature,Categorical Feature,File Path"
sample_csv_headers_alternate = "object_id,track,frame,centroid_x,centroid_y,continuous_feature,discrete_feature,categorical_feature,file_path"
sample_csv_data = """0,1,0,50,50,0.5,0,A,./colorizer_data/tests/assets/test_csv/frame_0.tiff
    1,1,1,55,60,0.6,1,B,./colorizer_data/tests/assets/test_csv/frame_1.tiff
    2,2,0,60,70,0.7,2,C,./colorizer_data/tests/assets/test_csv/frame_0.tiff
    3,2,1,65,75,0.8,3,A,./colorizer_data/tests/assets/test_csv/frame_1.tiff"""


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
    with open(path, "r") as f:
        if path.suffix == ".json":
            feature_data = json.load(f)
            loaded_data = feature_data["data"]
        else:
            feature_data = pd.read_parquet(f)
            loaded_data = feature_data["data"]

    assert len(loaded_data) == len(data)
    for i, d in enumerate(data):
        assert abs(loaded_data[i] - d) < 1e-6


def test_handles_simple_csv(tmp_path):
    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    convert_colorizer_data(csv_data, tmp_path / "dataset", use_json=True)

    expected_manifest = tmp_path / "dataset" / "manifest.json"
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
            "max": 0.8,
            "data": "feature_0.json",
        },
        {
            "name": "Discrete Feature",
            "key": "discrete_feature",
            "type": "discrete",
            "description": "",
            "unit": "",
            "min": 0,
            "max": 3,
            "data": "feature_1.json",
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
            "data": "feature_2.json",
        },
    ]
    assert os.path.exists(tmp_path / "dataset" / "feature_0.json")
    assert os.path.exists(tmp_path / "dataset" / "feature_1.json")
    assert os.path.exists(tmp_path / "dataset" / "feature_2.json")
    validate_data(tmp_path / "dataset" / "feature_0.json", [0.5, 0.6, 0.7, 0.8])
    validate_data(tmp_path / "dataset" / "feature_1.json", [0, 1, 2, 3])
    validate_data(tmp_path / "dataset" / "feature_2.json", [0, 1, 2, 0])

    assert manifest["frames"] == ["frame_0.png", "frame_1.png"]
    assert os.path.exists(tmp_path / "dataset" / "frame_0.png")
    assert os.path.exists(tmp_path / "dataset" / "frame_1.png")

    assert manifest["tracks"] == "tracks.json"
    assert os.path.exists(tmp_path / "dataset" / "tracks.json")
    validate_data(tmp_path / "dataset" / "tracks.json", [1, 1, 2, 2])

    assert manifest["centroids"] == "centroids.json"
    assert os.path.exists(tmp_path / "dataset" / "centroids.json")
    validate_data(
        tmp_path / "dataset" / "centroids.json", [50, 50, 55, 60, 60, 70, 65, 75]
    )

    assert manifest["times"] == "times.json"
    assert os.path.exists(tmp_path / "dataset" / "times.json")
    validate_data(tmp_path / "dataset" / "times.json", [0, 1, 0, 1])


"""
TODO: Test additional edge cases
- [x] Frame generation
  - [x] override switch works
  - [x] detects change in number of objects
  - [x] detects removal of frames
  - [x] does not regenerate frames if they already exist
- [ ] Handles missing centroid, outliers, or bounds data
- [ ] Keeps bounds data during frame regeneration
- [ ] Handles backdrop images via column
- [ ] Handles backdrop images via dictionary
"""


def test_handles_renamed_columns(tmp_path):
    pass


def test_does_not_rewrite_existing_frames(existing_dataset):
    # Record write time of both frames
    frame_0_time = os.path.getmtime(existing_dataset / "frame_0.png")
    frame_1_time = os.path.getmtime(existing_dataset / "frame_1.png")

    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    convert_colorizer_data(csv_data, existing_dataset, force_frame_generation=False)

    # Frame 0 and 1 should both have newer write times
    assert os.path.getmtime(existing_dataset / "frame_0.png") == frame_0_time
    assert os.path.getmtime(existing_dataset / "frame_1.png") == frame_1_time


def test_detects_missing_frames(existing_dataset):
    # Record write time of both frames
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


def test_forces_image_overwrite(existing_dataset):
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

    csv_content = f"{sample_csv_headers}\n{sample_csv_data}\n4,3,1,70,80,0.9,4,D,./colorizer_data/tests/assets/test_csv/frame_1.tiff"
    csv_data = pd.read_csv(StringIO(csv_content))
    convert_colorizer_data(csv_data, existing_dataset, force_frame_generation=False)

    # Frame 0 and 1 should both have newer write times
    assert os.path.getmtime(existing_dataset / "frame_0.png") > frame_0_time
    assert os.path.getmtime(existing_dataset / "frame_1.png") > frame_1_time


def test_handles_missing_data_columns(tmp_path):
    pass
