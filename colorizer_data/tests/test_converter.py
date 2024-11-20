from io import StringIO
import json
import pathlib
from colorizer_data import convert_colorizer_data
from colorizer_data.types import FeatureMetadata
import os
import pandas as pd
from typing import Dict, List, Union


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


sample_csv_headers = "ID,Track,Frame,Centroid X,Centroid Y,Continuous Feature,Discrete Feature,Categorical Feature,File Path"
sample_csv_headers_alternate = "object_id,track,frame,centroid_x,centroid_y,continuous_feature,discrete_feature,categorical_feature,file_path"
sample_csv_data = """0,1,0,50,50,0.5,0,A,./colorizer_data/tests/assets/basic_csv/frame_0.tiff
    1,1,1,55,60,0.6,1,B,./colorizer_data/tests/assets/basic_csv/frame_1.tiff
    2,2,0,60,70,0.7,2,C,./colorizer_data/tests/assets/basic_csv/frame_0.tiff
    3,2,1,65,75,0.8,3,A,./colorizer_data/tests/assets/basic_csv/frame_1.tiff"""


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


# def test_handles_documentation_csv(tmp_path):
#     csv_path = "colorizer_data/tests/assets/basic_csv/data.csv"
#     data = pd.read_csv(csv_path)
#     convert_colorizer_data(
#         data,
#         tmp_path / "dataset",
#     )

#     # Validate resulting dataset.
#     expected_manifest = tmp_path / "dataset" / "manifest.json"
#     assert os.path.exists(expected_manifest)

#     # Check for existence of all data files

#     # Check for existence of frames

"""
TODO: Test additional edge cases
- Frame generation
  - override switch works
  - detects change in number of objects
  - detects removal of frames
  - does not regenerate frames if they already exist
- Handles missing centroid, outliers, or bounds data
- Handles backdrop images via column
- Handles backdrop images via dictionary
"""


def test_handles_renamed_columns(tmp_path):
    pass


def test_detects_missing_frames(tmp_path):
    pass


def test_handles_missing_data_columns(tmp_path):
    pass


def test_handles_missing_frames(tmp_path):
    pass
