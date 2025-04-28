from io import StringIO
import json
import os
import pathlib
import shutil

import pandas as pd
import pytest
from typing import Dict, List, Union

from colorizer_data import convert_colorizer_data
from colorizer_data.types import DataFileType, FeatureMetadata
from colorizer_data.utils import read_data_array_file

asset_path = pathlib.Path(__file__).parent / "assets"


sample_csv_headers = "ID,Track,Frame,Centroid X,Centroid Y,Continuous Feature,Discrete Feature,Categorical Feature,Outlier,File Path"
sample_csv_headers_alternate = "object_id,track,frame,centroid_x,centroid_y,Continuous Feature,Discrete Feature,Categorical Feature,outlier,file_path"

raw_sample_csv_data = [
    f"0,1,0,50,50,0.5,0,A,0,{str(asset_path)}/test_csv/frame_0.tiff",
    f"1,1,1,55,60,0.6,1,B,0,{str(asset_path)}/test_csv/frame_1.tiff",
    f"2,2,0,60,70,0.7,2,C,0,{str(asset_path)}/test_csv/frame_0.tiff",
    f"3,2,1,65,75,0.8,3,A,1,{str(asset_path)}/test_csv/frame_1.tiff",
]
sample_csv_data = "\n".join(raw_sample_csv_data)

sample_csv_data_relative_paths = "\n".join(
    [
        "0,1,0,50,50,0.5,0,A,0,test_csv/frame_0.tiff",
        "1,1,1,55,60,0.6,1,B,0,test_csv/frame_1.tiff",
        "2,2,0,60,70,0.7,2,C,0,test_csv/frame_0.tiff",
        "3,2,1,65,75,0.8,3,A,1,test_csv/frame_1.tiff",
    ]
)

# ///////////////////////// METHODS /////////////////////////


# Computes once per test session.
@pytest.fixture(scope="session")
def existing_dataset(tmp_path_factory) -> pathlib.Path:
    tmp_path = tmp_path_factory.mktemp("dataset")
    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    convert_colorizer_data(
        csv_data, tmp_path, output_format=DataFileType.JSON, image_column="File Path"
    )
    return tmp_path


# ///////////////////////// METHODS /////////////////////////


@pytest.fixture
def existing_dataset(tmp_path) -> pathlib.Path:
    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    # TODO: Should I just write the relevant data files out without going through the image
    # processing step? Multiprocessing seems to make this very slow.
    convert_colorizer_data(
        csv_data, tmp_path, output_format=DataFileType.JSON, image_column="File Path"
    )
    return tmp_path


def feature_array_to_dict(
    feature_array: List[FeatureMetadata],
) -> Dict[str, FeatureMetadata]:
    return {feature.name: feature for feature in feature_array}


def validate_data(path: pathlib.Path, data: Union[List[int], List[float]]):
    loaded_data = read_data_array_file(path)

    assert len(loaded_data) == len(data)
    for i, d in enumerate(data):
        assert abs(loaded_data[i] - d) < 1e-6


def validate_default_dataset(
    dataset_dir: pathlib.Path, filetype: DataFileType = DataFileType.JSON
):
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
            "data": f"feature_0.{filetype.value}",
        },
        {
            "name": "Discrete Feature",
            "key": "discrete_feature",
            "type": "discrete",
            "description": "",
            "unit": "",
            "min": 0,
            "max": 2,
            "data": f"feature_1.{filetype.value}",
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
            "data": f"feature_2.{filetype.value}",
        },
    ]
    assert os.path.exists(dataset_dir / f"feature_0.{filetype.value}")
    assert os.path.exists(dataset_dir / f"feature_1.{filetype.value}")
    assert os.path.exists(dataset_dir / f"feature_2.{filetype.value}")
    validate_data(dataset_dir / f"feature_0.{filetype.value}", [0.5, 0.6, 0.7, 0.8])
    validate_data(dataset_dir / f"feature_1.{filetype.value}", [0, 1, 2, 3])
    validate_data(dataset_dir / f"feature_2.{filetype.value}", [0, 1, 2, 0])

    assert manifest["frames"] == ["frame_0.png", "frame_1.png"]
    assert os.path.exists(dataset_dir / "frame_0.png")
    assert os.path.exists(dataset_dir / "frame_1.png")

    assert manifest["tracks"] == f"tracks.{filetype.value}"
    assert os.path.exists(dataset_dir / f"tracks.{filetype.value}")
    validate_data(dataset_dir / f"tracks.{filetype.value}", [1, 1, 2, 2])

    assert manifest["centroids"] == f"centroids.{filetype.value}"
    assert os.path.exists(dataset_dir / f"centroids.{filetype.value}")
    validate_data(
        dataset_dir / f"centroids.{filetype.value}", [50, 50, 55, 60, 60, 70, 65, 75]
    )

    assert manifest["times"] == f"times.{filetype.value}"
    assert os.path.exists(dataset_dir / f"times.{filetype.value}")
    validate_data(dataset_dir / f"times.{filetype.value}", [0, 1, 0, 1])

    assert manifest["outliers"] == f"outliers.{filetype.value}"
    assert os.path.exists(dataset_dir / f"outliers.{filetype.value}")
    validate_data(dataset_dir / f"outliers.{filetype.value}", [0, 0, 0, 1])


# ///////////////////////// PARSING TESTS /////////////////////////


def test_handles_simple_csv(tmp_path):
    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    convert_colorizer_data(
        csv_data, tmp_path / "dataset", output_format=DataFileType.JSON
    )
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
        output_format=DataFileType.JSON,
    )
    validate_default_dataset(
        tmp_path / "dataset",
    )


def test_handles_default_csv_parquet(tmp_path):
    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    convert_colorizer_data(
        csv_data,
        tmp_path / "dataset",
        output_format=DataFileType.PARQUET,
        image_column="File Path",
    )
    validate_default_dataset(tmp_path / "dataset", DataFileType.PARQUET)


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
    convert_colorizer_data(csv_data, tmp_path, output_format=DataFileType.JSON)

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


def test_throws_error_if_times_column_is_missing(tmp_path):
    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    csv_data = csv_data.drop(["Frame"], axis=1)
    with pytest.raises(Exception):
        convert_colorizer_data(csv_data, tmp_path)


def test_uses_id_as_track_if_track_is_missing(tmp_path):
    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    csv_data = csv_data.drop(["Track"], axis=1)
    convert_colorizer_data(csv_data, tmp_path, output_format=DataFileType.JSON)

    manifest = {}
    with open(tmp_path / "manifest.json", "r") as f:
        manifest = json.load(f)
        print(manifest)
        assert manifest["tracks"] == "tracks.json"
        # Uses IDs as track IDs
        validate_data(tmp_path / "tracks.json", [0, 1, 2, 3])


def test_uses_source_dir_to_evaluate_relative_paths(tmp_path):
    # Use CSV with relative paths
    csv_content = f"{sample_csv_headers}\n{sample_csv_data_relative_paths}"
    data = pd.read_csv(StringIO(csv_content))

    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    # Check that conversion fails if no source directory is provided
    with pytest.raises(Exception):
        convert_colorizer_data(
            data,
            tmp_path,
        )

    # Current working directory is tmp_path, but all the assets are in
    # the asset_path directory.
    convert_colorizer_data(
        data,
        tmp_path,
        source_dir=asset_path,
        image_column="File Path",
    )
    validate_default_dataset(tmp_path, DataFileType.PARQUET)
    os.chdir(original_cwd)


def test_uses_default_segmentation_ids(tmp_path):
    data = pd.read_csv(StringIO(f"{sample_csv_headers}\n{sample_csv_data}"))
    convert_colorizer_data(
        data,
        tmp_path,
        output_format=DataFileType.JSON,
        object_id_column=None,
    )
    # Check that segmentation IDs are still written to the manifest
    # and the dataset directory.
    manifest = {}
    with open(tmp_path / "manifest.json", "r") as f:
        manifest = json.load(f)
        assert manifest["segIds"] == "seg_ids.json"
        assert os.path.exists(tmp_path / "seg_ids.json")
        validate_data(tmp_path / "seg_ids.json", [1, 2, 3, 4])


# ///////////////////////// FRAME GENERATION TESTS /////////////////////////


def test_does_not_rewrite_existing_frames(existing_dataset):
    frame_0_time = os.path.getmtime(existing_dataset / "frame_0.png")
    frame_1_time = os.path.getmtime(existing_dataset / "frame_1.png")
    # bounds_time = os.path.getmtime(existing_dataset / "bounds.json")

    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    convert_colorizer_data(csv_data, existing_dataset, force_frame_generation=False)

    # Frames + bbox data should not be modified
    assert os.path.getmtime(existing_dataset / "frame_0.png") == frame_0_time
    assert os.path.getmtime(existing_dataset / "frame_1.png") == frame_1_time
    # assert os.path.getmtime(existing_dataset / "bounds.json") == bounds_time

    # Reference to frames and bounds data should still be present in the manifest
    manifest = {}
    with open(existing_dataset / "manifest.json", "r") as f:
        manifest = json.load(f)
        assert manifest["frames"] == ["frame_0.png", "frame_1.png"]
        # assert manifest["bounds"] == "bounds.json"


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


def test_skips_frame_generation_if_no_image_column(existing_dataset):
    # Record write time of both frames
    frame_0_time = os.path.getmtime(existing_dataset / "frame_0.png")
    frame_1_time = os.path.getmtime(existing_dataset / "frame_1.png")

    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))
    convert_colorizer_data(
        csv_data, existing_dataset, force_frame_generation=False, image_column=None
    )

    # Frame 0 and 1 should not have newer write times
    assert os.path.getmtime(existing_dataset / "frame_0.png") == frame_0_time
    assert os.path.getmtime(existing_dataset / "frame_1.png") == frame_1_time


# ///////////////////////// 3D FRAME TESTS /////////////////////////


def test_writes_3d_data(tmp_path):
    csv_content = f"{sample_csv_headers}\n{sample_csv_data}"
    csv_data = pd.read_csv(StringIO(csv_content))

    convert_colorizer_data(
        csv_data,
        tmp_path,
        frames_3d_src="https://example.com/3d.ome.zarr",
        frames_3d_seg_channel=1,
    )

    with open(tmp_path / "manifest.json", "r") as f:
        manifest = json.load(f)
        assert manifest["frames3d"] == {
            "source": "https://example.com/3d.ome.zarr",
            "segmentationChannel": 1,
            # Total frames derived from times array if data source does not exist
            "totalFrames": 2,
        }


# ///////////////////////// BACKDROP TESTS /////////////////////////


class TestBackdropWriting:
    backdrop_headers = sample_csv_headers + ",Backdrop Image 1,Backdrop Image 2"
    raw_backdrop_csv_data = [
        raw_sample_csv_data[0]
        + f",{str(asset_path)}/backdrop-light/image_0.png,{str(asset_path)}/backdrop-dark/image_0.png",
        raw_sample_csv_data[1]
        + f",{str(asset_path)}/backdrop-light/image_1.png,{str(asset_path)}/backdrop-dark/image_1.png",
        raw_sample_csv_data[2]
        + f",{str(asset_path)}/backdrop-light/image_0.png,{str(asset_path)}/backdrop-dark/image_0.png",
        raw_sample_csv_data[3]
        + f",{str(asset_path)}/backdrop-light/image_1.png,{str(asset_path)}/backdrop-dark/image_1.png",
    ]
    backdrop_csv_data = "\n".join(raw_backdrop_csv_data)

    def test_writes_backdrop_images_using_column_names(self, tmp_path):
        csv_data = pd.read_csv(
            StringIO(self.backdrop_headers + "\n" + self.backdrop_csv_data)
        )

        convert_colorizer_data(
            csv_data,
            tmp_path,
            backdrop_column_names=["Backdrop Image 1", "Backdrop Image 2"],
            output_format=DataFileType.JSON,
        )

        assert os.path.exists(tmp_path / "backdrop_image_1" / "image_0.png")
        assert os.path.exists(tmp_path / "backdrop_image_1" / "image_1.png")
        assert os.path.exists(tmp_path / "backdrop_image_2" / "image_0.png")
        assert os.path.exists(tmp_path / "backdrop_image_2" / "image_1.png")

        # Load manifest, check that paths exist and are relative
        manifest = {}
        with open(tmp_path / "manifest.json", "r") as f:
            manifest = json.load(f)
        assert manifest["backdrops"] == [
            {
                "name": "Backdrop Image 1",
                "key": "backdrop_image_1",
                "frames": [
                    "backdrop_image_1/image_0.png",
                    "backdrop_image_1/image_1.png",
                ],
            },
            {
                "name": "Backdrop Image 2",
                "key": "backdrop_image_2",
                "frames": [
                    "backdrop_image_2/image_0.png",
                    "backdrop_image_2/image_1.png",
                ],
            },
        ]
        # Features should not have changed.
        validate_default_dataset(tmp_path, DataFileType.JSON)

    def test_does_not_rewrite_existing_backdrop_images(self, tmp_path):
        # Copy backdrop images to dataset directory so they already exist.
        # Because they're local, the images should not be copied or modified.
        # The paths should still be modified to be relative to the manifest, though.
        shutil.copytree(asset_path / "backdrop-light", tmp_path / "backdrop-light")
        shutil.copytree(asset_path / "backdrop-dark", tmp_path / "backdrop-dark")
        write_time = os.path.getmtime(tmp_path / "backdrop-light" / "image_0.png")

        local_raw_backdrop_csv_data = [
            raw_sample_csv_data[0]
            + f",{tmp_path}/backdrop-light/image_0.png,{tmp_path}/backdrop-dark/image_0.png",
            raw_sample_csv_data[1]
            + f",{tmp_path}/backdrop-light/image_1.png,{tmp_path}/backdrop-dark/image_1.png",
            raw_sample_csv_data[2]
            + f",{tmp_path}/backdrop-light/image_0.png,{tmp_path}/backdrop-dark/image_0.png",
            raw_sample_csv_data[3]
            + f",{tmp_path}/backdrop-light/image_1.png,{tmp_path}/backdrop-dark/image_1.png",
        ]
        local_backdrop_csv_data = "\n".join(local_raw_backdrop_csv_data)

        csv_data = pd.read_csv(
            StringIO(self.backdrop_headers + "\n" + local_backdrop_csv_data)
        )
        convert_colorizer_data(
            csv_data,
            tmp_path,
            backdrop_column_names=["Backdrop Image 1", "Backdrop Image 2"],
            output_format=DataFileType.JSON,
        )

        # Should not rename or rewrite images
        assert not os.path.exists(tmp_path / "backdrop_image_1" / "image_0.png")
        assert not os.path.exists(tmp_path / "backdrop_image_1" / "image_1.png")
        assert not os.path.exists(tmp_path / "backdrop_image_2" / "image_0.png")
        assert not os.path.exists(tmp_path / "backdrop_image_2" / "image_1.png")

        assert os.path.exists(tmp_path / "backdrop-light" / "image_0.png")
        assert os.path.exists(tmp_path / "backdrop-light" / "image_1.png")
        assert os.path.exists(tmp_path / "backdrop-dark" / "image_0.png")
        assert os.path.exists(tmp_path / "backdrop-dark" / "image_1.png")

        # Images should not be modified
        assert (
            os.path.getmtime(tmp_path / "backdrop-light" / "image_0.png") == write_time
        )

        # Load manifest, check that paths exist and are relative
        manifest = {}
        with open(tmp_path / "manifest.json", "r") as f:
            manifest = json.load(f)
        assert manifest["backdrops"] == [
            {
                "name": "Backdrop Image 1",
                "key": "backdrop_image_1",
                "frames": [
                    "backdrop-light/image_0.png",
                    "backdrop-light/image_1.png",
                ],
            },
            {
                "name": "Backdrop Image 2",
                "key": "backdrop_image_2",
                "frames": [
                    "backdrop-dark/image_0.png",
                    "backdrop-dark/image_1.png",
                ],
            },
        ]

    def test_override_some_backdrop_metadata(self, tmp_path):
        csv_data = pd.read_csv(
            StringIO(self.backdrop_headers + "\n" + self.backdrop_csv_data)
        )
        backdrop_info = {
            "Backdrop Image 1": {
                "name": "New Backdrop Name",
                "key": "new_backdrop_key",
            }
        }
        convert_colorizer_data(
            csv_data,
            tmp_path,
            backdrop_column_names=["Backdrop Image 1", "Backdrop Image 2"],
            backdrop_info=backdrop_info,
            output_format=DataFileType.JSON,
        )

        assert os.path.exists(tmp_path / "new_backdrop_key" / "image_0.png")
        assert os.path.exists(tmp_path / "new_backdrop_key" / "image_1.png")
        assert os.path.exists(tmp_path / "backdrop_image_2" / "image_0.png")
        assert os.path.exists(tmp_path / "backdrop_image_2" / "image_1.png")

        # Load manifest, check that paths exist and are relative
        manifest = {}
        with open(tmp_path / "manifest.json", "r") as f:
            manifest = json.load(f)
        assert manifest["backdrops"] == [
            {
                "name": "New Backdrop Name",
                "key": "new_backdrop_key",
                "frames": [
                    "new_backdrop_key/image_0.png",
                    "new_backdrop_key/image_1.png",
                ],
            },
            {
                "name": "Backdrop Image 2",
                "key": "backdrop_image_2",
                "frames": [
                    "backdrop_image_2/image_0.png",
                    "backdrop_image_2/image_1.png",
                ],
            },
        ]

    def test_override_backdrop_paths(self, tmp_path):
        csv_data = pd.read_csv(
            StringIO(self.backdrop_headers + "\n" + self.backdrop_csv_data)
        )
        backdrop_info = {
            "Backdrop Image 2": {
                "frames": [
                    f"{asset_path}/backdrop-light/image_0.png",
                    f"{asset_path}/backdrop-light/image_1.png",
                ]
            }
        }
        convert_colorizer_data(
            csv_data,
            tmp_path,
            backdrop_column_names=["Backdrop Image 1", "Backdrop Image 2"],
            backdrop_info=backdrop_info,
            output_format=DataFileType.JSON,
        )

        assert os.path.exists(tmp_path / "backdrop_image_1" / "image_0.png")
        assert os.path.exists(tmp_path / "backdrop_image_1" / "image_1.png")
        assert os.path.exists(tmp_path / "backdrop_image_2" / "image_0.png")
        assert os.path.exists(tmp_path / "backdrop_image_2" / "image_1.png")

        # Files should be the same because they both pulled from the same source
        assert (
            open(tmp_path / "backdrop_image_1" / "image_0.png", "rb").read()
            == open(tmp_path / "backdrop_image_2" / "image_0.png", "rb").read()
        )
        assert (
            open(tmp_path / "backdrop_image_1" / "image_1.png", "rb").read()
            == open(tmp_path / "backdrop_image_2" / "image_1.png", "rb").read()
        )

    def test_writes_backdrops_that_are_not_in_column_names(self, tmp_path):
        # Copy the backdrop images to the dataset directory
        shutil.copytree(asset_path / "backdrop-light", tmp_path / "backdrop-light")

        csv_data = pd.read_csv(StringIO(sample_csv_headers + "\n" + sample_csv_data))
        backdrop_info_1 = {
            "name": "Backdrop 1",
            "key": "new_backdrop_1",
            "frames": [
                f"{asset_path}/backdrop-light/image_0.png",
                f"{asset_path}/backdrop-light/image_1.png",
            ],
        }
        backdrop_info_2 = {
            "name": "Backdrop 2",
            "key": "new_backdrop_2",
            "frames": [
                f"{asset_path}/backdrop-dark/image_0.png",
                f"{asset_path}/backdrop-dark/image_1.png",
            ],
        }
        backdrop_info = {
            # Keys are arbitrary here
            "A backdrop": backdrop_info_1,
            "Some other backdrop": backdrop_info_2,
        }
        convert_colorizer_data(
            csv_data,
            tmp_path,
            backdrop_info=backdrop_info,
            output_format=DataFileType.JSON,
        )

        # File paths should be copied
        assert os.path.exists(tmp_path / "new_backdrop_1" / "image_0.png")
        assert os.path.exists(tmp_path / "new_backdrop_1" / "image_1.png")
        assert os.path.exists(tmp_path / "new_backdrop_2" / "image_0.png")
        assert os.path.exists(tmp_path / "new_backdrop_2" / "image_1.png")

        manifest = {}
        with open(tmp_path / "manifest.json", "r") as f:
            manifest = json.load(f)
        # Paths should be relative
        assert manifest["backdrops"] == [
            {
                "name": "Backdrop 1",
                "key": "new_backdrop_1",
                "frames": [
                    "new_backdrop_1/image_0.png",
                    "new_backdrop_1/image_1.png",
                ],
            },
            {
                "name": "Backdrop 2",
                "key": "new_backdrop_2",
                "frames": [
                    "new_backdrop_2/image_0.png",
                    "new_backdrop_2/image_1.png",
                ],
            },
        ]

    def test_throws_error_if_backdrop_does_not_exist(self, tmp_path):
        csv_data = pd.read_csv(StringIO(sample_csv_headers + "\n" + sample_csv_data))
        backdrop_info = {
            "backdrop": {
                "name": "Backdrop 1",
                "key": "new_backdrop_1",
                "frames": [
                    "path/does/not/exist/image_0.png",
                    "path/does/not/exist/image_1.png",
                ],
            }
        }
        with pytest.raises(Exception):
            convert_colorizer_data(
                csv_data,
                tmp_path,
                backdrop_info=backdrop_info,
            )

    def test_sanitizes_backdrop_key_names(self, tmp_path):
        csv_data = pd.read_csv(StringIO(sample_csv_headers + "\n" + sample_csv_data))
        backdrop_info = {
            "backdrop": {
                "name": "Backdrop",
                "key": "!!!BAD KEY NAME!!!",
                "frames": [
                    f"{asset_path}/backdrop-light/image_0.png",
                    f"{asset_path}/backdrop-light/image_1.png",
                ],
            }
        }
        convert_colorizer_data(
            csv_data,
            tmp_path,
            backdrop_info=backdrop_info,
        )
        manifest = {}
        with open(tmp_path / "manifest.json", "r") as f:
            manifest = json.load(f)
        assert manifest["backdrops"] == [
            {
                "name": "Backdrop",
                "key": "bad_key_name",
                "frames": [
                    "bad_key_name/image_0.png",
                    "bad_key_name/image_1.png",
                ],
            },
        ]

    def test_handles_none_backdrop(self, tmp_path):
        csv_data = pd.read_csv(StringIO(sample_csv_headers + "\n" + sample_csv_data))
        convert_colorizer_data(
            csv_data,
            tmp_path,
            backdrop_column_names=["Nonexistent Backdrop"],
            output_format=DataFileType.JSON,
        )
        manifest = {}
        with open(tmp_path / "manifest.json", "r") as f:
            manifest = json.load(f)

        assert manifest["backdrops"] == [
            {
                "name": "Nonexistent Backdrop",
                "key": "nonexistent_backdrop",
                "frames": [None, None],
            }
        ]
