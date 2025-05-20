import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest
import pyarrow.parquet as pq

from colorizer_data.types import (
    CURRENT_VERSION,
    ColorizerMetadata,
    DatasetManifest,
    FeatureInfo,
)
from colorizer_data.writer import ColorizerDatasetWriter

DEFAULT_DATASET_NAME = "dataset"

EXISTING_MANIFEST_CONTENT: DatasetManifest = {
    "features": [],
    "frames": [],
    "metadata": {
        "name": "my example dataset",
        "description": "description of my example dataset",
        "author": "john",
        "datasetVersion": "old version",
        "dateCreated": "2000-01-01T01:00:00.000Z",
        "lastModified": "2000-01-01T02:00:00.000Z",
        "revision": 4,
        "writerVersion": "v0.4.0",
        "frameDims": {"width": 500, "height": 340, "units": "um"},
        "startTimeSeconds": 120,
        "frameDurationSeconds": 0.5,
        "startingFrameNumber": 12,
    },
}

BLANK_MANIFEST_CONTENT: DatasetManifest = {
    "features": [],
    "frames": [],
}


def setup_dummy_writer_data(writer: ColorizerDatasetWriter):
    writer.write_data(times=np.ndarray([0]), write_json=True)
    writer.set_frame_paths([""])


def setup_manifest_and_writer(path, content):
    directory = path / DEFAULT_DATASET_NAME
    directory.mkdir()
    manifest_path = path / DEFAULT_DATASET_NAME / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(content, f, indent=2)
    writer = ColorizerDatasetWriter(path, DEFAULT_DATASET_NAME)
    setup_dummy_writer_data(writer)
    return writer, path, manifest_path


@pytest.fixture
def existing_manifest(tmp_path) -> Tuple[ColorizerDatasetWriter, Path, Path]:
    return setup_manifest_and_writer(tmp_path, EXISTING_MANIFEST_CONTENT)


@pytest.fixture
def blank_manifest(tmp_path) -> Tuple[ColorizerDatasetWriter, Path, Path]:
    return setup_manifest_and_writer(tmp_path, BLANK_MANIFEST_CONTENT)


def test_metadata_uses_frame_dims_subfield(tmp_path):
    # Test writing manifest saves frame dimensions as `frameDims` field within
    # the metadata to match data API

    frame_dimensions: Tuple[Optional[float], Optional[float], Optional[str]] = [
        (80.0, 60.0, "picometers"),
        (None, 44.0, None),
        (None, None, None),
    ]

    for i in range(len(frame_dimensions)):
        width, height, units = frame_dimensions[i]
        writer = ColorizerDatasetWriter(tmp_path, DEFAULT_DATASET_NAME)
        setup_dummy_writer_data(writer)
        writer.write_manifest(
            metadata=ColorizerMetadata(
                frame_width=width,
                frame_height=height,
                frame_units=units,
            )
        )

        expected_manifest = tmp_path / DEFAULT_DATASET_NAME / "manifest.json"
        assert os.path.exists(expected_manifest)
        manifest: DatasetManifest = {}
        with open(expected_manifest, "r") as f:
            manifest = json.load(f)

        # Expect manifest structure to use `frameDims`
        assert manifest["metadata"]["frameDims"]["width"] == width
        assert manifest["metadata"]["frameDims"]["height"] == height
        assert manifest["metadata"]["frameDims"]["units"] == units

        # Expect ColorizerMetadata fields to not be written directly
        assert "frame_width" not in manifest["metadata"].keys()
        assert "frame_height" not in manifest["metadata"].keys()
        assert "frame_units" not in manifest["metadata"].keys()

        # Expect manifest structure can be parsed to ColorizerMetadata
        # with correct fields
        metadata = ColorizerMetadata.from_dict(manifest["metadata"])

        assert metadata.frame_width == width
        assert metadata.frame_height == height
        assert metadata.frame_units == units

        # Cleanup
        os.remove(expected_manifest)


def test_writer_updates_revision_and_time(existing_manifest):
    # Should update revision number, updated time, and data version

    writer, tmp_path, manifest_path = existing_manifest
    writer.write_manifest()

    with open(manifest_path, "r") as f:
        manifest: DatasetManifest = json.load(f)
        metadata = manifest["metadata"]
        oldMetadata = EXISTING_MANIFEST_CONTENT["metadata"]

        # Updates expected fields
        assert metadata["lastModified"] is not None
        assert metadata["lastModified"] != oldMetadata["lastModified"]
        assert metadata["revision"] == oldMetadata["revision"] + 1
        assert metadata["writerVersion"] == CURRENT_VERSION


def test_writer_handles_renamed_fields(existing_manifest):
    # startingTimeSeconds => metadata.start_time_sec
    # startingFrameNumber => metadata.start_frame_num
    # frameDurationSeconds => metadata.frame_duration_sec

    writer, tmp_path, manifest_path = existing_manifest
    writer.write_manifest(
        metadata=ColorizerMetadata(
            start_time_sec=1.0, start_frame_num=2.0, frame_duration_sec=3.0
        )
    )

    with open(manifest_path, "r") as f:
        manifest: DatasetManifest = json.load(f)
        metadata_dict = manifest["metadata"]
        metadata = ColorizerMetadata.from_dict(metadata_dict)

        # Expect fields to be written out in the dictionary
        assert metadata_dict["startingTimeSeconds"] == 1.0
        assert metadata_dict["startingFrameNumber"] == 2.0
        assert metadata_dict["frameDurationSeconds"] == 3.0

        # Expect fields to be loaded correctly by `from_dict`
        assert metadata.start_time_sec == 1.0
        assert metadata.start_frame_num == 2.0
        assert metadata.frame_duration_sec == 3.0


def test_writer_keeps_manifest_metadata(existing_manifest):
    # Should keep name, author, time of creation, description
    writer, tmp_path, manifest_path = existing_manifest
    writer.write_manifest(
        metadata=ColorizerMetadata(start_time_sec=5, start_frame_num=6)
    )

    with open(manifest_path, "r") as f:
        manifest: DatasetManifest = json.load(f)
        metadata = manifest["metadata"]
        oldMetadata = EXISTING_MANIFEST_CONTENT["metadata"]

        # Check that changes were made
        assert metadata["startingTimeSeconds"] == 5
        assert metadata["startingFrameNumber"] == 6

        # Leaves other fields untouched
        assert metadata["name"] == oldMetadata["name"]
        assert metadata["description"] == oldMetadata["description"]
        assert metadata["author"] == oldMetadata["author"]
        assert metadata["dateCreated"] == oldMetadata["dateCreated"]


def test_writer_overrides_metadata_fields(existing_manifest):
    # Should override author, name, description, date of creation,
    # time of last modification, dataVersion, and revision number if provided.

    writer, tmp_path, manifest_path = existing_manifest
    writer.write_manifest(
        metadata=ColorizerMetadata(
            name="new name",
            description="new description",
            author="geoff",
            date_created="some-date",
            last_modified="some-other-date",
            _revision=250,
            _writer_version="abcdef",
        )
    )

    with open(manifest_path, "r") as f:
        manifest: DatasetManifest = json.load(f)
        metadata = manifest["metadata"]

        # Overrides fields using new metadata
        assert metadata["name"] == "new name"
        assert metadata["description"] == "new description"
        assert metadata["author"] == "geoff"
        assert metadata["dateCreated"] == "some-date"

        # Overwrites default updaters
        assert metadata["lastModified"] == "some-other-date"
        assert metadata["revision"] == 250
        assert metadata["writerVersion"] == "abcdef"


def test_writer_updates_fields_when_metadata_is_missing(blank_manifest):
    # Update name, revision number, creation time, updated time, and data version if
    # base manifest does not include this information
    writer, tmp_path, manifest_path = blank_manifest
    writer.write_manifest()

    with open(manifest_path, "r") as f:
        manifest: DatasetManifest = json.load(f)
        metadata: ColorizerMetadata = ColorizerMetadata.from_dict(manifest["metadata"])

        assert metadata.name == DEFAULT_DATASET_NAME
        assert metadata.date_created is not None
        assert metadata.date_created == metadata.last_modified
        assert metadata._writer_version == CURRENT_VERSION
        assert metadata._revision == 0


def test_writer_overwrites_duplicate_feature_keys(tmp_path):
    writer = ColorizerDatasetWriter(tmp_path, DEFAULT_DATASET_NAME)
    setup_dummy_writer_data(writer)

    feature_1_info = FeatureInfo(key="shared_feature_key", label="Feature 1")
    writer.write_feature(np.array([0, 1, 2, 3]), feature_1_info)
    feature_2_info = FeatureInfo(key="shared_feature_key", label="Feature 2")
    writer.write_feature(np.array([0, 1, 2, 3]), feature_2_info)
    writer.write_manifest()

    with open(tmp_path / DEFAULT_DATASET_NAME / "manifest.json", "r") as f:
        manifest: DatasetManifest = json.load(f)

        assert len(manifest["features"]) == 1
        assert manifest["features"][0]["key"] == "shared_feature_key"
        assert manifest["features"][0]["name"] == "Feature 2"


def test_writer_overwrites_duplicate_backdrop_keys(tmp_path):
    writer = ColorizerDatasetWriter(tmp_path, DEFAULT_DATASET_NAME)
    setup_dummy_writer_data(writer)

    writer.add_backdrops("Backdrop 1", [], "shared_backdrop_key")
    writer.add_backdrops("Backdrop 2", [], "shared_backdrop_key")
    writer.write_manifest()

    with open(tmp_path / DEFAULT_DATASET_NAME / "manifest.json", "r") as f:
        manifest: DatasetManifest = json.load(f)

        assert len(manifest["backdrops"]) == 1
        assert manifest["backdrops"][0]["key"] == "shared_backdrop_key"
        assert manifest["backdrops"][0]["name"] == "Backdrop 2"


def test_writer_ignores_infinity_values_for_feature_min_max(tmp_path):
    # Test that infinity values are ignored when calculating min/max
    writer = ColorizerDatasetWriter(tmp_path, DEFAULT_DATASET_NAME)
    setup_dummy_writer_data(writer)

    feature_info = FeatureInfo(key="feature", label="Feature")
    writer.write_feature(
        np.array([0, 1, 2, np.inf, -np.inf, 4]),
        feature_info,
        write_json=True,
    )
    writer.write_manifest()

    with open(tmp_path / DEFAULT_DATASET_NAME / "manifest.json", "r") as f:
        manifest: DatasetManifest = json.load(f)
        feature_file = manifest["features"][0]["data"]
        with open(tmp_path / DEFAULT_DATASET_NAME / feature_file, "r") as f2:
            feature_data = json.load(f2)
            assert feature_data["min"] == 0
            assert feature_data["max"] == 4
            assert feature_data["data"] == [0, 1, 2, np.inf, -np.inf, 4]


class TestWriteFeature:
    def test_write_feature_ignores_outliers_when_calculating_feature_min_max(
        self, tmp_path
    ):
        writer = ColorizerDatasetWriter(tmp_path, DEFAULT_DATASET_NAME)
        setup_dummy_writer_data(writer)

        feature_info = FeatureInfo(key="feature", label="Feature")
        writer.write_feature(
            np.array([0, 1, 2, 3, 4]),
            feature_info,
            outliers=[True, False, False, True, True],
            write_json=True,
        )
        writer.write_manifest()

        with open(tmp_path / DEFAULT_DATASET_NAME / "manifest.json", "r") as f:
            manifest: DatasetManifest = json.load(f)
            feature_file = manifest["features"][0]["data"]
            with open(tmp_path / DEFAULT_DATASET_NAME / feature_file, "r") as f2:
                feature_data = json.load(f2)
                assert feature_data["min"] == 1
                assert feature_data["max"] == 2
                assert feature_data["data"] == [0, 1, 2, 3, 4]

    def test_write_feature_uses_overrides_when_calculating_feature_min_max(
        self, tmp_path
    ):
        writer = ColorizerDatasetWriter(tmp_path, DEFAULT_DATASET_NAME)
        setup_dummy_writer_data(writer)

        feature_info = FeatureInfo(key="feature", label="Feature", min=-5, max=3)
        writer.write_feature(np.array([0, 1, 2, 3, 4]), feature_info, write_json=True)
        writer.write_manifest()

        with open(tmp_path / DEFAULT_DATASET_NAME / "manifest.json", "r") as f:
            manifest: DatasetManifest = json.load(f)

            # Check manifest min + max
            feature_file = manifest["features"][0]["data"]
            with open(tmp_path / DEFAULT_DATASET_NAME / feature_file, "r") as f2:
                feature_data = json.load(f2)
                assert feature_data["min"] == -5
                assert feature_data["max"] == 3
                assert feature_data["data"] == [0, 1, 2, 3, 4]

    def test_write_feature_writes_parquet_data(self, tmp_path):
        writer = ColorizerDatasetWriter(tmp_path, DEFAULT_DATASET_NAME)
        setup_dummy_writer_data(writer)

        feature_info = FeatureInfo(key="feature", label="Feature")
        data = np.array([-5, 0, 2, 3, 4])
        writer.write_feature(data, feature_info)
        writer.write_manifest()

        with open(tmp_path / DEFAULT_DATASET_NAME / "manifest.json", "r") as f:
            manifest: DatasetManifest = json.load(f)
            feature_info = manifest["features"][0]
            # Min and max should be saved to the manifest
            assert feature_info["min"] == -5
            assert feature_info["max"] == 4

            # Data should be saved to a parquet file
            feature_file = feature_info["data"]
            assert feature_file.endswith(".parquet")
            assert os.path.exists(tmp_path / DEFAULT_DATASET_NAME / feature_file)

            # Check parquet data has expected contents
            table = pq.read_table(tmp_path / DEFAULT_DATASET_NAME / feature_file)
            assert table.to_pandas()["data"].tolist() == data.tolist()

    def write_features_can_write_nan_to_parquet(self, tmp_path):
        writer = ColorizerDatasetWriter(tmp_path, DEFAULT_DATASET_NAME)
        setup_dummy_writer_data(writer)

        feature_info = FeatureInfo(key="feature", label="Feature")
        data = np.array([np.nan, 1, 2, np.nan, 4])
        writer.write_feature(data, feature_info)
        with open(tmp_path / DEFAULT_DATASET_NAME / "manifest.json", "r") as f:
            manifest: DatasetManifest = json.load(f)
            feature_info = manifest["features"][0]
            assert feature_info["min"] == 1
            assert feature_info["max"] == 4

            # Data should be saved to a parquet file
            feature_file = feature_info["data"]
            # Check parquet data has expected contents
            table = pq.read_table(tmp_path / DEFAULT_DATASET_NAME / feature_file)
            file_data = table.to_pandas()["data"].tolist()
            assert np.isnan(file_data[0])
            assert file_data[1] == 1
            assert file_data[2] == 2
            assert np.isnan(file_data[3])
            assert file_data[4] == 4


class TestWriteData:
    centroids_x = np.array([-1, 1, 2, 3, 4])
    centroids_y = np.array([5, 6, 7, 8, 9])

    default_data = {
        "tracks": np.array([0, 1, 0, 1, 2]),
        "times": np.array([1, 1, 2, 2, 3]),
        "centroids_x": centroids_x,
        "centroids_y": centroids_y,
        "centroids": np.ravel(np.dstack([centroids_x, centroids_y])),
        "outliers": np.array([1, 0, 0, 1, 0]),
        "bounds": np.array(
            [0, 3, 4, 5, 1, 5, 6, 7, 24, 65, 87, 54, 38, 234, 12, 34, 34, 56, 132, 24],
        ),
    }

    def write_default_data(self, writer: ColorizerDatasetWriter, write_json: bool):
        writer.write_data(
            tracks=self.default_data["tracks"],
            times=self.default_data["times"],
            centroids_x=self.default_data["centroids_x"],
            centroids_y=self.default_data["centroids_y"],
            outliers=self.default_data["outliers"],
            bounds=self.default_data["bounds"],
            write_json=write_json,
        )
        writer.set_frame_paths([""])

    def validate_json(self, path: str, expected_data: np.ndarray):
        assert str(path).endswith(".json")
        with open(path, "r") as f:
            data = json.load(f)
            assert data["data"] == expected_data.tolist()

    def validate_parquet(self, path: str, expected_data: np.ndarray):
        assert str(path).endswith(".parquet")
        table = pq.read_table(path)
        assert table.to_pandas()["data"].tolist() == expected_data.tolist()

    def validate_default_data(self, dataset_root: str, write_json: bool):
        # Check that all data files exist and match the default contents.
        data_keys = ["tracks", "times", "centroids", "outliers", "bounds"]
        with open(dataset_root / "manifest.json", "r") as f:
            manifest: DatasetManifest = json.load(f)
            for key in data_keys:
                data_path = dataset_root / manifest[key]
                if write_json:
                    self.validate_json(data_path, self.default_data[key])
                else:
                    self.validate_parquet(data_path, self.default_data[key])

    def test_write_data_writes_json_data(self, tmp_path):
        writer = ColorizerDatasetWriter(tmp_path, DEFAULT_DATASET_NAME)
        self.write_default_data(writer, write_json=True)
        writer.write_manifest()
        self.validate_default_data(tmp_path / DEFAULT_DATASET_NAME, write_json=True)

    def test_write_data_writes_parquet_data(self, tmp_path):
        writer = ColorizerDatasetWriter(tmp_path, DEFAULT_DATASET_NAME)
        self.write_default_data(writer, write_json=False)
        writer.write_manifest()
        self.validate_default_data(tmp_path / DEFAULT_DATASET_NAME, write_json=False)
