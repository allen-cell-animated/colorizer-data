# TODO: Write tests for the writer class. See https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html.
import json
import os
from pathlib import Path
from typing import Tuple
import numpy as np
import pytest

from colorizer_data.types import CURRENT_VERSION, ColorizerMetadata, DatasetManifest
from colorizer_data.writer import ColorizerDatasetWriter

DEFAULT_DATASET_NAME = "dataset"

EXISTING_MANIFEST_CONTENT: DatasetManifest = {
    "features": [],
    "frames": [],
    "metadata": {
        "name": "my example dataset",
        "description": "description of my example dataset",
        "author": "john",
        "dateCreated": "2000-01-01T01:00:00.000Z",
        "lastModified": "2000-01-01T02:00:00.000Z",
        "revision": 4,
        "writerVersion": "v0.4.0",
        "frameWidth": 500,
        "frameHeight": 340,
        "frameUnits": "um",
        "startTimeSeconds": 120,
        "frameDurationSeconds": 0.5,
        "startingFrameNumber": 12,
    },
}

DEPRECATED_MANIFEST_FEATURES: DatasetManifest = {
    "metadata": {
        "frameDims": {"width": 500, "height": 340, "units": "um"},
    }
}

BLANK_MANIFEST_CONTENT: DatasetManifest = {
    "features": [],
    "frames": [],
}

# TODO: Test deprecated manifest feature (should be ignored
# TODO: Test somewhere frame width, height, units, start time, frame duration, frame number, etc.
# in case of breaking API changes.


def setup_dummy_writer_data(writer: ColorizerDatasetWriter):
    writer.write_data(times=np.ndarray([]))
    writer.set_frame_paths([""])


@pytest.fixture
def existing_manifest(tmp_path) -> Tuple[ColorizerDatasetWriter, Path, Path]:
    directory = tmp_path / DEFAULT_DATASET_NAME
    directory.mkdir()
    manifest_path = tmp_path / DEFAULT_DATASET_NAME / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(EXISTING_MANIFEST_CONTENT, f, indent=2)
    writer = ColorizerDatasetWriter(tmp_path, DEFAULT_DATASET_NAME)
    setup_dummy_writer_data(writer)
    return writer, tmp_path, manifest_path


@pytest.fixture
def blank_manifest(tmp_path) -> Tuple[ColorizerDatasetWriter, Path, Path]:
    directory = tmp_path / DEFAULT_DATASET_NAME
    directory.mkdir()
    manifest_path = tmp_path / DEFAULT_DATASET_NAME / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(BLANK_MANIFEST_CONTENT, f, indent=2)
    writer = ColorizerDatasetWriter(tmp_path, DEFAULT_DATASET_NAME)
    setup_dummy_writer_data(writer)
    return writer, tmp_path, manifest_path


def test_write_and_read_new_metadata(tmp_path):
    # Test writing new manifest w/ metadata and validate that manifest structure is as expected

    writer = ColorizerDatasetWriter(tmp_path, DEFAULT_DATASET_NAME)
    setup_dummy_writer_data(writer)
    writer.write_manifest(
        metadata=ColorizerMetadata(
            name="my dataset",
            author="me",
            frame_width=80.0,
            frame_height=60.0,
            frame_units="picometers",
        )
    )

    expected_manifest = tmp_path / DEFAULT_DATASET_NAME / "manifest.json"
    assert os.path.exists(expected_manifest)
    manifest: DatasetManifest = {}
    with open(expected_manifest, "r") as f:
        manifest = json.load(f)
    manifest["metadata"] = ColorizerMetadata.from_dict(manifest["metadata"])

    assert manifest["metadata"].name == "my dataset"
    assert manifest["metadata"].author == "me"
    assert manifest["metadata"].frame_width == 80.0
    assert manifest["metadata"].frame_height == 60.0
    assert manifest["metadata"].frame_units == "picometers"


def test_writer_updates_revision_and_time(existing_manifest):
    # Should update revision number, updated time, and data version

    writer, tmp_path, manifest_path = existing_manifest
    writer.write_manifest()

    with open(manifest_path, "r") as f:
        manifest: DatasetManifest = json.load(f)
        metadata = manifest["metadata"]
        oldMetadata = EXISTING_MANIFEST_CONTENT["metadata"]

        # Updates expected fields
        assert metadata["lastModified"] != oldMetadata["lastModified"]
        assert metadata["revision"] == oldMetadata["revision"] + 1
        assert metadata["writerVersion"] == CURRENT_VERSION


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
            revision=250,
            writer_version="abcdef",
        )
    )

    with open(manifest_path, "r") as f:
        manifest: DatasetManifest = json.load(f)
        metadata = manifest["metadata"]

        # Leaves other fields untouched
        assert metadata["name"] == "new name"
        assert metadata["description"] == "new description"
        assert metadata["author"] == "geoff"
        assert metadata["dateCreated"] == "some-date"
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

        # Leaves other fields untouched
        assert metadata.name == DEFAULT_DATASET_NAME
        assert metadata.date_created != None
        assert metadata.date_created == metadata.last_modified
        assert metadata.writer_version == CURRENT_VERSION
        assert metadata.revision == 0
