import json
import os
from pathlib import Path
from typing import Optional, Tuple
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
    writer.write_data(times=np.ndarray([]))
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
        assert metadata["lastModified"] != None
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
        assert metadata.date_created != None
        assert metadata.date_created == metadata.last_modified
        assert metadata._writer_version == CURRENT_VERSION
        assert metadata._revision == 0
