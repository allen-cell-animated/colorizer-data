# TODO: Write tests for the writer class. See https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html.
import json
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
        "dateCreated": "01/01/2000, 01:00:00 PDT-0700",
        "lastModified": "01/01/2000, 02:00:00 PDT-0700",
        "revision": 4,
        "dataVersion": "v0.4.0",
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


@pytest.fixture
def existing_manifest(tmp_path) -> Tuple[Path, Path]:
    directory = tmp_path / DEFAULT_DATASET_NAME
    directory.mkdir()
    manifest_path = tmp_path / DEFAULT_DATASET_NAME / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(EXISTING_MANIFEST_CONTENT, f, indent=2)
    writer = ColorizerDatasetWriter(tmp_path, DEFAULT_DATASET_NAME)
    setup_dummy_writer_data(writer)
    return writer, tmp_path, manifest_path


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
        assert metadata["dataVersion"] == CURRENT_VERSION


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
            data_version="abcdef",
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
        assert metadata["dataVersion"] == "abcdef"


def test_writer_updates_revision_and_time_when_none():
    # Update revision number, creation time, updated time, and data version if
    # base manifest does not include this information
    pass
