import json
from pathlib import Path
from typing import List
import numpy as np
from colorizer_data.types import (
    CURRENT_VERSION,
    CollectionDatasetEntry,
    CollectionManifest,
    CollectionMetadata,
    FeatureInfo,
    FeatureType,
)
from colorizer_data.utils import (
    cast_feature_to_info_type,
    infer_feature_type,
    merge_dictionaries,
    replace_out_of_bounds_values_with_nan,
    update_collection,
)
import pytest

DEFAULT_DATASETS: List[CollectionDatasetEntry] = [
    {"name": "dataset1", "path": "dataset1_path"},
    {"name": "dataset2", "path": "dataset2_path"},
    {"name": "dataset3", "path": "dataset3_path"},
]
DEFAULT_COLLECTION_JSON: CollectionManifest = {
    "datasets": DEFAULT_DATASETS,
    "metadata": {
        "name": "c_name",
        "description": "c_description",
        "author": "c_author",
        "collectionVersion": "c_collection_version",
        "dateCreated": "2024-01-01T00:00Z",
        "lastModified": "2024-01-01T00:00Z",
        "revision": 3,
        "writerVersion": "v0.0.0",
    },
}


@pytest.fixture
def default_collection_json(tmp_path) -> Path:
    collection_path = tmp_path / "collection.json"
    with open(collection_path, "w") as f:
        json.dump(DEFAULT_COLLECTION_JSON, f)
    return collection_path


def test_update_collection_handles_deprecated_format(tmp_path):
    collection_path = tmp_path / "collection.json"
    # Make an collection array JSON file
    with open(collection_path, "w") as f:
        json.dump(DEFAULT_DATASETS, f)

    update_collection(collection_path, "dataset4", "dataset4_path")

    # File should be converted to collection object
    with open(collection_path, "r") as f:
        collection: CollectionManifest = json.load(f)

        # Check existing data was kept
        assert len(collection["datasets"]) == 4
        assert collection["datasets"][0]["name"] == "dataset1"
        assert collection["datasets"][1]["name"] == "dataset2"
        assert collection["datasets"][2]["name"] == "dataset3"
        assert collection["datasets"][3]["name"] == "dataset4"

        metadata = CollectionMetadata.from_dict(collection["metadata"])
        assert metadata._revision == 0
        assert metadata.date_created is not None
        assert metadata.last_modified == metadata.date_created


def test_update_collection_creates_file_if_none(tmp_path):
    collection_path = tmp_path / "collection.json"
    update_collection(collection_path, "dataset", "dataset_path")

    with open(collection_path, "r") as f:
        collection: CollectionManifest = json.load(f)

        assert len(collection["datasets"]) == 1
        assert collection["datasets"][0]["name"] == "dataset"
        assert collection["datasets"][0]["path"] == "dataset_path"

        metadata = CollectionMetadata.from_dict(collection["metadata"])
        assert metadata._revision == 0
        assert metadata.date_created is not None
        assert metadata.last_modified == metadata.date_created


def test_update_collection_overwrites_existing_entry(default_collection_json):
    update_collection(default_collection_json, "dataset2", "custom_dataset2_path")

    with open(default_collection_json, "r") as f:
        collection: CollectionManifest = json.load(f)
        assert len(collection["datasets"]) == 3
        assert collection["datasets"][0]["name"] == "dataset1"
        assert collection["datasets"][0]["path"] == "dataset1_path"
        assert collection["datasets"][1]["name"] == "dataset2"
        assert collection["datasets"][1]["path"] == "custom_dataset2_path"
        assert collection["datasets"][2]["name"] == "dataset3"
        assert collection["datasets"][2]["path"] == "dataset3_path"

        # Check that user-defined metadata values are not overridden
        metadata = CollectionMetadata.from_dict(collection["metadata"])
        assert metadata.name == "c_name"
        assert metadata.description == "c_description"
        assert metadata.author == "c_author"
        assert metadata.collection_version == "c_collection_version"
        assert metadata.date_created == "2024-01-01T00:00Z"

        # Check that auto-updated fields are updated as expected
        assert metadata.last_modified != "2024-01-01T00:00Z"
        assert metadata._writer_version == CURRENT_VERSION
        assert metadata._revision == 4


def test_update_collection_adds_dataset_entry(default_collection_json):
    update_collection(default_collection_json, "dataset4", "dataset4_path")

    with open(default_collection_json, "r") as f:
        collection: CollectionManifest = json.load(f)
        assert len(collection["datasets"]) == 4
        assert collection["datasets"][3]["name"] == "dataset4"
        assert collection["datasets"][3]["path"] == "dataset4_path"

    update_collection(default_collection_json, "dataset5", "dataset5_path")
    update_collection(default_collection_json, "dataset6", "dataset6_path")

    with open(default_collection_json, "r") as f:
        collection: CollectionManifest = json.load(f)
        assert len(collection["datasets"]) == 6
        assert collection["datasets"][4]["name"] == "dataset5"
        assert collection["datasets"][4]["path"] == "dataset5_path"
        assert collection["datasets"][5]["name"] == "dataset6"
        assert collection["datasets"][5]["path"] == "dataset6_path"


def test_update_collection_writes_metadata(default_collection_json):
    update_collection(
        default_collection_json,
        "dataset4",
        "dataset4_path",
        metadata=CollectionMetadata(
            name="new_name",
            description="new_description",
            author="new_author",
            collection_version="new_collection_version",
            date_created="2020-01-01T00:00Z",
            last_modified="2020-01-01T00:00Z",
            _revision=50,
            _writer_version="v0.0.0",
        ),
    )

    with open(default_collection_json, "r") as f:
        collection: CollectionManifest = json.load(f)

        metadata = CollectionMetadata.from_dict(collection["metadata"])
        assert metadata.name == "new_name"
        assert metadata.description == "new_description"
        assert metadata.author == "new_author"
        assert metadata.collection_version == "new_collection_version"
        assert metadata.date_created == "2020-01-01T00:00Z"

        # Overrides default updating behavior
        assert metadata.last_modified == "2020-01-01T00:00Z"
        assert metadata._revision == 50
        assert metadata._writer_version == "v0.0.0"


def test_infer_feature_type_detects_categories():
    info = FeatureInfo(type=FeatureType.INDETERMINATE, categories=["False, True"])
    data = np.array([0, 1, 0, 1])
    assert infer_feature_type(data, info) == FeatureType.CATEGORICAL


def test_infer_feature_type_detects_categories_via_dtype():
    info = FeatureInfo(type=FeatureType.INDETERMINATE)
    data = np.array(["0.1", "1.2", "2.3", "3.4"], dtype=str)
    assert infer_feature_type(data, info) == FeatureType.CATEGORICAL

    data = np.array(["A", "B", "C", "D"])
    assert infer_feature_type(data, info) == FeatureType.CATEGORICAL


def test_infer_feature_type_detects_categories_on_mixed_dtype():
    info = FeatureInfo(type=FeatureType.INDETERMINATE)
    data = np.array(["0.1", 0.0, 4, "C"], dtype=str)
    assert infer_feature_type(data, info) == FeatureType.CATEGORICAL


def test_infer_feature_type_detects_integers():
    info = FeatureInfo(type=FeatureType.INDETERMINATE)
    data = np.array([0, 1, 2, 3], dtype=int)
    assert infer_feature_type(data, info) == FeatureType.DISCRETE


def test_infer_feature_type_detects_floats():
    info = FeatureInfo(type=FeatureType.INDETERMINATE)
    data = np.array([0.1, 1.2, 2.3, 3.4], dtype=float)
    assert infer_feature_type(data, info) == FeatureType.CONTINUOUS


@pytest.fixture
def categorical_info():
    return FeatureInfo(type=FeatureType.CATEGORICAL, categories=["a", "b", "c", "d"])


@pytest.fixture
def continuous_info():
    return FeatureInfo(type=FeatureType.CONTINUOUS)


@pytest.fixture
def discrete_info():
    return FeatureInfo(type=FeatureType.DISCRETE)


@pytest.fixture
def int_data():
    return np.array([0, 1, 2, 3, 4], dtype=int)


@pytest.fixture
def float_data():
    return np.array([0.2, 0.5, 0.7, 1.2, np.nan, 3], dtype=float)


@pytest.fixture
def string_data():
    return np.array(["a", "a", "b", "c", "d", "a"], dtype=str)


def test_cast_feature_to_info_type_exceptions(
    string_data, continuous_info, discrete_info
):
    with pytest.raises(Exception):
        cast_feature_to_info_type(string_data, continuous_info)

    with pytest.raises(Exception):
        cast_feature_to_info_type(string_data, discrete_info)


def test_cast_feature_to_info_type_infers_categories():
    # Gets categories in order of appearance
    empty_info = FeatureInfo(type=FeatureType.INDETERMINATE)
    string_data = np.array(["a", "a", "b", "c", "d", "a"])

    data, info = cast_feature_to_info_type(string_data, empty_info)
    assert info.type == FeatureType.CATEGORICAL
    assert np.array_equal(data, [0, 0, 1, 2, 3, 0], True)
    assert info.categories == ["a", "b", "c", "d"]


def test_cast_feature_to_info_type_handles_none_and_nan_values_in_categorical_data():
    # Safely ignores None values during auto-casting
    empty_info = FeatureInfo(type=FeatureType.INDETERMINATE)
    data = np.array([None, None, "a", None, "b", np.NaN], dtype=object)

    data, info = cast_feature_to_info_type(data, empty_info)
    assert info.type == FeatureType.CATEGORICAL
    assert np.array_equal(data, [np.nan, np.nan, 0, np.nan, 1, np.NaN], True)
    assert info.categories == ["a", "b"]


def test_cast_feature_to_info_type_keeps_old_categories_when_provided():
    # Replaces unrecognized feature values with `np.nan`.
    categorical_info = FeatureInfo(
        type=FeatureType.INDETERMINATE, categories=["a", "b", "c"]
    )
    string_data = np.array(["d", "b", "a", "a", "b", "c", "d", None], dtype=str)

    data, info = cast_feature_to_info_type(string_data, categorical_info)
    assert info.type == FeatureType.CATEGORICAL
    assert np.array_equal(data, [np.nan, 1, 0, 0, 1, 2, np.nan, np.nan], equal_nan=True)
    assert info.categories == ["a", "b", "c"]


def test_cast_feature_to_info_type_truncates_floats_for_discrete_features(
    discrete_info,
):
    data, info = cast_feature_to_info_type(
        np.array([0.2, 0.5, 0.7, 1.2, 5], dtype=float), discrete_info
    )
    expected_data = np.array([0, 0, 0, 1, 5], dtype=int)
    assert info.type == FeatureType.DISCRETE
    assert data.dtype.kind == "i"
    assert np.array_equal(expected_data, data)


def test_cast_feature_to_info_type_keeps_discrete_data_as_float_for_nan_values(
    discrete_info,
):
    # NaN values are unrepresentable with int, so expect float type with truncated
    # integer values.
    data, info = cast_feature_to_info_type(
        np.array([0.2, 0.5, 0.7, 1.2, 5, np.nan], dtype=float), discrete_info
    )
    expected_data = np.array([0, 0, 0, 1, 5, np.nan], dtype=float)
    assert data.dtype.kind == "f"
    assert info.type == FeatureType.DISCRETE
    assert np.array_equal(expected_data, data, True)


def test_cast_feature_to_info_type_handles_nan_in_categorical_data():
    data = np.array([0, 1, 3, 2, np.nan], dtype=float)
    info = FeatureInfo(type=FeatureType.CATEGORICAL, categories=["a", "b", "c", "d"])

    cast_data, cast_info = cast_feature_to_info_type(data, info)
    assert cast_data.dtype.kind == "f"
    assert cast_info.type == FeatureType.CATEGORICAL
    assert np.array_equal(cast_data, data, True)


def test_cast_feature_to_info_type_does_not_modify_matching_types(
    float_data, int_data, continuous_info, discrete_info, categorical_info
):
    continuous_data, continuous_info_new = cast_feature_to_info_type(
        float_data, continuous_info
    )
    assert np.array_equal(continuous_data, float_data, True)
    assert continuous_info_new == continuous_info

    discrete_data, discrete_info_new = cast_feature_to_info_type(
        int_data, discrete_info
    )
    assert np.array_equal(discrete_data, int_data)
    assert discrete_info_new == discrete_info

    categorical_data, categorical_info_new = cast_feature_to_info_type(
        int_data, categorical_info
    )
    assert np.array_equal(categorical_data, int_data)
    assert categorical_info_new == categorical_info


def test_cast_feature_to_info_type_accepts_bad_categories_length(
    int_data, categorical_info
):
    categorical_info.categories = ["a"]
    # Does nothing
    cast_feature_to_info_type(int_data, categorical_info)


def test_replace_out_of_bounds_values_with_nan():
    data = np.array([-1, 0, 2, 4, 5, 10, np.nan])
    expected = np.array([np.nan, 0, 2, 4, np.nan, np.nan, np.nan])
    replace_out_of_bounds_values_with_nan(data, 0, 4)
    assert np.array_equal(data, expected, True)


def test_merge_dictionaries_ignores_none_and_missing():
    a = {"a": 1, "b": 10, "c": "some-value"}
    b = {"a": None, "b": 0}
    result = merge_dictionaries(a, b)
    assert result["a"] == 1
    assert result["b"] == 0
    assert result["c"] == "some-value"


def test_merge_dictionaries():
    a = {"a": 1, "b": 10, "c": "some-value"}
    b = {"a": 2, "b": 0, "c": "some-other-value"}
    result = merge_dictionaries(a, b)
    assert result["a"] == 2
    assert result["b"] == 0
    assert result["c"] == "some-other-value"


def test_merge_dictionaries_handles_nesting():
    # Handles merging (including None values) for nested
    # dictionary structures
    a = {
        "1": {"1": "a", "2": "a"},
        "2": {
            "1": {"1": "a", "2": "a"},
            "2": "a",
        },
        "3": {"1": "a"},
    }
    b = {
        "1": {"1": "b", "2": None},
        "2": {
            "1": {"1": "b", "2": None},
            "2": "b",
        },
        "3": None,
    }
    result = merge_dictionaries(a, b)
    assert result["1"]["1"] == "b"
    assert result["1"]["2"] == "a"
    assert result["2"]["1"]["1"] == "b"
    assert result["2"]["1"]["2"] == "a"
    assert result["2"]["2"] == "b"
    assert result["3"]["1"] == "a"
