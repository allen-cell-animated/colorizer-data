import numpy as np
from colorizer_data.types import FeatureInfo, FeatureType
from colorizer_data.utils import cast_feature_to_info_type, infer_feature_type
import pytest


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
