import numpy as np
from colorizer_data.types import FeatureInfo, FeatureType
from colorizer_data.utils import cast_feature_to_info_type, infer_feature_type
import pytest


def test_infer_feature_type_detects_categories():
    info = FeatureInfo(type=FeatureType.INDETERMINATE, categories=["False, True"])
    data = np.array([0, 1, 0, 1])
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
    return np.array([0, 1, 2, 3], dtype=int)


@pytest.fixture
def float_data():
    return np.array([0.2, 0.5, 0.7, 1.2], dtype=float)


@pytest.fixture
def string_data():
    return np.array(["a", "a", "b", "c", "d", "a"], dtype=str)


def test_infer_feature_type_detects_strings():
    info = FeatureInfo(type=FeatureType.INDETERMINATE)
    data = np.array(["0.1", "1.2", "2.3", "3.4"], dtype=str)
    assert infer_feature_type(data, info) == FeatureType.CATEGORICAL

    data = np.array(["A", "B", "C", "D"])
    assert infer_feature_type(data, info) == FeatureType.CATEGORICAL


def test_cast_feature_to_info_type_exceptions():
    with pytest.raises(Exception):
        cast_feature_to_info_type(*string_data, *continuous_info)

    with pytest.raises(Exception):
        cast_feature_to_info_type(*string_data, *discrete_info)


def test_cast_feature_to_info_type_handles_string_arrays():
    categorical_info = FeatureInfo(
        type=FeatureType.INDETERMINATE, categories=["a", "b", "c", "d"]
    )
    string_data = np.array(["a", "a", "b", "c", "d", "a"], dtype=str)

    data, info = cast_feature_to_info_type(string_data, categorical_info)
    assert info.type == FeatureType.CATEGORICAL
    assert data.dtype.kind == "i"
    assert data.tolist() == [0, 0, 1, 2, 3, 0]
    assert info.categories == ["a", "b", "c", "d"]


def test_cast_feature_to_info_type_does_not_modify_matching_types(
    float_data, int_data, continuous_info, discrete_info, categorical_info
):
    continuous_data, continuous_info_new = cast_feature_to_info_type(
        float_data, continuous_info
    )
    assert np.array_equal(continuous_data, float_data)
    assert np.array_equal(continuous_info_new, continuous_info)

    discrete_data, discrete_info_new = cast_feature_to_info_type(
        int_data, discrete_info
    )
    assert np.array_equal(discrete_data, int_data)
    assert np.array_equal(discrete_info_new, discrete_info)

    categorical_data, categorical_info_new = cast_feature_to_info_type(
        int_data, categorical_info
    )
    assert np.array_equal(categorical_data, int_data)
    assert np.array_equal(categorical_info_new, categorical_info)


def test_cast_feature_to_info_type_accepts_bad_categories_length(
    int_data, categorical_info
):
    categorical_info.categories = ["a"]
    # Does nothing
    cast_feature_to_info_type(int_data, categorical_info)
