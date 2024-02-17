import numpy as np
from colorizer_data.types import FeatureInfo, FeatureType
from colorizer_data.utils import infer_feature_type
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


def test_infer_feature_type_detects_strings():
    info = FeatureInfo(type=FeatureType.INDETERMINATE)
    data = np.array(["0.1", "1.2", "2.3", "3.4"], dtype=str)
    assert infer_feature_type(data, info) == FeatureType.CATEGORICAL

    data = np.array(["A", "B", "C", "D"])
    assert infer_feature_type(data, info) == FeatureType.CATEGORICAL


# TODO: Test all the error handlers for each of the feature types
# TODO: Check behavior of auto-retrieving categories
