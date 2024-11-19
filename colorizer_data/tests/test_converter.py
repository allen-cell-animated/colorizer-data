import pandas as pd
import pytest
from colorizer_data import convert_colorizer_data


def test_handles_default_csv(tmp_path):
    csv_path = "colorizer_data/tests/assets/basic_csv/data.csv"
    data = pd.read_csv(csv_path)
    convert_colorizer_data(
        data,
        tmp_path / "dataset",
    )
