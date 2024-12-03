import pandas as pd
from colorizer_data import convert_colorizer_data

"""
To test, run the following commands:
```
cd documentation/getting_started_guide/raw_dataset
python ../../../colorizer_data/tests/test_api.py
```

"""

if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    convert_colorizer_data(
        data,
        "converted_data",
        object_id_column="object_id",
        track_column="track_id",
        times_column="time",
        centroid_x_column="centroid_x",
        centroid_y_column="centroid_y",
        image_column="segmentation_path",
    )
