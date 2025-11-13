import collections
from io import StringIO
import os
from bioio_ome_tiff.writers import OmeTiffWriter
from colorizer_data import convert_colorizer_data
import numpy as np
import pandas as pd

from colorizer_data.types import Frames3dMetadata
from colorizer_data.utils import update_collection


image_dims = [3, 300]

for image_dim in image_dims:
    # TCZYX
    image = np.ndarray([2, 1, image_dim, image_dim, image_dim], dtype=np.uint16)
    image.fill(0)

    base_path = "S:/aics/users/peyton.lee/test-data/test-scale"
    dataset_folder = "test-scale-{}".format(image_dim)
    dataset_path = os.path.join(base_path, dataset_folder)

    os.makedirs(dataset_path, exist_ok=True)
    PixelSizes = collections.namedtuple("PhysicalPixelSizes", ["X", "Y", "Z"])
    OmeTiffWriter.save(
        image,
        dataset_path + "/file.ome.tif",
        dim_order="TCZYX",
        physical_pixel_sizes=PixelSizes(1, 1, 1),
    )

    sample_csv_headers = "ID,Track,Frame,Centroid X,Centroid Y,Centroid Z,Discrete Feature,Categorical Feature,Outlier"
    raw_sample_csv_data = [
        "0,1,0,0,0,0,0,A,0",
        f"1,1,1,{image_dim},{image_dim},{image_dim},1,B,0",
    ]
    sample_csv_data = sample_csv_headers + "\n" + "\n".join(raw_sample_csv_data)

    csv_data = pd.read_csv(StringIO(sample_csv_data))

    frames3d = Frames3dMetadata(
        source="file.ome.tif",
        segmentation_channel=0,
        total_frames=2,
    )

    convert_colorizer_data(
        csv_data,
        dataset_path,
        object_id_column="ID",
        times_column="Frame",
        track_column="Track",
        centroid_x_column="Centroid X",
        centroid_y_column="Centroid Y",
        centroid_z_column="Centroid Z",
        frames_3d=frames3d,
        image_column=None,
    )

    update_collection(
        "{}/collection.json".format(base_path),
        "{}x{}x{}".format(image_dim, image_dim, image_dim),
        dataset_folder,
    )
