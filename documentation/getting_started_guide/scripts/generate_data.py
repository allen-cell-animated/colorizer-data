import numpy as np
import pandas as pd
from skimage import draw
from bioio.writers import OmeTiffWriter

import os

"""
Generate some sample data for the `GETTING_STARTED` tutorial.

Run with
```
python generate_data.py
```
"""

directory = "raw_dataset"
os.makedirs(directory, exist_ok=True)

# Create the data frame; this will be turned into a CSV file.
df = pd.DataFrame(
    columns=[
        "segmentation_id",
        "track_id",
        "time",
        "centroid_x",
        "centroid_y",
        "area",
        "radius",
        "location",
        "segmentation_path",
    ]
)

# Generate the images.
frame_dimensions = (200, 200)
images = []
num_frames = 20
num_circles = 5
circle_base_radius = 15
circle_min_radius = 3
circle_max_radius = frame_dimensions[0] // num_circles
circle_radius_variance = 2
circle_max_position_change = 15

circle_last_y_position = np.full(num_circles, frame_dimensions[0] / 2)
circle_last_radius = np.full(num_circles, circle_base_radius)

for i in range(num_frames):
    image = np.zeros(frame_dimensions, dtype=np.uint8)
    t = i / num_frames

    # Draw each of the circles on the image.
    for j in range(num_circles):
        # Randomize the radius and position of the circle.
        radius = circle_last_radius[j] + np.random.randint(
            -circle_radius_variance, circle_radius_variance
        )
        radius = min(max(radius, circle_min_radius), circle_max_radius)
        circle_last_radius[j] = radius

        x = (j + 1) * (frame_dimensions[1] / (num_circles + 1))
        y = circle_last_y_position[j] + np.random.randint(
            -circle_max_position_change, circle_max_position_change
        )
        # Clamp to make sure the circle stays within the frame
        y = min(max(y, circle_max_radius), frame_dimensions[0] - circle_max_radius)
        circle_last_y_position[j] = y

        # Draw the circle in the segmentation image, filling it with the
        # segmentation ID. (0 is reserved for the background, so we add 1 to the
        # segmentation ID to avoid conflicts.)
        rr, cc = draw.disk((y, x), radius)
        segmentation_id = j + 1
        image[rr, cc] = segmentation_id

        # Add the circle's data to the data frame.
        # Calculate any additional features
        circle_area = np.pi * radius**2
        circle_height = y

        # Location is a categorical feature based on the y position of the circle.
        location = "middle"
        if y < frame_dimensions[0] * 0.4:
            location = "top"
        elif y > frame_dimensions[0] * 0.6:
            location = "bottom"

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "segmentation_id": segmentation_id,
                        "track_id": j,
                        "time": i,
                        "centroid_x": round(x),
                        "centroid_y": round(y),
                        "area": round(circle_area, 1),
                        "radius": round(radius),
                        "location": location,
                        "segmentation_path": f"frame_{i}.tiff",
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

    images.append(image)

# Write the resulting data frame + images to disk.
df.to_csv(f"{directory}/data.csv", index=False)
for i, image in enumerate(images):
    tiff_writer = OmeTiffWriter()
    tiff_writer.save(image, f"{directory}/frame_{i}.tiff")
