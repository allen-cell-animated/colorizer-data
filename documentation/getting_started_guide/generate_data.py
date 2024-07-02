import numpy as np
import pandas as pd
from skimage import draw
from bioio import BioImage

"""
Generate some sample data for the `GETTING_STARTED` tutorial.
"""

# Create the data frame; this will be turned into a CSV file.
df = pd.DataFrame(
    columns=[
        "object_id",
        "track_id",
        "time",
        "centroid_x",
        "centroid_y",
        "area",
        "height",
        "segmentation_path",
    ]
)

# Generate the images.
frame_dimensions = (100, 100)
images = []
num_frames = 10
num_circles = 5
circle_base_radius = 8
circle_radius_variance = 2
circle_max_speed = 3

circle_last_y_position = np.full(num_circles, 50)
circle_last_radius = np.full(num_circles, circle_base_radius)

# TODO: Make a prettier pattern, like the circles spinning?
for i in range(num_frames):
    image = np.zeros((100, 100), dtype=np.uint8)
    t = i / num_frames

    for j in range(num_circles):
        # Draw the circle on the image
        radius = circle_last_radius[j] + np.random.randint(
            -circle_radius_variance, circle_radius_variance
        )
        circle_last_radius[j] = radius

        x = (j + 1) * (frame_dimensions[0] / (num_circles + 1))
        print(circle_last_y_position)
        y = circle_last_y_position[j] + np.random.randint(
            -circle_max_speed, circle_max_speed
        )
        circle_last_y_position[j] = y

        rr, cc = draw.disk((x, y), radius)
        print(rr, cc)

        object_id = i * num_circles + j
        image[rr, cc] = object_id

        # Add the circle to the data frame
        circle_area = np.pi * radius**2
        circle_height = y
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "object_id": object_id,
                        "track_id": j,
                        "time": i,
                        "centroid_x": x,
                        "centroid_y": y,
                        "area": circle_area,
                        "height": circle_height,
                        "segmentation_path": f"frame_{i}.png",
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

    # Write the image
    images.append(image)

# Write the resulting data frame + images to disk.
df.to_csv("data.csv", index=False)
for i, image in enumerate(images):
    bioimage = BioImage(image)
    bioimage.save(f"frame_{i}.png")
