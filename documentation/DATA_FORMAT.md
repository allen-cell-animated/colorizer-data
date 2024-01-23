# Timelapse-Colorizer Data Format

Timelapse-Colorizer can only load datasets that follow the defined data specification.

The easiest way to get started is to modify one of our existing data processing scripts, like [`convert_nucmorph_data.py`](../timelapse-colorizer-data/convert_nucmorph_data.py)!

(Check with your team or one of the developers on the Animated Cell team to see if there's already a data generation script for your project!)

## Terms

Here are a few important terms:

- **Dataset**: A dataset is a single time-series, and can have any number of tracked objects and features.
- **Collection**: An arbitrary grouping of datasets.
- **Object ID**: Every segmentation object in every frame has an integer identifier that is unique across all time steps. This identifier will be used to map an object to relevant data. Object IDs must be sequential, starting from 0, across the whole dataset.

## Dataset

A dataset consists of a group of files that describe the segmentations, tracks, feature data, processed images, and additional metadata for a single time-series.

The most important file is the **manifest**, which is a JSON file that describes all the files in the dataset. (Manifests should be named `manifest.json` by default.)

`manifest.json:`

```txt
{
    "frames": [
        <relative path to image frame 0>,
        <relative path to image frame 1>,
        ...
    ],
    "features": [
        {
            "key": <feature key>,                                 // Must be unique across features. Lowercase alphanumeric characters and underscores only.
            "name": <feature display name>,
            "data": <relative path to feature JSON>,
            "unit": <unit label>,                                 //< optional
            "type": <"continuous" | "discrete" | "categorical">,  //< optional
            "categories": [<category 1>, <category 2>, ...,]      //< optional unless type is "categorical"; max 12 categories
        }
        {
            "name": <feature display name>,
            ...
        },
        ...
    ],
    "tracks": <relative path to tracks JSON>,
    "times": <relative path to times JSON>,
    "outliers": <relative path to outlier JSON>,    //< optional
    "centroids": <relative path to centroids JSON>, //< optional
    "bounds": <relative path to bounds JSON>        //< optional
    "backdrops": <array of backdrop image sets>     //< optional, see 2. Backdrops for more details
}
```

*Note: all paths are relative to the location of the manifest file.*

Note that the `outliers`, `centroids`, and `bounds` files are optional, but certain features of Timelapse-Colorizer won't work without them.

**Features** can also define additional optional metadata, such as the units and type. Note that there are additional restrictions on some of these fields. **`type`** must have values `continuous` for floats or decimals, `discrete` for integers, or `categorical` for distinct labels.

Features that have the `categorical` type must also define an array of string `categories`, up to a maximum of 12.

A complete example dataset is also available in the [`documentation`](./example_dataset) directory of this project, and can be [viewed on Timelapse-Colorizer](https://dev-aics-dtp-001.int.allencell.org/nucmorph-colorizer/dist/?dataset=https://raw.githubusercontent.com/allen-cell-animated/colorizer-data/main/documentation/example_dataset/manifest.json).

<details>
<summary><b>[Show me an example!]</b></summary>

---

An example dataset directory could look like this:

```txt
📂 my_dataset/
  - 📄 manifest.json
  - 📄 outliers.json
  - 📄 tracks.json
  - 📄 times.json
  - 📄 centroids.json
  - 📄 bounds.json
  - 📕 feature_0.json
  - 📗 feature_1.json
  - 📘 feature_2.json
  - 📁 frames/
    - 📷 frame_0.png
    - 📷 frame_1.png
    - 📷 frame_2.png
    ...
    - 📷 frame_245.png
```

The `manifest.json` file would look something like this:

`manifest.json:`

```txt
{
    "frames": [
        "frames/frame_0.png",
        "frames/frame_1.png",
        "frames/frame_2.png",
        ...
        "frames/frame_245.png",
    ],
    "features": [
        {
            "key": "temperature",
            "name": "Temperature",
            "data": "feature_0.json",
            "unit": "°C",
            "type": "continuous"
        },
        {
            "key": "neighboring_cells",
            "name": "Neighboring Cells",
            "data": "feature_1.json",
            "unit": "cell(s)",
            "type": "discrete"
        },
        {
            "key": "life_stage",
            "name": "Life Stage",
            "data": "feature_2.json",
            "type": "categorical",
            "categories": ["G1", "S", "G2", "Prophase", "Metaphase", "Anaphase", "Telophase" ]
        },
    ],
    "tracks": "tracks.json",
    "times": "times.json",
    "outliers": "outliers.json",
    "centroids": "centroids.json",
    "bounds": "bounds.json",
    "backdrops": [],
}
```

See the [included example dataset](./example_dataset) for another example of backdrop images in action.

---

</details>

### 1. Metadata

Manifests can include some optional **metadata** about the dataset and its features.

Besides the details shown above, these are additional parameters that the manifest can include:

`manifest.json:`

```txt
{
    ...
    "metadata": {
        "frameDims": {
            "units": <unit label for frame dimensions>,
            "width": <width of frame in units>,
            "height": <height of frame in units>
        },
        "frameDurationSeconds": <duration of a frame in seconds>,
        "startTimeSeconds": <start time of timestamp in seconds>  // 0 by default
    }

}
```

These metadata parameters are used to configure additional features of the Timelapse Colorizer UI, such as showing scale bars or timestamps on the main display. Additional metadata will likely be added as the project progresses.

Note that the interface will directly show the unit labels and does not scale or convert units from one type to another (for example, it will not convert 1000 µm to 1 mm). If you need to present your data with different units, create a (scaled) duplicate of the feature with a different unit label.

<details>
<summary><b>[Show me an example!]</b></summary>

---

Let's say a dataset has a microscope viewing area 3200 µm wide by 2400 µm tall, and there are 5 minutes (`=300 seconds`) between each frame. We also want to show the timestamp in colony time, which started 30 minutes (`=1800 seconds`) before the start of the recording.

The manifest file would look something like this:

`manifest.json:`

```txt
{
    ...,
    "metadata": {
        "frameDims": {
            "width": 3200,
            "height": 2400,
            "units": "µm"
        },
        "frameDurationSeconds": 300,
        "startTimeSeconds": 1800
    }
}

```

---

</details>

### 2. Backdrops (optional)

Multiple sets of **backdrop images** can be included in the manifest, which will be shown behind the colored objects in the UI. Each backdrop image set is defined by a JSON object with a `name`, `key`, and `frames`.

The `key` must be unique across all backdrop image sets, and must only contain lowercase alphanumeric characters and underscores.

`frames` is a list of **relative image paths** corresponding to each frame in the time series. Each set must have **one backdrop image for every frame in the time series**, and they must all be listed in order in the manifest file.

`manifest.json:`

```txt
{
    ...
    "backdrops": [
        {
            "name": <backdrop name>,
            "key": <backdrop key>, // Must be unique across backdrops. Lowercase alphanumeric characters and underscores only.
            "frames": [
                <relative path to backdrop frame 0>,
                <relative path to backdrop frame 1>,
                ...
            ]
        },
        {
            "name": <backdrop name>,
            "key": <backdrop key>,
            ...
        },
        ...
    ] 
}
```

<details>
<summary><b>[Show me an example!]</b></summary>

---

Extending our previous example, we could add two sets of backdrop images. The directory structure would look like this:

```txt
📂 my_dataset/
  - 📄 manifest.json
  - ...
  - 📁 backdrop_brightfield/
    - 📷 img_0.png
    - 📷 img_1.png
    - 📷 img_2.png
    ...
    - 📷 img_245.png
  - 📁 backdrop_h2b/
    - 📷 img_0.png
    - 📷 img_1.png
    - 📷 img_2.png
    ...
    - 📷 img_245.png
```

We would need to add the `backdrops` key to our `manifest.json` file as well:

`manifest.json:`

```txt
{
    "frames": [
        "frames/frame_0.png",
        "frames/frame_1.png",
        "frames/frame_2.png",
        ...
        "frames/frame_245.png",
    ],
    ...
    "backdrops": [
        {
            "name": "Brightfield",
            "key": "brightfield",
            "frames": [
                "backdrop_brightfield/img_0.png",
                "backdrop_brightfield/img_1.png",
                ...
                "backdrop_brightfield/img_245.png",
            ]
        },
        {
            "name": "H2B-GFP",
            "key": "h2b_gfp",
            "frames": [
                "backdrop_h2b/img_0.png",
                "backdrop_h2b/img_1.png",
                ...
                "backdrop_h2b/img_245.png",
            ]
        }
    ]
}
```

---

</details>

### 3. Tracks

Every segmented object in each time step has an **object ID**, an integer identifier that is unique across all time steps. To recognize the same object across multiple frames, these object IDs must be grouped together into a **track** with a single **track number/track ID**.

A **track JSON file** consists of a JSON object with a `data` array, where for each object ID `i`, `data[i]` is the track number that object is assigned to.

`tracks.json:`

```txt
{
    "data": [
        <track number for id 0>,
        <track number for id 1>,
        <track number for id 2>,
        ...
    ]
}
```

<details>
<summary><b>[Show me an example!]</b></summary>

---

For example, if there were the following two tracks in some dataset, the track file might look something like this.

| Track # | Object IDs |
| ------- | ---------- |
| 1       | 0, 1, 4    |
| 2       | 2, 3, 5    |

Note that the object IDs in a track are not guaranteed to be sequential!

`tracks.json:`

```txt
{
    "data": [
        1, // 0
        1, // 1
        2, // 2
        2, // 3
        1, // 4
        2  // 5
    ]
}
```

---

</details>

### 4. Times

The times JSON is similar to the tracks JSON. It also contains a `data` array that maps from object IDs to the frame number that they appear on.

`times.json:`

```txt
{
    "data": [
        <frame number for id 0>,
        <frame number for id 1>,
        <frame number for id 2>,
        ...
    ]
}
```

### 5. Frames

*Example frame:*
![Segmented cell nuclei on a black background, in various shades of green, yellow, red.](./frame_example.png)
*Each unique color in this frame is a different object ID.*

**Frames** are image textures that store the object IDs for each time step in the time series. Each pixel in the image can encode a single object ID in its RGB value (`object ID = R + G*256 + B*256*256 - 1`), and background pixels are `#000000` (black).

Additional notes:

- Encoded object ID's in the frame data start at `1` instead of `0`, because `#000000` (black) is reserved for the background.
- The highest object ID that can currently be represented is `16,843,007`.
  - If the **total number of segmented objects** for an entire time series exceeds this number, it is possible to remove this limit. [Submit an issue](https://github.com/allen-cell-animated/colorizer-data/issues) or send us a message!

There should be one frame for every time step in the time series, and they must all be listed in order in the **manifest** file to be included in the dataset.

<details>
<summary><b>[Show me an example!]</b></summary>

---

Let's say we have a simple 3x3 image, and the center pixel is mapped to the object ID `640` surrounded by the background.

The calculation for the RGB value would follow this process.

1. Add one to the object ID, because of 1-based indexing. (`ID = 641`)
2. Get the Red channel value. (`R = ID % 256 = 641 % 256 = 129`)
3. Get the Green channel value. (`G = ⌊ID / 256⌋ % 256 = 1 % 256 = 2`)
4. Get the Blue channel value. (`B = ⌊ID / (256^2)⌋ = ⌊641 / (256^2)⌋ = 0`)

The RGB value for ID `640` will be `RGB(129, 2, 0)`, or `#810200`.

The resulting frame would look like this:

!["A magnified 3x3 frame with a single red pixel (#810200) in the center, surrounded by black pixels."](./frame_example_simple.png)

---

</details>

### 6. Features

Datasets can contain any number of `features`, which are a numeric value assigned to each object ID in the dataset. Features are used by the Timelapse-Colorizer to colorize objects, and each feature file corresponds to a single column of data. Examples of relevant features might include the volume, depth, number of neighbors, age, etc. of each object.

Features include a `data` array, specifying the feature value for each object ID, and should also provide a `min` and `max` range property. How feature values
should be interpreted can be defined in the `manifest.json` metadata.

For continuous features, decimal and float values will be shown directly, and discrete features will be rounded to the nearest int. For categorical features,
the feature values will be parsed as integers (rounded) and used to index into the `categories` array provided in the `manifest.json`.

`feature1.json:`

```txt
{
    "data": [
        <feature value for id 0>,
        <feature value for id 1>,
        <feature value for id 2>,
        ...
    ],
    "min": <min value for all features>,
    "max": <max value for all features>
}
```

<details>
<summary><b>[Show me an example!]</b></summary>

---

Let's use the "Life Stages" feature example from before, in the manifest. Here's a snippet of the feature metadata in the manifest.

`manifest.json:`

```txt
...,
"features": [
    {
        "key": "life_stage",
        "name": "Life Stage",
        "data": "feature_2.json",
        "type": "categorical",
        "categories": ["G1", "S", "G2", "Prophase", "Metaphase", "Anaphase", "Telophase" ]
    },
    ...
]
...
```

There are 7 categories, so our feature values should be integer indexes ranging from 0 to 6. Let's say our dataset has only one frame, for simplicity, and the following cells are visible:

| Cell # | Life Stage | Index |
| ------ | ---------- | ----- |
| 0      | Metaphase  | 4     |
| 1      | G1         | 0     |
| 2      | Telophase  | 6     |
| 3      | G2         | 2     |

Our feature file should look something like this.

`feature2.json:`

```txt
{
    "data": [
        4,  // Cell #0
        0,  // Cell #1
        6,  // Cell #2
        2   // Cell #3
    ],
    "min": 0,
    "max": 6
}
```

---

</details>

### 7. Centroids (optional)

The centroids file defines the center of each object ID in the dataset. It follows the same format as the feature file, but each ID has two entries corresponding to the `x` and `y` coordinates of the object's centroid, making the `data` array twice as long.

For each index `i`, the coordinates are `(x: data[2i], y: data[2i + 1])`.
Coordinates are defined in pixels in the frame, where the upper left corner of the frame is (0, 0).

`centroids.json:`

```txt
{
    "data": [
        // <x coordinate for id 0>,
        // <y coordinate for id 0>,
        // <x coordinate for id 1>,
        // <y coordinate for id 1>,
        // ...
    ]
}
```

### 8. Bounds (optional)

The bounds file defines the rectangular boundary occupied by each object ID. Like centroids and features, the file defines a `data` array, but has four entries for each object ID to represent the `x` and `y` coordinates of the upper left and lower right corners of the bounding box.

For each object ID `i`, the minimum bounding box coordinates (upper left corner) are given by `(x: data[4i], y: data[4i + 1])`, and the maximum bounding box coordinates (lower right corner) are given by `(x: data[4i + 2], y: data[4i + 3])`.

Again, coordinates are defined in pixels in the image frame, where the upper left corner is (0, 0).

`bounds.json:`

```txt
{
    "data": [
        <upper left x for id 0>,
        <upper left y for id 0>,
        <lower right x for id 0>,
        <lower right y for id 0>,
        <upper left x for id 1>,
        <upper left y for id 1>,
        ...
    ]
}
```

### 9. Outliers (optional)

The outliers file stores whether a given object ID should be marked as an outlier using an array of booleans (`true`/`false`). Indices that are `true` indicate outlier values, and are given a unique color in Timelapse-Colorizer.

`outliers.json:`

```txt
{
    "data": [
        <whether id 0 is an outlier>,
        <whether id 1 is an outlier>,
        <whether id 2 is an outlier>,
        ...
    ]
}
```

<details>
<summary><b>[Show me an example!]</b></summary>

---

For example, if a dataset had the following tracks and outliers, the file might look something like this.

| Track # | Object IDs | Outliers |
| ------- | ---------- | -------- |
| 1       | 0, 1, 4    | 1        |
| 2       | 2, 3, 5    | 2, 5     |

`outliers.json`

```txt
{
    "data": [
        false, // 0
        true,  // 1
        true,  // 2
        false, // 3
        false, // 4
        true   // 5
    ]
}
```

---

</details>

## Collections

Collections are defined by an optional JSON file and group one or more datasets for easy access. Timelapse-Colorizer can parse collection files and present its datasets for easier comparison and analysis from the UI.

By default, collection files should be named `collection.json`.

Collections are an array of JSON objects, each of which define the `name` (an **alias**) and the `path` of a dataset. This can either be a relative path from the location of the collection file, or a complete URL.

If the path does not define a `.json` file specifically, Timelapse-Colorizer will assume that the dataset's manifest is named `manifest.json` by default.

`collection.json:`

```txt
[
    { "name": <some_name_1>, "path": <some_path_1>},
    { "name": <some_name_2>, "path": <some_path_2>},
    ...
]
```

<details>
<summary><b>[Show me an example!]</b></summary>

---

For example, let's say a collection is located at `https://example.com/data/collection.json`, and the `collection.json` contains this:

```txt
[
  { "name": "Mama Bear", "path": "mama_bear" },
  { "name": "Baby Bear", "path": "nested/baby_bear" },
  { "name": "Babiest Bear", "path": "babiest_bear/dataset.json" },
  { "name": "Goldilocks", "path": "https://example2.com/files/goldilocks" },
  { "name": "Papa Bear", "path": "https://example3.com/files/papa_bear.json"}
]
```

Here's a list of where Timelapse-Colorizer will check for the manifest files for all of the datasets:

| Dataset      | Expected URL Path                                         |
| ------------ | --------------------------------------------------------- |
| Mama Bear    | `https://example.com/data/mama_bear/manifest.json`        |
| Baby Bear    | `https://example.com/data/nested/baby_bear/manifest.json` |
| Babiest Bear | `https://example.com/data/babiest_bear/dataset.json`      |
| Goldilocks   | `https://example2.com/files/goldilocks/manifest.json`     |
| Papa Bear    | `https://example3.com/files/papa_bear.json`               |

---

</details>
