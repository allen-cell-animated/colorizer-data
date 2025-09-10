# Timelapse Feature Explorer - Data Format

Last release: v1.6.4

**NOTE:** If you are looking to create a dataset, follow our [getting started guide (`GETTING_STARTED.ipynb`)](./getting_started_guide/GETTING_STARTED.ipynb), and see the [readme (`README.md`)](../README.md) for more details on how to install this package.

This document describes the dataset format used by Timelapse Feature Explorer. Utilities in `colorizer_data` automatically write datasets in this format, but this document exists as a technical resource users who want to create or edit their own datasets manually.

## 1. Terms

A few important terms:

- **Dataset**: A dataset is a single time-series, and can have any number of tracked objects and features.
- **Collection**: A grouping of datasets.
- **Segmentation ID**: An integer identifier for a segmented object at a single time step, usually as the output of a segmentation algorithm. Segmentation IDs are unique within a single time step, but are not guaranteed to be unique across time steps.
- **Global/object ID**: An integer identifier for each segmented object which is unique across ALL time steps. A tracked object will have a different object ID at each time step.

## 2. Dataset

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
            "key": <feature key>,                                 // See note on keys below.
            "name": <feature display name>,
            "data": <relative path to feature file>,
            // Optional fields:
            "unit": <unit label>,
            "type": <"continuous" | "discrete" | "categorical">,
            "categories": [<category 1>, <category 2>, ...,]      //< required if type "categorical"; max 12 categories
            "min": <min value for feature>,
            "max": <max value for feature>,
            "description": <feature description>,
        }
        {
            "name": <feature display name>,
            ...
        },
        ...
    ],
    "tracks": <relative path to tracks file>,
    "times": <relative path to times file>,
    "segIds": <relative path to segmentation IDs file>, //< optional, for 3D datasets
    "outliers": <relative path to outlier file>,    //< optional
    "centroids": <relative path to centroids file>, //< optional
    "bounds": <relative path to bounds file>        //< optional
    "backdrops": <array of backdrop image sets>     //< optional, see 2. Backdrops for more details
}
```

_Note: all paths are relative to the location of the manifest file._

A complete example dataset is also available in the [`documentation`](./example_dataset) directory of this project, and can be [viewed on Timelapse Feature Explorer](https://timelapse.allencell.org/viewer?dataset=https://raw.githubusercontent.com/allen-cell-animated/colorizer-data/main/documentation/example_dataset/manifest.json).

### File formats

Linked data files (e.g. `tracks`, `times`, `segIds`, `outliers`, `centroids`, `bounds`, features, etc.) should either be a `.json` file or `.parquet` file. JSON files should be an object with a `data` array property. `.parquet` files should contain a single column.

Note that the `outliers`, `centroids`, and `bounds` files are optional, but certain features of Timelapse Feature Explorer won't work without them.

**Features** can also define additional optional metadata, such as the units and type. Note that there are additional restrictions on some of these fields. **`type`** must have values `continuous` for floats or decimals, `discrete` for integers, or `categorical` for distinct labels.

Features that have the `categorical` type must also define an array of string `categories`, up to a maximum of 12.

### Note on keys

Several fields in the manifest file have a `key` property. These keys must be unique and contain only lowercase **alphanumeric characters and underscores**. For example, `my_feature_1` is a valid key, but `My Feature 1` is not.

<details>
<summary><b>[🔍 Show me an example!]</b></summary>

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
            "key": "cycle_stage",
            "name": "Cell Cycle Stage",
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

### 2.1. Metadata

Manifests can include some optional **metadata** about the dataset and its features.

Besides the details shown above, these are additional parameters that the manifest can include:

`manifest.json:`

```txt
{
    ...
    "metadata": {
        "name": <name of dataset>,
        "description": <description text>,
        "author": <string author name>,
        "dateCreated": <datestring>,
        "lastModified": <datestring>,
        "revision": <number of times the datset has been modified>,
        "dataVersion": <version number of the data scripts used to write this dataset>
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

These metadata parameters are used to configure additional features of the Timelapse Feature Explorer UI, such as showing scale bars or timestamps on the main display. Additional metadata fields will likely be added over time.

Note that the interface will directly show the unit labels and does not scale or convert units from one type to another (for example, it will not convert 1000 µm to 1 mm). If you need to present your data with different units, create a (scaled) duplicate of the feature with a different unit label.

If using the provided writer utility scripts, the `revision`, `dataVersion`, `dateCreated`, and `lastModified` fields will be automatically written and updated.

<details>
<summary><b>[🔍 Show me an example!]</b></summary>

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

### 2.2. Backdrops (optional)

Multiple sets of **backdrop images** can be included in the manifest, which will be shown behind the colored objects in the UI. Each backdrop image set is defined by a JSON object with a `name`, `key`, and `frames`.

The `key` must be unique across all backdrop image sets, and must only contain lowercase alphanumeric characters and underscores. (See [note in 1. Dataset](#note-on-keys) for more details.)

`frames` is a list of **relative image paths** corresponding to each frame in the time series. Each set must have **one backdrop image for every frame in the time series**, and they must all be listed in order in the manifest file.

`manifest.json:`

```txt
{
    ...
    "backdrops": [
        {
            "name": <backdrop name>,
            "key": <backdrop key>,
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
<summary><b>[🔍 Show me an example!]</b></summary>

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

### 2.3. Tracks

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
<summary><b>[🔍 Show me an example!]</b></summary>

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

### 2.4. Times

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

### 2.5. 2D Frames and Segmentation IDs

_Example frame:_
![Segmented cell nuclei on a black background, in various shades of green, yellow, red.](./frame_example.png)
_Each unique color in this frame is a different object ID._

**Frames** are image textures that store the object IDs for each time step in the time series. Each pixel in the image can encode a single object ID in its RGB value (`object ID = R + G*256 + B*256*256 - 1`), and background pixels are `#000000` (black).

Additional notes:

- Encoded object ID's in the frame data start at `1` instead of `0`, because `#000000` (black) is reserved for the background.
- The highest object ID that can currently be represented is `16,843,007`.
  - If the **total number of segmented objects** for an entire time series exceeds this number, it is possible to remove this limit. [Submit an issue](https://github.com/allen-cell-animated/colorizer-data/issues) or send us a message!

There should be one frame for every time step in the time series, and they must all be listed in order in the **manifest** file to be included in the dataset.

<details>
<summary><b>[🔍 Show me an example!]</b></summary>

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

#### Segmentation IDs (optional)

It's also possible to load frames where segmentation labels are not unique across all time steps, by providing a `segIds` file. This is typically used for very large image files like 3D OME-Zarr data (see section on 3D frames).

For each object ID `i`, the `segIds[i]` is the segmentation ID (e.g. "label" or "raw pixel value") of that object in the frame data at some time `t`.

### 2.6. Features

Datasets can contain any number of `features`, which are a numeric value assigned to each object ID in the dataset. Features are used by the Timelapse Feature Explorer to colorize objects, and each feature file corresponds to a single column of data. Examples of relevant features might include the volume, depth, number of neighbors, age, etc. of each object.

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
<summary><b>[🔍 Show me an example!]</b></summary>

---

Let's use the "Cell Cycle Stage" feature example from before, in the manifest. Here's a snippet of the feature metadata in the manifest.

`manifest.json:`

```txt
...,
"features": [
    {
        "key": "cycle_stage",
        "name": "Cell Cycle Stage",
        "data": "feature_2.json",
        "type": "categorical",
        "categories": ["G1", "S", "G2", "Prophase", "Metaphase", "Anaphase", "Telophase" ],

    },
    ...
]
...
```

There are 7 categories, so our feature values should be integer indexes ranging from 0 to 6. Let's say our dataset has only one frame, for simplicity, and the following cells are visible:

| Cell # | Cell Cycle Stage | Index |
| ------ | ---------------- | ----- |
| 0      | Metaphase        | 4     |
| 1      | G1               | 0     |
| 2      | Telophase        | 6     |
| 3      | G2               | 2     |

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

### 2.7. Centroids (optional)

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

### 2.8. Bounds (optional)

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

### 2.9. Outliers (optional)

The outliers file stores whether a given object ID should be marked as an outlier using an array of booleans (`true`/`false`). Indices that are `true` indicate outlier values, and are given a unique color in Timelapse Feature Explorer.

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
<summary><b>[🔍 Show me an example!]</b></summary>

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

### 2.10. 3D Frames (experimental)

Timelapse Feature Explorer has experimental support for viewing and interacting with 3D segmentation data.

For best performance, 3D segmentation sources should be stored as a multiscale [OME-Zarr file](https://link.springer.com/article/10.1007/s00418-023-02209-1) ending in `.ome.zarr`. It should be a time-series Zarr that directly stores the integer segmentation IDs (if the `segIds` file is provided) or the global object IDs.

To do so, include the `frames3d` key in the manifest file, which will replace the `frames` and `backdrops` parameters.

`manifest.json:`

```txt
{
    "frames3d": {
        "source": <relative path or URL of an OME-Zarr file>,
        "segmentationChannel": <channel index for segmentation IDs>, // optional, defaults to 0
        "totalFrames": <total number of frames in the time series>,
        "backdrops": [...] <array of backdrop parameters>
    },
}
```

If you want to include additional channels as backdrop images, you can specify them in the `backdrops` array. Each backdrop object should have the following format:

```text
{
    "source": <relative path or URL of an OME-Zarr file>, // can be the same as the main source
    "name": <name of the backdrop>,
    "description": <description of the backdrop>, // optional
    "channel": <channel index for the backdrop>,  // optional, defaults to 0
    "min": <min value for transfer function>, // optional
    "max": <max value for transfer function>, // optional
}
```

Multiple backdrops sharing the same source can be included by specifying different `channel` indices.

If a min and max value is provided, the backdrop channel will be displayed using a transfer function that maps from raw data values to an intensity value. Values at `min` will be mapped to 0, and values at `max` will be mapped to 1, and values between will be ramped linearly. Values outside this range will be clamped to the nearest value.

If `centroids` data are provided, they will be assumed to be in voxel coordinates.

## 3. Collections

Collections are defined by an optional JSON file and group one or more datasets together. Timelapse Feature Explorer can parse collection files and present its datasets for easier comparison and analysis from the UI.

By default, collection files should be named `collection.json`.

`collection.json:`

```txt
{
    "datasets": [
        { "name": <some_name_1>, "path": <some_path_1>},
        { "name": <some_name_2>, "path": <some_path_2>},
        ...
    ],
    "metadata": {
        ...
    }
}
```

_Note: The legacy collection format was a JSON array instead of a JSON object. Backwards-compatibility is preserved in the viewer, but the JSON array format is considered deprecated._

### 3.1. Defining datasets in collections

Collections contain an array of dataset objects, each of which define the `name` (an **alias**) and the `path` of a dataset. This can either be a relative path from the location of the collection file, or a complete URL.

If the path does not define a `.json` file specifically, Timelapse Feature Explorer will assume that the dataset's manifest is named `manifest.json` by default.

<details>
<summary><b>[🔍 Show me an example!]</b></summary>

---

For example, let's say a collection is located at `https://example.com/data/collection.json`, and the `collection.json` contains this:

```txt
{
    "datasets": [
        { "name": "Mama Bear", "path": "mama_bear" },
        { "name": "Baby Bear", "path": "nested/baby_bear" },
        { "name": "Babiest Bear", "path": "babiest_bear/dataset.json" },
        { "name": "Goldilocks", "path": "https://example2.com/files/goldilocks" },
        { "name": "Papa Bear", "path": "https://example3.com/files/papa_bear.json"}
    ]
}
```

Here's a list of where Timelapse Feature Explorer will check for the manifest files for all of the datasets:

| Dataset      | Expected URL Path                                         |
| ------------ | --------------------------------------------------------- |
| Mama Bear    | `https://example.com/data/mama_bear/manifest.json`        |
| Baby Bear    | `https://example.com/data/nested/baby_bear/manifest.json` |
| Babiest Bear | `https://example.com/data/babiest_bear/dataset.json`      |
| Goldilocks   | `https://example2.com/files/goldilocks/manifest.json`     |
| Papa Bear    | `https://example3.com/files/papa_bear.json`               |

---

</details>

### 3.2. Collection metadata

A collection file can also include optional metadata fields, saved under the `metadata` key.

`collection.json`

```txt
{
    ...
    "metadata": {
        "name": <name of collection>,
        "description": <description text>,
        "author": <string author name>,
        "dateCreated": <datestring>,
        "lastModified": <datestring>,
        "revision": <number of times the collection has been modified>,
        "dataVersion": <version of the data scripts used to write this collection>
    }
}
```

## 4. FAQ

### My data needs to start at a timepoint other than zero

Once you get the first frame in your dataset, you'll need to save the `starting_timepoint` to your dataset's metadata and include the information when generating the frame paths.

```python
writer = ColorizerDatasetWriter(...)
starting_timepoint = 5
num_frames = 100

# Generate file paths starting at some offset
frame_paths = generate_frame_paths(num_frames, start_frame=starting_timepoint)
writer.set_frame_paths(frame_paths)

# Including starting frame number in the metadata
metadata = ColorizerMetadata(start_frame_num=starting_timepoint)
writer.write_manifest(metadata)
```
