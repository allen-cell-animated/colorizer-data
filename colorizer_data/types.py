from dataclasses import dataclass
import dataclasses
from enum import Enum
from typing import List, TypedDict, Union


class FeatureType(str, Enum):
    CONTINUOUS = "continuous"
    """For continuous decimal values."""
    DISCRETE = "discrete"
    """For integer-only values."""
    CATEGORICAL = "categorical"
    """For category labels. Can include up to 12 string categories, stored in the feature's
    `categories` field. The feature data value for each object ID should be the
    integer index of the name in the `categories` field.
    """
    INDETERMINATE = "indeterminate"
    """An unknown or indeterminate feature type; the default for FeatureInfo.
    The writer will attempt to detect indeterminate feature types in provided feature data
    and cast it to a continuous, discrete, or categorical value.
    """


@dataclass
class FeatureInfo:
    """Represents a feature's metadata.

    Args:
        - `label` (`str`): The human-readable name of the feature. Empty string (`""`) by default.
        - `column_name` (`str`): The column name, in the dataset, of the feature. Used for the feature's name
        if no label is provided. Empty string (`""`) by default.
        - `key` (`str`): The internal key name of the feature. Formats the feature label if no
        key is provided. Empty string (`""`) by default.
        - `unit` (`str`): Units this feature is measured in. Empty string (`""`) by default.
        - `type` (`FeatureType`): The type, either continuous, discrete, or categorical, of this feature.
        `FeatureType.INDETERMINATE` by default.
        - `categories` (`List`[str]): The ordered categories for categorical features. `None` by default.
    """

    label: str = ""
    key: str = ""
    column_name: str = ""
    unit: str = ""
    type: FeatureType = FeatureType.INDETERMINATE
    categories: Union[List[str], None] = None

    def get_name(self) -> str:
        if self.label != "":
            return self.label
        if self.key != "":
            return self.key
        if self.column_name != "":
            return self.column_name
        return "N/A"

    # TODO: Use Self return type
    def clone(self):
        new_info = dataclasses.replace(self)
        new_info.categories = self.categories.copy()
        return new_info


class FeatureMetadata(TypedDict):
    """For data writer internal use. Represents the metadata that will be saved for each feature."""

    data: str
    key: str
    name: str
    """The relative path from the manifest to the feature JSON file."""
    unit: str
    type: FeatureType
    categories: List[str]


class BackdropMetadata(TypedDict):
    frames: List[str]
    name: str
    key: str


class FrameDimensions(TypedDict):
    """Dimensions of each frame, in physical units (not pixels)."""

    units: str
    width: float
    """Width of a frame in physical units (not pixels)."""
    height: float
    """Height of a frame in physical units (not pixels)."""


class DatasetMetadata(TypedDict):
    frameDims: FrameDimensions
    frameDurationSeconds: float
    startTimeSeconds: float


@dataclass
class ColorizerMetadata:
    """Data class representation of metadata for a Colorizer dataset."""

    frame_width: float = 0
    frame_height: float = 0
    frame_units: str = ""
    frame_duration_sec: float = 0
    start_time_sec: float = 0
    start_frame_num: int = 0

    def to_json(self) -> DatasetMetadata:
        return {
            "frameDims": {
                "width": self.frame_width,
                "height": self.frame_height,
                "units": self.frame_units,
            },
            "startTimeSeconds": self.start_time_sec,
            "frameDurationSeconds": self.frame_duration_sec,
            "startingFrameNumber": self.start_frame_num,
        }


class DatasetManifest(TypedDict):
    features: List[FeatureMetadata]
    outliers: str
    tracks: str
    centroids: str
    times: str
    bounds: str
    metadata: DatasetMetadata
    frames: List[str]
