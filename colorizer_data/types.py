import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import List, TypedDict, Union

CURRENT_VERSION = "v1.0.0"


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
    """
    Represents a feature's metadata.

    Args:
        `label`: The human-readable name of the feature. Empty string (`""`) by default.
        `column_name`: The column name, in the dataset, of the feature. Used for the feature's name
        if no label is provided. Empty string (`""`) by default.
        `key`: The internal key name of the feature. Formats the feature label if no
        key is provided. Empty string (`""`) by default.
        `unit`: Units this feature is measured in. Empty string (`""`) by default.
        `type`: The type, either continuous, discrete, or categorical, of this feature.
        `FeatureType.INDETERMINATE` by default.
        `categories`: The ordered categories for categorical features. `None` by default.
    """

    label: str = ""
    key: str = ""
    column_name: str = ""
    unit: str = ""
    type: FeatureType = FeatureType.INDETERMINATE
    categories: Union[List[str], None] = None

    def get_name(self) -> Union[str, None]:
        """
        Gets the name of the feature, returning the first non-empty string from `label`, `key`, or
        `column_name` in that order. Returns None if all fields are empty strings.
        """
        if self.label != "":
            return self.label
        if self.key != "":
            return self.key
        if self.column_name != "":
            return self.column_name
        return None

    # TODO: Use Self return type here if we support Python 3.11
    def clone(self):
        """
        Returns a copy of this FeatureInfo instance.
        """
        new_info = dataclasses.replace(self)
        if self.categories:
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


class BaseMetadataJson(TypedDict):
    """Shared metadata, in JSON form."""

    name: str
    description: str
    dateCreated: str
    """Datetime, formatted as `%m/%d/%Y, %H:%M:%S`"""
    lastModified: str
    author: str
    revision: str
    dataVersion: str


@dataclass
class BaseMetadata:
    """Shared metadata between datasets and collection files."""

    name: str = ""
    description: str = ""
    date_created: str = ""
    last_modified: str = ""
    author: str = ""
    revision: str = ""
    data_version: str = CURRENT_VERSION

    def to_json(self) -> BaseMetadataJson:
        return {
            "name": self.name,
            "description": self.description,
            "dateCreated": self.date_created,
            "lastModified": self.last_modified,
            "author": self.author,
            "revision": self.revision,
            "dataVersion": self.data_version,
        }


# TODO: Rename this and ColorizerMetadata.
class DatasetMetadata(BaseMetadataJson):
    """JSON-exported metadata for the dataset"""

    frameDims: FrameDimensions
    frameDurationSeconds: float
    startTimeSeconds: float


@dataclass
class ColorizerMetadata(BaseMetadata):
    """Data class representation of metadata for a Colorizer dataset. Can be
    converted to JSON-compatible format using `to_json()`."""

    frame_width: float = 0
    frame_height: float = 0
    frame_units: str = ""
    frame_duration_sec: float = 0
    start_time_sec: float = 0
    start_frame_num: int = 0

    def to_json(self) -> DatasetMetadata:
        base_json = super(self)
        base_json["frameDims"] = {
            "width": self.frame_width,
            "height": self.frame_height,
            "units": self.frame_units,
        }
        base_json["startTimeSeconds"] = self.start_time_sec
        base_json["frameDurationSeconds"] = self.frame_duration_sec
        base_json["startingFrameNumber"] = self.start_frame_num
        return base_json


class DatasetManifest(TypedDict):
    features: List[FeatureMetadata]
    outliers: str
    tracks: str
    centroids: str
    times: str
    bounds: str
    metadata: DatasetMetadata
    frames: List[str]


class CollectionDatasetEntry(TypedDict):
    name: str
    path: str


class CollectionMetadataJson(BaseMetadataJson):
    datasets: List[CollectionDatasetEntry]


@dataclass
class CollectionMetadata(BaseMetadata):
    datasets: List[CollectionDatasetEntry]

    def to_json(self) -> BaseMetadataJson:
        base_json = BaseMetadata.to_json(self)
        base_json["datasets"] = self.datasets
        return base_json


class CollectionManifest(TypedDict):
    """Collection manifest JSON file format."""

    datasets: List[CollectionDatasetEntry]
    metadata: CollectionMetadata
