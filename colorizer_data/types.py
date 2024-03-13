import dataclasses
from dataclasses import dataclass
from dataclasses_json import dataclass_json, LetterCase
from enum import Enum
from typing import List, TypedDict, Union

CURRENT_VERSION = "v1.0.0"
# TODO: Add colon to z
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


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
    """JSON dictionary format for `BaseMetadata`."""

    name: str
    description: str
    author: str
    dateCreated: str
    lastModified: str
    revision: str
    dataVersion: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ColorizerMetadata:
    """
    Data class representation of metadata for a Colorizer dataset.
    Can be converted to and from camelCase JSON format; see https://pypi.org/project/dataclasses-json/.
    """

    name: str = None
    description: str = None
    author: str = None
    date_created: str = None
    """Formatted datetime string. See `DATETIME_FORMAT`. """
    last_modified: str = None
    """Formatted datetime string. See `DATETIME_FORMAT`. """
    revision: int = None
    """
    Revision number. Will be updated each time the dataset or collection
    is rewritten. Starts at 0.
    """
    writer_version: str = CURRENT_VERSION
    """Version of the data writer utility scripts. Uses semantic versioning (e.g. v1.0.0)"""

    frame_width: float = 0
    frame_height: float = 0
    frame_units: str = ""
    frame_duration_sec: float = 0
    start_time_sec: float = 0
    start_frame_num: int = 0


class DatasetManifest(TypedDict):
    features: List[FeatureMetadata]
    outliers: str
    tracks: str
    centroids: str
    times: str
    bounds: str
    metadata: ColorizerMetadata
    frames: List[str]
    backdrops: List[BackdropMetadata]


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class CollectionMetadata:
    """
    Data class representation of metadata for a Colorizer collection file.
    Can be converted to and from camelCase JSON format; see https://pypi.org/project/dataclasses-json/.
    """

    name: str = None
    description: str = None
    author: str = None
    date_created: str = None
    """Formatted datetime string. See `DATETIME_FORMAT`. """
    last_modified: str = None
    """Formatted datetime string. See `DATETIME_FORMAT`. """
    revision: int = None
    """
    Revision number. Will be updated each time the dataset or collection
    is rewritten. Starts at 0.
    """
    writer_version: str = CURRENT_VERSION
    """Version of the data writer utility scripts. Uses semantic versioning (e.g. v1.0.0)"""


class CollectionDatasetEntry(TypedDict):
    """Represents a single dataset in the collection file."""

    name: str
    path: str


class CollectionManifest(TypedDict):
    """Collection manifest file format."""

    datasets: List[CollectionDatasetEntry]
    metadata: CollectionMetadata
