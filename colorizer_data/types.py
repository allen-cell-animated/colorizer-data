import dataclasses
from dataclasses import dataclass, field
from dataclasses_json import LetterCase, DataClassJsonMixin, config
from dataclasses_json.core import _decode_dataclass
from enum import Enum
from typing import Dict, List, Optional, Type, TypeVar, TypedDict, Union

Json = Union[dict, str, int, float, bool, None]


CURRENT_VERSION = "v1.5.2"
DEFAULT_COLLECTION_VERSION = "v1.0"
DEFAULT_DATASET_VERSION = "v1.0"
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
"""
Note: time MUST be in UTC!
Use `datetime.now(timezone.utc).strftime(DATETIME_FORMAT)`.
"""


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
        `min`: The minimum value for continuous or discrete features. `None` by default.
        `max`: The maximum value for continuous or discrete features. `None` by default.
        `description`: A description of the feature. Empty string (`""`) by default.
    """

    label: str = ""
    key: str = ""
    column_name: str = ""
    unit: str = ""
    type: FeatureType = FeatureType.INDETERMINATE
    categories: Optional[List[str]] = None
    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None
    description: str = ""

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
    """The relative path from the manifest to the feature JSON file."""
    key: str
    name: str
    unit: str
    type: FeatureType
    categories: List[str]
    min: Union[int, float]
    max: Union[int, float]
    description: str


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


@dataclass
class ColorizerMetadata(DataClassJsonMixin):
    """
    Data class representation of metadata for a Colorizer dataset.
    Can be converted to and from camelCase JSON format; see https://pypi.org/project/dataclasses-json/.
    """

    dataclass_json_config = config(letter_case=LetterCase.CAMEL, undefined=None)[
        "dataclasses_json"
    ]

    name: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    dataset_version: Optional[str] = None
    """User-defined dataset version."""
    date_created: Optional[str] = None
    """ISO-formatted datetime string in UTC. See `DATETIME_FORMAT`."""
    last_modified: Optional[str] = None
    """ISO-formatted datetime string in UTC. See `DATETIME_FORMAT`."""

    # Internal use
    _revision: Optional[int] = None
    """
    Revision number. Will be updated each time the dataset or collection
    is rewritten. Starts at 0.
    """
    _writer_version: Optional[str] = CURRENT_VERSION
    """Version of the data writer utility scripts. Uses semantic versioning (e.g. v1.5.2)"""

    # Exclude these three fields from auto-encode/decode, because they need to be structured
    # together under the frameDims subfield and not as their own root-level fields.
    frame_width: Optional[float] = field(
        default=None, metadata=config(exclude=lambda x: True)
    )
    frame_height: Optional[float] = field(
        default=None, metadata=config(exclude=lambda x: True)
    )
    frame_units: Optional[str] = field(
        default=None, metadata=config(exclude=lambda x: True)
    )

    # Exclude in order to rename when saving
    frame_duration_sec: float = field(
        default=0, metadata=config(exclude=lambda x: True)
    )
    start_time_sec: float = field(default=0, metadata=config(exclude=lambda x: True))
    start_frame_num: int = field(default=0, metadata=config(exclude=lambda x: True))

    # Override to and from dict behaviors to allow nesting of frame-related variables in their own
    # dictionary object.

    def to_dict(self, encode_json=False) -> Dict[str, Json]:
        base_json = DataClassJsonMixin.to_dict(self)
        base_json["frameDims"] = {
            "width": self.frame_width,
            "height": self.frame_height,
            "units": self.frame_units,
        }
        base_json["startingTimeSeconds"] = self.start_time_sec
        base_json["startingFrameNumber"] = self.start_frame_num
        base_json["frameDurationSeconds"] = self.frame_duration_sec
        return base_json

    A = TypeVar("A", bound="DataClassJsonMixin")

    @classmethod
    def from_dict(
        cls: Type[A],
        kvs: Union[dict, list, str, int, float, bool, None],
        *,
        infer_missing=True,
    ) -> A:
        # Hacky. This is what DataClassJsonMixin.from_dict() calls internally, passing in the
        # inferred class. In this case, we want to explicitly pass in this class (ColorizerMetadata)
        # and use the parent behavior, but we can't call DataClassJsonMixin.from_dict() directly
        # because it is unaware of ColorizerMetadata's dataclass fields.
        metadata: ColorizerMetadata = _decode_dataclass(
            ColorizerMetadata, kvs, infer_missing
        )

        if "frameDims" in kvs.keys() and isinstance(kvs["frameDims"], dict):
            if "width" in kvs["frameDims"].keys():
                metadata.frame_width = kvs["frameDims"]["width"]
            if "height" in kvs["frameDims"].keys():
                metadata.frame_height = kvs["frameDims"]["height"]
            if "units" in kvs["frameDims"].keys():
                metadata.frame_units = kvs["frameDims"]["units"]

        # Handle rename
        if "startingTimeSeconds" in kvs.keys():
            metadata.start_time_sec = kvs["startingTimeSeconds"]
        if "startingFrameNumber" in kvs.keys():
            metadata.start_frame_num = kvs["startingFrameNumber"]
        if "frameDurationSeconds" in kvs.keys():
            metadata.frame_duration_sec = kvs["frameDurationSeconds"]

        return metadata


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


@dataclass
class CollectionMetadata(DataClassJsonMixin):
    """
    Data class representation of metadata for a Colorizer collection file.
    Can be converted to and from camelCase JSON format; see https://pypi.org/project/dataclasses-json/.
    """

    dataclass_json_config = config(letter_case=LetterCase.CAMEL, undefined=None)[
        "dataclasses_json"
    ]

    name: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    collection_version: Optional[str] = None
    """User-defined collection version."""
    date_created: Optional[str] = None
    """ISO-formatted datetime string in UTC. See `DATETIME_FORMAT`."""
    last_modified: Optional[str] = None
    """ISO-formatted datetime string in UTC. See `DATETIME_FORMAT`."""

    # Internal use
    _revision: Optional[int] = None
    """
    Revision number. Will be incremented each time the dataset or collection
    is rewritten, starting at 0.
    """
    _writer_version: Optional[str] = CURRENT_VERSION
    """Version of the data writer utility scripts. Uses semantic versioning (e.g. v1.5.2)"""


class CollectionDatasetEntry(TypedDict):
    """Represents a single dataset in the collection file."""

    name: str
    path: str


class CollectionManifest(TypedDict):
    """Collection manifest file format."""

    datasets: List[CollectionDatasetEntry]
    metadata: CollectionMetadata


class DataFileType(Enum):
    JSON = "json"
    PARQUET = "parquet"
