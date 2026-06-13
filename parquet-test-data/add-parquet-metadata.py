"""Copy a Parquet file while adding file-level metadata.

Example:
	python parquet-test-data/add-parquet-metadata.py \
		nodes.parquet nodes.with-metadata.parquet \
		--metadata dataset=nodes \
		--metadata generated_by=script
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pdb

import pyarrow as pa
import pyarrow.parquet as pq


def _parse_metadata_pairs(pairs: list[str]) -> dict[bytes, bytes]:
    metadata: dict[bytes, bytes] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid metadata entry '{pair}'. Use key=value format.")
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid metadata entry '{pair}'. Key cannot be empty.")
        metadata[key.encode("utf-8")] = value.encode("utf-8")
    return metadata


def add_metadata(
    input_path: Path, output_path: Path, metadata_pairs: list[str]
) -> None:
    table = pq.read_table(input_path)

    existing_metadata = dict(table.schema.metadata or {})
    new_metadata = _parse_metadata_pairs(metadata_pairs)
    existing_metadata.update(new_metadata)
    updated_table = table.replace_schema_metadata(existing_metadata)

    field = table.schema.field("is_split_node")
    new_field = field.with_metadata(
        {b"categories": b"False,True", b"min": b"0", b"max": b"1"}
    )
    updated_schema = updated_table.schema.set(
        updated_table.schema.get_field_index("is_split_node"), new_field
    )

    updated_table = updated_table.cast(updated_schema)
    with pq.ParquetWriter(output_path, updated_schema) as writer:
        # Writes first row group
        writer.write_table(updated_table)
        # writer.write_table(updated_table)


def read_metadata(input_path: Path) -> None:
    table = pq.read_table(input_path)
    metadata = table.schema.metadata or {}
    for key, value in metadata.items():
        print(f"{key.decode('utf-8')}: {value.decode('utf-8')}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a Parquet file, add metadata, and save a copy."
    )
    parser.add_argument("input", type=Path, help="Path to source Parquet file.")
    parser.add_argument("output", type=Path, help="Path to output Parquet file.")
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Metadata entry to add. Repeat flag for multiple values.",
    )
    return parser


def main() -> None:
    metadata_entries = [
        "num_seg_channels=1",
        "num_channels=1",
        "c0_source=https://vast-files.int.allencell.org/users/peyton.lee/liberali-data/Liberali_Lightsheet_OMEZarrs/002/Deconv.ome.zarr",
        "c0_name=Custom Channel Name ✨",
        "c0_channel=0",
        # "seg0_source=https://vast-files.int.allencell.org/users/peyton.lee/liberali-data/Liberali_Lightsheet_OMEZarrs/002/Deconv.ome.zarr/labels/cell",
        # ⬇️ test that paths will be resolved relative to the Parquet file location
        "seg0_source=Liberali_Lightsheet_OMEZarrs/002/Deconv.ome.zarr/labels/cell",
        "seg0_name=Cell segmentation",
        "seg0_channel=0",
    ]
    # parser = _build_parser()
    # args = parser.parse_args()
    # add_metadata(args.input, args.output, args.metadata)
    # add_metadata("./nodes.parquet", "./nodes_copy.parquet", metadata_entries)

    src = "./nodes.parquet"
    dst = "S:/aics/users/peyton.lee/liberali-data/nodes_metadata.parquet"
    add_metadata(
        src,
        dst,
        metadata_entries,
    )
    print("Wrote metadata")

    read_metadata(dst)


if __name__ == "__main__":
    main()
