from colorizer_data.types import ColorizerMetadata


def test_manifest_from_dict_handles_empty_dict():
    manifest_dict = {}
    # Test that this does not crash
    metadata = ColorizerMetadata.from_dict(manifest_dict)


def test_manifest_from_dict_handles_partial_frame_dims():
    # frameDims is a special case. This test checks both that parsing it doesn't
    # crash (!!!) and that values can be safely stored and retrieved.
    frame_dimensions = [
        {"width": 100.0, "height": 50.0, "units": "picometers"},
        {"width": None, "height": 40.0, "units": "meters"},
        {"height": 30.0, "units": None},
        {},
    ]

    for i in range(len(frame_dimensions)):
        frame_dims = frame_dimensions[i]

        manifest_dict = {"frameDims": frame_dims}
        metadata = ColorizerMetadata.from_dict(manifest_dict)

        # Check for default values for nested fields
        if "width" in frame_dims.keys():
            assert metadata.frame_width == frame_dims["width"]
        if "height" in frame_dims.keys():
            assert metadata.frame_height == frame_dims["height"]
        if "units" in frame_dims.keys():
            assert metadata.frame_units == frame_dims["units"]
