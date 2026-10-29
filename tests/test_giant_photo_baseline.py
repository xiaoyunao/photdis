from giant_photo_baseline import field_name


def test_fits_field_string_representation():
    assert field_name(b"ra_field  ") == "ra_field"
    assert field_name("ra_field  ") == "ra_field"
