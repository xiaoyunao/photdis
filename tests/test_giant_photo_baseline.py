from giant_photo_baseline import field_name, specdis_crosscheck


def test_fits_field_string_representation():
    assert field_name(b"ra_field  ") == "ra_field"
    assert field_name("ra_field  ") == "ra_field"


def test_specdis_comparison_uses_paired_stars_only():
    values = [(1, "f", 0, .1, 1., .5, -1, -.5, 1.5, 2, 3, 20, -1, True),
              (2, "f", 0, .1, None, None, -1, -.5, 9., 2, 3, 20, -1, True)]
    result = specdis_crosscheck(values)
    assert result["n"] == 1
    assert result["median_pred_minus_specdis_mg"] == .5
