import numpy as np

from photdis.class_evidence import desi_class_evidence, desi_class_evidence_v3


def test_spectral_evidence_keeps_unknown_and_conflicts_separate():
    t = {"desi_match": np.array([1, 1, 1, 0, 1], bool),
         "desi_match_ambiguous": np.array([0, 0, 0, 0, 1], bool),
         "desi_snr": np.array([20., 20., 20., 20., 20.]),
         "desi_quality": np.array([1, 1, 0, 0, 1], bool),
         "desi_quality_reason": np.zeros(5, np.uint16),
         "desi_is_stellar": np.array([1, 1, 0, 0, 1], bool),
         "desi_known_contaminant": np.array([0, 0, 1, 0, 0], bool),
         "desi_teff": np.array([4600., 4600., np.nan, np.nan, 4600.]),
         "desi_logg": np.array([4.6, 2.8, np.nan, np.nan, 2.8]),
         "desi_logg_err": np.array([.1, .1, np.nan, np.nan, .1])}
    e = desi_class_evidence(t)
    assert e["class_evidence"].tolist() == [b"cool_main_sequence", b"cool_giant",
        b"nonstellar", b"unknown", b"unknown"]
    assert e["class_conflicts_cold_dwarf"].tolist() == [False, True, True, False, False]
    assert e["class_evidence_source"].tolist() == [b"DESI_RVS", b"DESI_RVS",
        b"DESI_RR", b"none", b"none"]


def test_v3_separates_general_class_training_abundance_and_raw_redrock():
    t = {"desi_match": np.ones(5, bool),
         "desi_match_ambiguous": np.zeros(5, bool),
         "desi_spectype": np.array([b"STAR", b"STAR", b"STAR", b"STAR", b"QSO"]),
         "desi_quality_reason": np.array([0, 0, 1 << 4, 0,
                                          (1 << 0) | (1 << 1) | (1 << 4)], np.uint16),
         "desi_teff": np.array([4000., 5200., 4600., 4600., np.nan]),
         "desi_teff_err": np.array([30., 30., 30., 30., np.nan]),
         "desi_logg": np.array([2.4, 2.4, 4.6, 4.6, np.nan]),
         "desi_logg_err": np.array([.1, .1, .1, .1, np.nan]),
         "desi_feh": np.array([-.5, -.5, np.nan, -1.8, np.nan]),
         "desi_feh_err": np.array([.1, .1, np.nan, .1, np.nan])}
    e = desi_class_evidence_v3(t)
    assert e["class_evidence"].tolist() == [b"conservative_giant",
        b"conservative_giant", b"conservative_dwarf", b"conservative_dwarf",
        b"nonstellar_unassessed"]
    assert e["class_conflicts_cold_dwarf"].tolist() == [True, True, False, False, False]
    assert e["blue_training_eligible"].tolist() == [False, False, False, True, False]
    assert e["abundance_calibration_valid"].tolist() == [False, True, False, True, False]
    assert e["redrock_nonstellar_evidence"].tolist() == [False, False, False, False, True]
    assert e["redrock_quality_state"][-1] == b"unassessed_no_rr_quality"
