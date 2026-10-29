import numpy as np

from photdis.class_evidence import desi_class_evidence


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
