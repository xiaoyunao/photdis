"""Conservative existing class evidence for conditional cold-dwarf exports."""
from __future__ import annotations

import numpy as np


def desi_class_evidence(table: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Keep Redrock and RVS evidence distinct; unknown is an explicit state.

    RVS quality requires a stellar spectrum. A nonstellar Redrock type instead
    uses an unambiguous match, S/N>=10 and valid pipeline/primary quality bits.
    It is a conflict flag, not a calibrated class probability.
    """
    n = len(table["desi_match"])
    matched = np.asarray(table["desi_match"], bool) & ~np.asarray(
        table["desi_match_ambiguous"], bool)
    reason = np.asarray(table["desi_quality_reason"], np.uint16)
    snr = np.asarray(table["desi_snr"], float)
    nonstar = (matched & np.asarray(table["desi_known_contaminant"], bool)
               & np.isfinite(snr) & (snr >= 10)
               & ((reason & ((1 << 1) | (1 << 2) | (1 << 5))) == 0))
    stellar = (matched & np.asarray(table["desi_quality"], bool)
               & np.asarray(table["desi_is_stellar"], bool))
    teff = np.asarray(table["desi_teff"], float)
    logg = np.asarray(table["desi_logg"], float)
    logg_err = np.asarray(table["desi_logg_err"], float)
    cool = stellar & np.isfinite(teff) & (teff >= 4200) & (teff < 5000)
    dwarf = cool & np.isfinite(logg_err) & (logg_err > 0) & (logg - logg_err > 4.2)
    giant = cool & np.isfinite(logg_err) & (logg_err > 0) & (logg + logg_err < 3.2)
    label = np.full(n, b"unknown", dtype="S24")
    source = np.full(n, b"none", dtype="S12")
    quality = np.full(n, b"none", dtype="S28")
    label[stellar] = b"other_star_or_transition"
    source[stellar] = b"DESI_RVS"
    quality[stellar] = b"qualified_parameters"
    label[dwarf] = b"cool_main_sequence"
    label[giant] = b"cool_giant"
    label[nonstar] = b"nonstellar"
    source[nonstar] = b"DESI_RR"
    quality[nonstar] = b"unambiguous_snr10_clean"
    return {"class_evidence": label, "class_evidence_source": source,
            "class_evidence_quality": quality,
            "class_conflicts_cold_dwarf": giant | nonstar,
            "class_evidence_available": stellar | nonstar}
