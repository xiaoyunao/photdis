"""Conservative existing class evidence for conditional cold-dwarf exports."""
from __future__ import annotations

import numpy as np

from .audit import rvs_metallicity


def desi_class_evidence_v3(table: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """General RVS class, Redrock type and branch eligibility as separate facts.

    The MWS projection contains RR_SPECTYPE but not Redrock ZWARN or
    DELTACHI2. Its QSO/GALAXY type is therefore retained as raw, unassessed
    evidence. RVS parameter quality may survive a missing iron abundance;
    iron validity is evaluated separately below.
    """
    n = len(table["desi_match"])
    matched = np.asarray(table["desi_match"], bool) & ~np.asarray(
        table["desi_match_ambiguous"], bool)
    reason = np.asarray(table["desi_quality_reason"], np.uint16)
    spectype = np.char.strip(np.asarray(table["desi_spectype"]).astype("U"))
    stellar_type = matched & (spectype == "STAR")
    teff = np.asarray(table["desi_teff"], float)
    teff_err = np.asarray(table["desi_teff_err"], float)
    logg = np.asarray(table["desi_logg"], float)
    logg_err = np.asarray(table["desi_logg_err"], float)
    parameter_values = (np.isfinite(teff) & (teff > 0)
                        & np.isfinite(teff_err) & (teff_err > 0)
                        & np.isfinite(logg) & np.isfinite(logg_err)
                        & (logg_err > 0))
    # Bit 4 combines invalid Teff/logg/FeH. Recheck Teff/logg explicitly,
    # while all other RVS quality bits remain mandatory for physical class.
    other_failures = reason & np.uint16(~(1 << 4) & 0xffff)
    rvs_parameters = stellar_type & (other_failures == 0) & parameter_values
    cool_parameters = rvs_parameters & (teff < 7000)
    dwarf = cool_parameters & (logg - logg_err > 4.2)
    giant = cool_parameters & (logg + logg_err < 3.2)
    blue_core = dwarf & (teff >= 4200) & (teff < 5000)

    feh = np.asarray(table["desi_feh"], float)
    feh_err = np.asarray(table["desi_feh_err"], float)
    _, abundance_domain = rvs_metallicity(feh, teff, logg)
    abundance_valid = (rvs_parameters & abundance_domain
                       & np.isfinite(feh_err) & (feh_err > 0)
                       & ((reason & (1 << 4)) == 0))

    rr_nonstellar = matched & np.isin(spectype, ["QSO", "GALAXY"])
    rr_type = np.full(n, b"none", dtype="S8")
    rr_type[matched & (spectype != "")] = spectype[matched & (spectype != "")].astype("S8")
    rr_quality = np.full(n, b"none", dtype="S28")
    rr_quality[rr_nonstellar] = b"unassessed_no_rr_quality"
    label = np.full(n, b"unknown", dtype="S28")
    source = np.full(n, b"none", dtype="S12")
    quality = np.full(n, b"none", dtype="S28")
    label[rvs_parameters] = b"other_star_or_transition"
    label[dwarf] = b"conservative_dwarf"
    label[giant] = b"conservative_giant"
    source[rvs_parameters] = b"DESI_RVS"
    quality[rvs_parameters] = b"qualified_parameters"
    label[rr_nonstellar & ~rvs_parameters] = b"nonstellar_unassessed"
    source[rr_nonstellar & ~rvs_parameters] = b"DESI_RR"
    quality[rr_nonstellar & ~rvs_parameters] = b"unassessed_no_rr_quality"
    return {
        "class_evidence": label,
        "class_evidence_source": source,
        "class_evidence_quality": quality,
        "class_evidence_available": rvs_parameters | rr_nonstellar,
        "class_conflicts_cold_dwarf": giant,
        "rvs_parameter_quality": rvs_parameters,
        "blue_training_eligible": blue_core & abundance_valid,
        "abundance_calibration_valid": abundance_valid,
        "redrock_type": rr_type,
        "redrock_quality_state": rr_quality,
        "redrock_nonstellar_evidence": rr_nonstellar,
    }


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
