# Potential fresh field references for v3.2

The v3.1 field roles (170 new-sky and 356 new-source/seen-sky objects)
and common-scale role (3032 objects in one system) were already opened.
The 3250 v3.0 identities were also opened and yielded no qualifying
distance evidence under that protocol. Preserve each role's identity
and footprint exclusions. See the v3.1 validation role receipt and the
v3.0 reservation availability report.

Start with the SDSS DR19 Astra 0.6.0 APOGEE AstroNNDist and ASPCAP
summary products. The archive responded with HTTP 200 and lengths
534769970 and 1171102556 bytes on 2026-11-14. Download and verify gzip
integrity on NERSC. Use the APOGEE extension and join the summaries on
consistent positive Gaia DR3 and SDSS IDs. Exclude previously exposed
IDs, historical imaging IDs, used sky blocks and systems, and require
native DECam DR11 TYPE=PSF under the locked photometric selection.
This acquisition alone assigns no final role and reads no outcomes.

The archive includes combined SDSS-V spectra as well as legacy DR17
spectra. Exclude DR17 APOGEE IDs and Gaia identities from the proposed
fresh DR19-only channel and require the APO 2.5m SDSS-V 1.3
reduction and at least one good visit in both products.
[SDSS DR19 infrared data](https://www.sdss.org/dr19/data_access/get_data/)
Use the Astra ASPCAP flags for independent stellar-parameter eligibility.
AstroNNDist uses spectra, 2MASS Ks and extinction and was trained using
DR17 plus Gaia EDR3. Use it as a teacher-distance channel. Keep
signed Gaia, whole-system distances and candidate-parent contamination
as separately qualified channels. [Astra summary schemas](https://www.sdss.org/dr19/mwm/astra/pipelines-in-astra/) · [AstroNNDist method](https://www.sdss.org/dr19/mwm/astra/pipelines-in-astra/astronn/)

DR20 copies the DR19 APOGEE spectra and analysis. Treat duplicate releases
as the same source lineage. The NERSC public DESI DR2 tree contains a
placeholder README and its `vac/dr2` directory had no listed files at this
check. DESI DR1 SpecDis and SPDist were used historically. A new sky and
identity audit is required for any additional DR1 reference use. The BHB
VAC has spectroscopic classification and a photometric absolute-magnitude
relation. Any distant BHB use would be a separate tracer channel.
[SDSS DR20 release](https://sdss.org/dr20/) · [DESI DR1 SpecDis](https://data.desi.lbl.gov/doc/releases/dr1/vac/mws-specdis/) · [DESI DR1 BHB](https://data.desi.lbl.gov/doc/releases/dr1/vac/mws-bhb/)

The index reads only provenance, positions and identity columns.
The allocated native matching job applies exact G3, primary, release,
morphology, historical imaging identity and photometric criteria. Its
reference index contains row pointers paired with the same PSF identities.
Archive checksums and both files' explicit column selection precede any
reference-value access.

Acquire and index sources without model residuals. Freeze the candidate,
reference eligibility, outcome-access rule and actual source roles before
evaluating any new validation outcomes.
