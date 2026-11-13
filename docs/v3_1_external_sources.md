# v3.1 external-source availability and ancestry

This is a reference-availability census before correction fitting, not an opened
final score or a qualified scientific sample. Native DR11 TYPE is still required
before admission. Acquisition records carry no accepted scientific role.

## APOGEE DR17

The local allStar synspec_rev1 file and astroNN VAC have733901rows each. Verify
APOGEE_ID and APOGEE coordinates before treating them as aligned; equal length
alone is insufficient. Preserve the original row, flags, repeats and exact int64
Gaia ID. The existing highest-S/N/row-order duplicate policy and conservative
ASPCAP quality/branch definitions are reused without photdis residual selection.
Acquisition uses the union of qualified class evidence and finite, positive
spectroscopic distance/error with relative error below20%; distance availability
does not require qualified FeH/class labels. Neither is native admission yet.

The official release notes distinguish synspec_rev1 from the synspec tree used
by VACs; this is why alignment and parameter provenance are checked explicitly.
[SDSS DR17 access documentation](https://www.sdss4.org/dr17/irspec/spectro_data/)

AstroNN distance uses a GaiaEDR3-trained network, so an independent spectrum/star
does not establish an independent absolute teacher scale.
[Official VAC description](https://www.sdss.org/dr20/data_access/value-added-catalogs/?vac_id=85)

The producer code converts the network luminosity prediction and extinction-
corrected K magnitude to parsecs with fakemag_to_pc. Its weighted_dist instead
combines network and Gaia parallaxes. Use dist/dist_error for the availability
bins; never weighted_dist as an independent distance observation, nor invert weak
or negative Gaia. Local FITS headers omit units, so the conversion is documented
from the producer implementation rather than inferred from numerical magnitudes.
[Producer distance code](https://raw.githubusercontent.com/henrysky/astroNN_APOGEE_VAC/master/3_astroNN_chem_dist_ages.py)

Field/source/cell counts are acquisition metadata. Separate outside-v3 identity,
outside-v3 sky and outside-all-recorded-field footprints. Historical global
catalog inventory is not proof of fitted/scored use, and unused identities in
seen sky are not called new-sky validation. Final blocks and common-scale ancestry
must be frozen before any candidate correction fitting or final residual scoring.

## Globular-cluster VAC

The local v1.0 header reports7260rows and72cluster parameter rows, with dFeH/dRV/
dPM/dPOS columns but no explicit membership probability/flag. The current online
description refers to7732observations/6422stars, so do not silently treat the
local version as the updated catalog or assume missing membership flags.
[Official cluster VAC description](https://www.sdss.org/dr20/data_access/value-added-catalogs/?vac_id=105)

Inventory local group names and exact Gaia IDs first. Membership cuts, repeat
handling, common distance scales and historical system usage require their own
checks. Previously examined M3/NGC2419/NGC288 and other historical systems cannot
be made blind by choosing new member rows. Cluster scale uncertainty is shared
across members, and teacher/photometric membership dependence remains explicit.


## Common-scale evidence availability, still unassigned

The [APOGEE GC paper](https://arxiv.org/html/2310.07764v1) defines candidate
membership from position, proper motion and radial velocity, with a more
permissive outlier group. It does not certify every local v1.0 row as a member.
The local7260-row schema differs from the current published catalogue; use an
independently verified membership source or document version-specific selection.
No source identity alone establishes a newly blind system.

A potential10–20kpc scale is [Arellano Ferro et al.2024, NGC1851](https://www.astroscu.unam.mx/rmaa/RMxAA..60-2/PDF/RMxAA..60-2_aarellano-XVI.pdf):
the abstract reports11.9±0.6kpc from member RR Lyrae Fourier light curves.
Section6 adopts E(B−V)=.032±.002, which may lie outside the fixed application
EBV<.03 domain; actual native metadata decides admission. Section7 refers its
absolute-magnitude zero-point calibration to earlier work. That ancestry,
shared systematic budget, native PSF eligibility and historical-system exposure
must be verified before registering a scale test. This is an availability lead,
not an adopted reference, new test outcome or permission to loosen cuts.
