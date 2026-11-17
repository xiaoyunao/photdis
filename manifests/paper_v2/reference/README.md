# Full-reference census

The denominator is 72,289,386 owned native PSFs in 28,917 frozen parts.
The 43,347,612 full PDFs are state 3. The other production states are
0 quality rejected, 1 lower-S/N not sampled, and 2 attempted but unsupported.
The southern bright branch had already passed its earlier support admission.
Every staged row is validated as DR11 PSF and joined by native ownership.

The `coverage.csv` columns count unique parent rows meeting increasingly
strict criteria. A Gaia record is a catalogue ID match. A valid measurement
also clears the frozen angular counterpart test and has a finite parallax
and positive error. Geometry additionally requires the valid Gaia DR3 zero
point, astrometric quality and corrected S/N ≥ 10. The signed-parallax
tables retain measurements in the same 4/5 S/N windows used in the v1 work.

A SpecDis or SPDist record means the native Gaia ID occurs in that VAC.
A valid distance also has a unique position-confirmed row and a positive
finite center. A qualified distance has the original reference-specific
quality and usable uncertainty flags. The held-out frame excludes either
system's distance fitting and northern zero-point calibrators, requires
strict native photometry, the role-4 sky pattern, and excludes ambiguous Gaia
aliases in the owned census. The exact F15 replay uses its own original
identity set and method pairing.

Population code 6 is the frozen-setting blue turnoff proxy, which requires
a blue observed color, qualified intermediate gravity and useful spectroscopic
temperature. It is disjoint from the main intermediate-gravity entry.
Giant labels group the bright evolved branch without resolving RGB from
other giant populations. WD is a published candidate or one of the
known spectroscopic WDs in the archived control roster. Unknown includes
sources with no qualified stellar labels. Nonstellar is an archived
quality-confirmed Redrock control.

`c68` and `c95` count reference-center containment in raw QRF intervals.
The reported reference modulus uncertainty is separate. Modal fraction
means at least two interior local maxima above 20% of the maximum
density after three-bin `[1,2,1]/4` smoothing. It depends on grid resolution.
C68, C95 and modes are diagnostics on the reference subset. The weak Gaia
histogram uses a signed observation-space CDF conditional on the original
observed S/N window, with three-point Gaussian quadrature in modulus.
