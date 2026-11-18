# Halo response, 18 November 2026

This is a scientific data/figure delivery on frozen production `87069408`,
reviewing evidence `7f9fcc78`. It does not replace the production catalogue.
The complete review is archived at
[`docs/pro_review_7f9fcc78_halo.md`](../../docs/pro_review_7f9fcc78_halo.md);
the registered experiment is
[`configs/halo_response_20261118.json`](../../configs/halo_response_20261118.json).

Read [scientific decisions](science_decisions.md) first, then the
[figure index](figures.md) and [execution notes](execution.md).

Main figure:
[full-footprint distance-slice CMD atlas](../../figures/halo_response_20261118/atlas/cmd_filter_aitoff_distance_slices.pdf).
It uses actual Nside512 counts, independent random areas, Galactic Aitoff,
English labels and density colorbars. The southern bright/faint sky area
is counted once. The northern physical sequence is an explicitly
validated empirical native transfer of the DECam template.

Numerical directories retain source-bound receipts, actual selections,
counts, uncertainty scope and failed outcomes. `sequence/revision_01`
retains the earlier combined low-crowding table before separating the
unmasked primary sample. Original noise/power/local-filter products in
`science_validation_20261118` remain unchanged. Execution logs identify
numerical/API recovery; none of those failures changed the production model.

Large source arrays, the 61-node FITS cube and full-sphere smoothed maps stay
on NERSC under `/pscratch/sd/y/yunao/photdis_pipeline/data/halo_response_20261118`.
`redraw/private_cube.json` gives its exact location, size, checksum, axes
and units. Compact losslessly compressed plotted grids are in `redraw/`;
matching scales and sample counts are in `atlas/*.csv`.
The large task inventory is stored as lossless `inventory/tasks.json.gz`;
its decompressed checksum is recorded in `inventory/complete.json`.

Replot a density plate from the repository:

```bash
PYTHONPATH=scripts:src:.deps:.deps/v39_dust python scripts/halo_redraw.py \
  manifests/halo_response_20261118/redraw/cmd_filter_aitoff_distance_slices \
  --output /tmp/photdis_halo_redraw
```

Copy the matching `atlas/<figure>.csv` beside the compact NPZ first,
or use the copies supplied in `redraw/`. `--annotated` overlays the
pre-existing cluster positions and frozen GD-1 track. These labels do not
constitute new structure detections. Raw plotting grids contain physical
values, not a screenshot or RGB composite.
