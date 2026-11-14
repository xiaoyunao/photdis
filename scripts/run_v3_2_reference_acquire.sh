#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 00:40:00
#SBATCH -J photdis_v3_2_reference_acquire
#SBATCH -o logs/photdis_v3_2_reference_acquire_%j.out
set -euo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
dest=data/v3_2/reference_sources/dr19_astra_0.6.0
mkdir -p "$dest"
base=https://dr19.sdss.org/sas/dr19/spectro/astra/0.6.0/summary
for product in AstroNNdist ASPCAP; do
    name=astraAllStar${product}-0.6.0.fits.gz
    case $product in
        AstroNNdist) bytes=534769970 ;;
        ASPCAP) bytes=1171102556 ;;
    esac
    if [[ -f "$dest/$name" ]]; then
        test "$(stat -c %s "$dest/$name")" = "$bytes"
        gzip -t "$dest/$name"
    else
        work="$dest/$name.${SLURM_JOB_ID:?Allocated job required}.partial"
        curl -fL --retry 2 --connect-timeout 20 --max-time 900 "$base/$name" -o "$work"
        test "$(stat -c %s "$work")" = "$bytes"
        gzip -t "$work"
        if ! ln "$work" "$dest/$name"; then cmp "$work" "$dest/$name"; fi
        rm "$work"
    fi
    sha256sum "$dest/$name"
done
