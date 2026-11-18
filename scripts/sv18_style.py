#!/usr/bin/env python3
"""Replay the original v3.8 plates under their locked rendering context."""
from photdis.stage1 import require_compute
from sv18_common import complete, out, plt
from v3_10_figures import STYLE, regression


if __name__ == '__main__':
    require_compute()
    folder = out('style_regression')
    with plt.rc_context(STYLE):
        baseline_images = regression(folder)
    complete(folder, baseline_images, pixel_comparison='both original v3.8 PNGs match exactly')
