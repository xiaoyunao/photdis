"""Pal 5 New Step 5 typography and palette for photdis science figures."""
BLUE = '#1f77b4'
GREEN = '#2ca02c'
ORANGE = '#ff8c00'
GRAY = '#7f7f7f'
STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.labelsize': 17,
    'axes.titlesize': 18, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'legend.fontsize': 12, 'figure.dpi': 140, 'savefig.dpi': 300,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': .22, 'axes.axisbelow': True,
}

# Report maps retain empty cells and the page background as transparent.
TRANSPARENT_STYLE = {**STYLE, 'figure.facecolor': 'none',
                     'axes.facecolor': 'none', 'savefig.facecolor': 'none',
                     'savefig.transparent': True}
