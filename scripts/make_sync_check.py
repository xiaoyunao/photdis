#!/usr/bin/env python3
"""Write a deterministic synchronization-check SVG, containing no science data."""

from __future__ import annotations

import argparse
from pathlib import Path


SVG = '''<svg xmlns="http://www.w3.org/2000/svg" width="960" height="320" viewBox="0 0 960 320">
<rect width="960" height="320" fill="#f5f7fb"/>
<g font-family="sans-serif" text-anchor="middle" fill="#18253d">
<text x="480" y="55" font-size="26" font-weight="bold">Photdis synchronization check</text>
<text x="480" y="85" font-size="16">Infrastructure demonstration · no scientific data</text>
<g stroke="#355da8" stroke-width="3" fill="#ffffff">
<rect x="70" y="130" width="220" height="90" rx="12"/>
<rect x="370" y="130" width="220" height="90" rx="12"/>
<rect x="670" y="130" width="220" height="90" rx="12"/>
<path d="M300 175H360 M345 165L360 175L345 185 M600 175H660 M645 165L660 175L645 185" fill="none"/>
</g>
<text x="180" y="182" font-size="22">NERSC</text>
<text x="480" y="182" font-size="22">Local Git</text>
<text x="780" y="182" font-size="22">GitHub</text>
<text x="480" y="275" font-size="16">Programs and selected figures · catalogs remain on NERSC</text>
</g>
</svg>
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("figures/sync_check.svg"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != SVG:
            parser.error(f"Output exists with different contents: {args.output}")
        print(f"Already identical: {args.output}")
        return 0
    args.output.write_text(SVG, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
