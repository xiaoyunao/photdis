#!/usr/bin/env python3
"""Render physical-width previews of the delivered scientific PDFs."""
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageDraw


def main():
    root = Path(__file__).resolve().parents[1] / 'figures/science_validation_20261118'
    plates = sorted(root.glob('*/*.pdf'))
    previews = []
    with tempfile.TemporaryDirectory(prefix='photdis-sv18-preview-') as temporary:
        for index, path in enumerate(plates):
            images = []
            for width in (852, 420):
                output = Path(temporary) / f'{index}_{width}'
                subprocess.run(['pdftoppm', '-scale-to-x', str(width), '-scale-to-y', '-1',
                                '-singlefile', '-png', str(path), str(output)], check=True)
                with Image.open(output.with_suffix('.png')) as image:
                    images.append(image.convert('RGB').copy())
            row = Image.new('RGB', (1320, max(image.height for image in images) + 50), 'white')
            draw = ImageDraw.Draw(row)
            draw.text((8, 8), str(path.relative_to(root)), fill='black')
            draw.text((8, 26), '7.1 in at 120 ppi', fill='black')
            draw.text((880, 26), '3.5 in at 120 ppi', fill='black')
            row.paste(images[0], (0, 50))
            row.paste(images[1], (880, 50))
            previews.append(row)
    sheet = Image.new('RGB', (1320, sum(image.height for image in previews)), 'white')
    y = 0
    for image in previews:
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(root / 'aas_width_preview.png', dpi=(120, 120))


if __name__ == '__main__':
    main()
