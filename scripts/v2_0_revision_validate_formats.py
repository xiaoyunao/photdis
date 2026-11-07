"""Validate downloaded revision figure formats and English-only PDF text."""
import json,re,concurrent.futures
from pathlib import Path
from PIL import Image
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
M=ROOT/'manifests/v2_0_revision'
def check(record):
    with Image.open(ROOT/record['png']) as im:
        assert im.mode=='RGBA',record['name']
        assert abs(im.info['dpi'][0]-300)<.1,record['name']
        assert im.getpixel((0,0))[3]==0,record['name']
    pdf=PdfReader(ROOT/record['pdf'])
    assert len(pdf.pages)==1,record['name']
    text=pdf.pages[0].extract_text()
    assert not re.search(r'[\u3400-\u9fff]',text),record['name']
    return record['name']
if __name__=='__main__':
    records=json.loads((M/'figure_registry.json').read_text())
    with concurrent.futures.ProcessPoolExecutor(4) as pool:
        result=list(pool.map(check,records.values()))
    (M/'format_validation.json').write_text(json.dumps(dict(figure_sets=len(result),all_single_page_pdfs=True,all_png_rgba_300dpi=True,transparent_corner=True,no_chinese_pdf_text=True),indent=2)+'\n')
    print('Format checks pass',len(result))
