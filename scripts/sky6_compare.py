#!/usr/bin/env python3
"""Compose a labeled comparison from completed public density-panel images."""
import argparse,json
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw,ImageFont
from matplotlib.font_manager import findfont
from sky6_common import ROOT,sha,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'figures/sky_sequence_20261119');p.add_argument('--output',type=Path);args=p.parse_args();out=args.output or args.root/'comparisons';out.mkdir(parents=True,exist_ok=True)
    fontpath=findfont('DejaVu Sans');font=ImageFont.truetype(fontpath,21);canvas=Image.new('RGB',(2800,820),'white');draw=ImageDraw.Draw(canvas);inputs={}
    layouts=[('Original:10–15 /15–26 /26–100','rgb/rgb_overview.png'),('0.01–5 /5–10 /10–100','channel_trials/rgb_near5_10_100_{kind}.png'),('0.01–3 /3–15 /15–100','channel_trials/rgb_near3_15_100_{kind}.png'),('Final:5–10 /10–20 /20–100','selected_channels/rgb_selected_{kind}.png')]
    for row,kind in enumerate(('fixed_gain','balanced')):
        for col,(label,template) in enumerate(layouts):
            path=args.root/template.format(kind=kind);inputs[str(path.relative_to(args.root))]=sha(path)
            with Image.open(path) as im:
                crop=im.crop((int(im.width*.025),0,im.width,int(im.height*.48)));tile=ImageOps.contain(crop.convert('RGB'),(700,350));canvas.paste(tile,(col*700,row*410+60))
            draw.text((col*700+8,row*410+5),label+' kpc',font=font,fill='black');draw.text((col*700+8,row*410+32),'Previous overview (unchanged)' if col==0 else 'Fixed V5 color gains' if row==0 else 'Balanced on old-domain reference',font=font,fill='black')
    for ext in ('png','pdf'):canvas.save(out/('channel_layout_comparison.'+ext),resolution=200)
    save_json(out/'comparison.json',dict(inputs=inputs,font_sha256=sha(Path(fontpath)),scope='Density-panel thumbnail comparison of already completed scientific plates; final user choice is5–10/10–20/20–100; full-size density/residual plates are separate'))
    if (args.root/'smoothing_trials').exists():
        canvas=Image.new('RGB',(2100,470),'white');draw=ImageDraw.Draw(canvas);inputs={}
        draw.text((15,5),'Background smoothing preview · B5–10 / G10–20 / R20–100 kpc · same density and color stretch',font=font,fill='black')
        for col,sigma in enumerate((2,5,10)):
            path=args.root/'smoothing_trials'/f'rgb_background_{sigma}deg.png';inputs[str(path.relative_to(args.root))]=sha(path)
            with Image.open(path) as im:
                crop=im.crop((int(im.width*.025),int(im.height*.505),im.width,int(im.height*.95)));tile=ImageOps.contain(crop.convert('RGB'),(700,355));canvas.paste(tile,(col*700,75))
            draw.text((col*700+15,40),f'Background σ={sigma}°'+(' · main reference' if sigma==5 else ' · preview only'),font=font,fill='black')
        draw.text((15,435),'Density smoothing remains σ=0.3°; only the background subtraction changes. The main5° image is unchanged.',font=font,fill='black')
        for ext in ('png','pdf'):canvas.save(out/('smoothing_comparison.'+ext),resolution=200)
        save_json(out/'smoothing_comparison.json',dict(inputs=inputs,font_sha256=sha(Path(fontpath)),scope='Positive residual only;2/5/10degree Gaussian background; identical main stretch/density; previews do not replace the main image'))


if __name__=='__main__':main()
