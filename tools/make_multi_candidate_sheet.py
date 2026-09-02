from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "extracted_marine_3model_comparison/marine_3model_comparison/prepared_datasets/trash_icra19_clean"
NAMES = [
    "obj1662_frame0000170_64005058", "obj0011_frame0000040_46758303", "obj0000_frame0000041_85690965",
    "obj1656_frame0000372_38282188", "obj1658_frame0001112_12480953", "obj1202_frame0000637_45522786",
    "obj1202_frame0000128_37657814", "obj1111_frame0000050_65542203", "obj0309_frame0000070_41367500",
    "obj1629_frame0000068_33958813", "obj0340_frame0000030_127868", "obj1628_frame0000231_96475813",
    "obj1297_frame0000326_864394", "bio0015_frame0000154_45527758", "obj1654_frame0000370_72786289",
    "bio0000_frame0000016_92318997", "obj1024_frame0000563_87067394", "obj0309_frame0000130_93266437",
    "obj1658_frame0000474_89418041", "obj1523_frame0000034_44518154", "bio0001_frame0000003_20525094",
    "obj1297_frame0000247_77297097", "obj1629_frame0000035_36418266", "bio0001_frame0000145_88772937",
]

tw, th, lh = 360, 240, 30
sheet = Image.new("RGB", (tw * 4, (th + lh) * 6), "white")
sd = ImageDraw.Draw(sheet)
for i, name in enumerate(NAMES):
    source = next((DATA / "images/test").glob(name + ".*"))
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    rows = (DATA / "labels/test" / f"{name}.txt").read_text().splitlines()
    for row in rows:
        cls, cx, cy, bw, bh = map(float, row.split())
        x1, y1, x2, y2 = (cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h
        draw.rectangle((x1, y1, x2, y2), outline="#ff3b30", width=max(3, w//300))
    image.thumbnail((tw, th))
    x, y = i % 4 * tw, i // 4 * (th + lh)
    sheet.paste(image, (x+(tw-image.width)//2, y+(th-image.height)//2))
    sd.text((x+7, y+th+6), f"{i+1}: {name}", fill="black")
sheet.save(ROOT / "research_figures/multi_candidate_sheet.png")
