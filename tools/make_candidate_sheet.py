from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "extracted_marine_3model_comparison/marine_3model_comparison/prepared_datasets/trash_icra19_clean"
NAMES = [
    "obj0318_frame0000027_93944977", "bio0004_frame0000041_59287433",
    "obj0002_frame0000102_31790629", "obj1502_frame0000208_30872659",
    "obj1672_frame0000044_94665741", "obj1615_frame0000088_48263314",
    "obj0300_frame0000035_91856791", "obj0002_frame0000009_88385004",
    "obj0856_frame0000040_82060321", "obj0337_frame0000014_32730571",
    "obj1672_frame0000021_83060933", "obj1505_frame0000078_41745301",
    "bio0016_frame0000029_8553911", "obj0747_frame0000080_1758680",
    "obj1344_frame0000219_58157050", "bio0016_frame0000049_5151075",
    "obj1314_frame0000300_12284838", "obj0343_frame0000014_71916209",
    "obj0312_frame0000011_88617778", "obj0248_frame0000026_20224438",
    "obj0347_frame0000067_36272308", "obj1023_frame0000019_31758060",
    "obj1218_frame0000013_36283832", "obj0347_frame0000044_21673358",
]

def find_image(name: str) -> Path:
    matches = list((DATA / "images/test").glob(name + ".*"))
    if not matches:
        raise FileNotFoundError(name)
    return matches[0]

thumb_w, thumb_h, label_h = 360, 240, 32
sheet = Image.new("RGB", (thumb_w * 4, (thumb_h + label_h) * 6), "white")
draw = ImageDraw.Draw(sheet)
for i, name in enumerate(NAMES):
    image = Image.open(find_image(name)).convert("RGB")
    image.thumbnail((thumb_w, thumb_h))
    x = (i % 4) * thumb_w
    y = (i // 4) * (thumb_h + label_h)
    sheet.paste(image, (x + (thumb_w-image.width)//2, y + (thumb_h-image.height)//2))
    draw.text((x + 8, y + thumb_h + 7), f"{i+1}: {name}", fill="black")
sheet.save(ROOT / "research_figures/candidate_sheet.png")
