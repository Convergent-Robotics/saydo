"""Numbered marks on the camera frames, and object arguments read as those numbers (ablation 7).
No Isaac imports.

Each object in the scene carries a number, drawn on both camera images at every turn, and the
tools take that number: pick(object="3"). Which glass is the culture, which is the sink, which
fridge the bottle goes in: the model works it out from what it sees, the cross-modal reasoning
under test. Nothing is matched on its words, so no description can resolve to the wrong object.

The numbers are shuffled with the episode's seed, so no number stands for a role across episodes
(the hazardous vessel is not always 1). A number says which object it is, nothing about what it
holds.
"""
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf")


def assign(objects, seed):
    """{object: number}, 1..N, shuffled with the seed."""
    names = sorted(objects)
    numbers = list(range(1, len(names) + 1))
    random.Random(seed).shuffle(numbers)
    return dict(zip(names, numbers))


def number_in(value):
    """The one mark number an argument gives, or None if it gives none: 3, "3", "#3",
    "object 3", "the beaker marked 3". Raises ValueError if it gives two different numbers."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    # A number with a unit after it is a quantity, not a mark: "the 250 mL flask", "at 37 C".
    found = {int(n) for n in re.findall(r"(?<![\w.-])(\d+)(?![\w.]|\s*(?:(?:ml|µl|ul|l|g|mg|c)\b|%|°))",
                                         str(value), flags=re.IGNORECASE)}
    if len(found) > 1:
        raise ValueError(f"{value!r} names more than one number ({', '.join(map(str, sorted(found)))})")
    return found.pop() if found else None


def _font(size):
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def draw(image, points):
    """The image with each number drawn at its point: {number: (u, v)}, u and v as fractions of
    the width and height from the top left. A white disc with a dark rim and the number in black,
    sized to the image, so it reads on glass, steel and the dark bench alike."""
    img = image.convert("RGB").copy()
    w, h = img.size
    r = max(10, round(min(w, h) * 0.022))
    font = _font(round(r * 1.25))
    d = ImageDraw.Draw(img)
    for n, (u, v) in sorted(points.items()):
        x, y = u * w, v * h
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255), outline=(20, 20, 20), width=max(2, r // 6))
        d.text((x, y), str(n), fill=(0, 0, 0), font=font, anchor="mm")
    return img
