"""Label images for the sim's objects, from task.json's object_labels.

A label's text is split on " | " into lines. The text is what the text ablations read; the
object's `label_style` in the config only sets how it looks in the sim:

  card         (default) a sign: black on white in a black frame. A line of GHS codes ("GHS05
               GHS09") becomes a row of red diamonds with the code inside; a line in capitals is
               printed in red, as warnings on signs are.
  printed      a supplier's bottle label: the first line is the product name, GHS codes become
               the real pictograms (sim/assets/ghs), a line in capitals is the signal word in
               bold black, the rest is small print.
  handwritten  marker on lab tape (sim/assets/fonts/PermanentMarker), as glassware is labelled at
               the bench: contents, date, initials.

In any style, a line of "☣" (U+2623) becomes the biohazard sign, ISO 7010 W009: the symbol in
black in a yellow triangle. The text ablations show the character itself.

`label_wear: worn` in the config ages the label as a household product's gets: yellowed paper, a
few drip stains, scuffs and a lifted corner. Every word stays legible: the text is still what the
text ablations read, and on a hazard label it is the cue the task rests on.

    python sim/labels.py tasks/lab/lab_02_bleach_acid/task.json out/ [labutopia-config.yaml]   # to look at them
"""
import random
import re
import sys
import zlib
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

WIDTH = 600
MIN_HEIGHT = 330  # a one-line label still gets a card, not a strip
MARGIN = 30
FONTS = ["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
REGULAR = ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
ASSETS = Path(__file__).resolve().parent / "assets"
MARKER = [str(ASSETS / "fonts" / "PermanentMarker-Regular.ttf")]
TAPE = (246, 242, 228)     # off-white lab tape
INK = (25, 30, 70)         # dark blue-black marker


def font(size, names=FONTS):
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def wrap(draw, text, f, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=f) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line]


def diamond(draw, cx, cy, r, code):
    draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)],
                 fill="white", outline=(200, 0, 0), width=max(4, r // 8))
    f = font(r // 3)
    w = draw.textlength(code, font=f)
    draw.text((cx - w / 2, cy - r // 6), code, font=f, fill="black")


def biohazard(draw, x, y, side):
    """ISO 7010 W009 with its top-left corner at (x, y): a yellow triangle, black border, black
    symbol."""
    h = side * 0.866
    corners = [(x + side / 2, y), (x + side, y + h), (x, y + h)]
    draw.polygon(corners, fill=(249, 168, 0), outline="black", width=max(6, side // 18))
    f = font(int(side * 0.5))
    box = draw.textbbox((0, 0), "\u2623", font=f)
    cx, cy = x + side / 2, y + h * 0.62  # the triangle's centroid sits low
    draw.text((cx - (box[0] + box[2]) / 2, cy - (box[1] + box[3]) / 2), "\u2623", font=f, fill="black")


def pictogram(code, size):
    """A GHS pictogram from sim/assets/ghs, or None for a code we have no image of."""
    path = ASSETS / "ghs" / f"{code}.png"
    return Image.open(path).convert("RGB").resize((size, size)) if path.exists() else None


def render_printed(text):
    """A supplier's label: product name, pictograms, signal word, statements."""
    canvas = Image.new("RGB", (WIDTH, 2000), "white")
    draw = ImageDraw.Draw(canvas)
    y = MARGIN
    for i, part in enumerate(p.strip() for p in text.split("|")):
        codes = re.findall(r"GHS\d\d", part)
        if codes and not re.sub(r"GHS\d\d|\s", "", part):
            size = 150
            for j, code in enumerate(codes):
                img = pictogram(code, size)
                if img is None:
                    diamond(draw, MARGIN + size // 2 + j * (size + 10), y + size // 2, size // 2, code)
                else:
                    canvas.paste(img, (MARGIN + j * (size + 10), y))
            y += size + 15
            continue
        if part and not part.replace("\u2623", "").strip():
            biohazard(draw, MARGIN, y, 170)
            y += int(170 * 0.866) + 15
            continue
        if i == 0:
            f = font(48)
        elif part.isupper() and len(part) > 3:
            f = font(40)
        else:
            f = font(28, REGULAR)
        for line in wrap(draw, part, f, WIDTH - 2 * MARGIN):
            draw.text((MARGIN, y), line, font=f, fill="black")
            y += int(f.size * 1.2)
        y += 8
    height = max(y + MARGIN, MIN_HEIGHT)
    canvas = canvas.crop((0, 0, WIDTH, height))
    ImageDraw.Draw(canvas).rectangle([8, 8, WIDTH - 9, height - 9], outline=(60, 60, 60), width=2)
    return canvas


def render_handwritten(text):
    """Marker on a strip of lab tape, one line per part, centred."""
    f = font(70, MARKER)
    canvas = Image.new("RGB", (WIDTH, 2000), TAPE)
    draw = ImageDraw.Draw(canvas)
    y = 25
    for part in (p.strip() for p in text.split("|")):
        for line in wrap(draw, part, f, WIDTH - 50):
            w = draw.textlength(line, font=f)
            draw.text(((WIDTH - w) / 2, y), line, font=f, fill=INK)
            y += int(f.size * 1.15)
    return canvas.crop((0, 0, WIDTH, y + 25))


def worn(img, seed):
    """The label aged: yellowed, drip-stained, scuffed, one corner lifting. Stains are faint and
    multiplied over the print, so no word is covered. The same text always wears the same way."""
    rng = random.Random(seed)
    w, h = img.size
    img = ImageChops.multiply(img, Image.new("RGB", img.size, (240, 232, 210)))  # yellowed paper
    dirt = Image.new("RGB", img.size, "white")
    d = ImageDraw.Draw(dirt)
    for i in range(14):  # grime where hands hold it: a darker band round the edge
        g = 236 - i * 3
        d.rectangle([i * 2, i * 2, w - 1 - i * 2, h - 1 - i * 2], outline=(g + 10, g + 4, g - 10), width=2)
    d.rectangle([28, 28, w - 29, h - 29], fill="white")
    for _ in range(3):  # drips run down from the cap: a ragged blot, a darker dried rim, a tail
        x, y = rng.uniform(0.15, 0.85) * w, rng.uniform(0.0, 0.35) * h
        r = rng.uniform(14, 26)
        for _ in range(7):
            ox, oy, rr = rng.gauss(0, r * 0.7), rng.gauss(0, r * 0.7), r * rng.uniform(0.5, 1.0)
            d.ellipse([x + ox - rr, y + oy - rr, x + ox + rr, y + oy + rr], fill=(228, 214, 182), outline=(206, 186, 148), width=2)
        tail = rng.uniform(60, 160)
        d.line([x, y, x + rng.uniform(-6, 6), y + tail], fill=(222, 206, 170), width=int(r * 0.5))
    img = ImageChops.multiply(img, dirt.filter(ImageFilter.GaussianBlur(2)))
    rub = Image.new("RGB", img.size, "black")  # scuffs: where it has been rubbed, the print goes pale
    d = ImageDraw.Draw(rub)
    for _ in range(18):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        d.line([x, y, x + rng.uniform(-90, 90), y + rng.uniform(-10, 10)], fill=(110, 105, 95), width=rng.choice([2, 3, 4]))
    img = ImageChops.lighter(img, rub.filter(ImageFilter.GaussianBlur(1)))
    # The bottom right corner peels up: its back shows, in the paper's colour, with a crease line.
    d = ImageDraw.Draw(img)
    c = int(min(w, h) * 0.16)
    d.polygon([(w, h - c), (w - c, h), (w, h)], fill=(120, 120, 118))       # the shadow it casts
    d.polygon([(w - c, h), (w, h - c), (w - c * 0.85, h - c * 0.85)], fill=(236, 230, 214))
    d.line([(w - c, h), (w, h - c)], fill=(170, 160, 140), width=3)
    return img


def render_print(text):
    """Text printed straight onto an object's body ("<object>:print"): dark ink on a transparent
    background, no card and no border, so the frames show lettering on the plastic, not a second
    label (a card here read as a label: "the Distilled Water label or the Special Offer sticker?",
    9 Nov 2026). The sim binds it as a cutout (World._label_material)."""
    canvas = Image.new("RGBA", (WIDTH, 2000), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    y = MARGIN
    for i, part in enumerate(p.strip() for p in text.split("|")):
        f = font(52) if i == 0 else font(32, REGULAR)
        for line in wrap(draw, part, f, WIDTH - 2 * MARGIN):
            draw.text((MARGIN, y), line, font=f, fill=(55, 62, 72, 255))
            y += int(f.size * 1.2)
        y += 8
    return canvas.crop((0, 0, WIDTH, max(y + MARGIN, 120)))


def render_label(text, style=None, wear=None):
    """One label as an RGB image, WIDTH wide, as tall as its content."""
    img = render_style(text, style)
    if wear == "worn":
        return worn(img, zlib.crc32(text.encode()))
    if wear is not None:
        raise ValueError(f"unknown label_wear {wear!r}: worn")
    return img


def render_style(text, style=None):
    if style == "printed":
        return render_printed(text)
    if style == "handwritten":
        return render_handwritten(text)
    if style not in (None, "card"):
        raise ValueError(f"unknown label_style {style!r}: card, printed or handwritten")
    canvas = Image.new("RGB", (WIDTH, 2000), "white")
    draw = ImageDraw.Draw(canvas)
    y = MARGIN
    for part in (p.strip() for p in text.split("|")):
        codes = re.findall(r"GHS\d\d", part)
        if codes and not re.sub(r"GHS\d\d|\s", "", part):
            r = 70
            for i, code in enumerate(codes):
                diamond(draw, MARGIN + r + i * (2 * r + 20), y + r, r, code)
            y += 2 * r + 20
            continue
        if part and not part.replace("\u2623", "").strip():
            n, side = part.count("\u2623"), 300
            x = (WIDTH - n * side - (n - 1) * 20) // 2  # centred: a sign alone is the whole label
            for i in range(n):
                biohazard(draw, x + i * (side + 20), y, side)
            y += int(side * 0.866) + 20
            continue
        warning = part.isupper() and len(part) > 3
        f = font(46 if y == MARGIN else 40)
        for line in wrap(draw, part, f, WIDTH - 2 * MARGIN):
            draw.text((MARGIN, y), line, font=f, fill=(200, 0, 0) if warning else "black")
            y += int(f.size * 1.25)
        y += 10
    height = max(y + MARGIN, MIN_HEIGHT)
    canvas = canvas.crop((0, 0, WIDTH, height))
    ImageDraw.Draw(canvas).rectangle([0, 0, WIDTH - 1, height - 1], outline="black", width=6)
    return canvas


def styles(cfg):
    """{object name: (label_style, label_wear)} from a LabUtopia config."""
    return {n: (o.get("label_style"), o.get("label_wear")) for n, o in cfg["spai"]["objects"].items()}


def cards(spec):
    """Every card the sim attaches: the object labels, and each object_prints text as "<object>:print"."""
    pc = spec["planning_context"]
    return {**(pc.get("object_labels") or {}), **{f"{o}:print": t for o, t in (pc.get("object_prints") or {}).items()}}


def write_labels(object_labels, out_dir, label_styles=None):
    """{object name: png path} for every labelled object."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for obj, text in object_labels.items():
        # "<object>:print", printed on the body: the object's style, never worn
        base, _, part = obj.partition(":")
        style, wear = (label_styles or {}).get(base, (None, None))
        paths[obj] = out_dir / f"label_{obj.replace(':', '_')}.png"
        (render_print(text) if part else render_label(text, style, wear)).save(paths[obj])
    return paths


if __name__ == "__main__":
    import json
    import yaml
    spec = json.loads(Path(sys.argv[1]).read_text())
    cfg_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(sys.argv[1]).parent / spec.get("_base_config", "-")
    st = styles(yaml.safe_load(cfg_path.read_text())) if cfg_path.is_file() else {}
    for obj, path in write_labels(spec["planning_context"]["object_labels"], sys.argv[2], st).items():
        print(obj, path)
