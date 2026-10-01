import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

SIZE = 64
BG = 25
PART = 200


def draw_part(rng, has_defect):
    '''Creates a part, with or without defect optionally'''

    img = Image.new("L", (SIZE, SIZE), color=BG)
    draw = ImageDraw.Draw(img)
    # Blank greyscale image with pen attached. Draws a part with or without defect.
    cx = SIZE / 2 + rng.uniform(-4, 4)
    cy = SIZE / 2 + rng.uniform(-4, 4)
    r = rng.uniform(14, 18)
    box = [cx - r, cy - r, cx + r, cy + r]
    # Random centre shift of part from centre of image. Random radius, generate bounding box.
    # randomly choose shaped. Draw a filled ellipse that fits box
    # or filled rectangle with rounded corners
    shape = str(rng.choice(["circle", "rect"]))

    if shape == "circle":
        draw.ellipse(box, fill=PART)
    else:
        draw.rounded_rectangle(box, radius=r * 0.3, fill=PART)

    defect_type = "none"
    # pick which type of defect and create a thin crack or a round pore represented by a darker area.
    if has_defect:
        defect_type = str(rng.choice(["crack", "pore"]))
        defect_val = int(PART - rng.uniform(60, 120))
        # polar position of defect
        a = rng.uniform(0, 2 * np.pi)
        d = rng.uniform(0, 0.4 * r)
        px = cx + d * np.cos(a)
        py = cy + d * np.sin(a)

        if defect_type == "crack":
            half = rng.uniform(2.5, 5)
            t = rng.uniform(0, np.pi)
            dx = half * np.cos(t)
            dy = half * np.sin(t)  # length, direction, offsets from centre to ends.
            draw.line([px - dx, py - dy, px + dx, py + dy],  # line between endpoints.
                      fill=defect_val, width=int(rng.integers(1, 3)))  # width of crack.
        else:
            pr = rng.uniform(1.5, 3)
            draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=defect_val)  # round pore.

    # rotate image by random angle
    img = img.rotate(rng.uniform(0, 360), resample=Image.Resampling.BILINEAR, fillcolor=BG)

    # return tuple, image and metadata
    return img, shape, defect_type


def apply_imaging(img, rng, noise_range, contrast_range, brightness_range=(-0.1, 0.1)):
    '''Applies imaging to an image, with optional noise, contrast, and brightness adjustments.
     returns image and alteration values used.'''
    # *range arguments are (min,max) tuples. * unpacks tupple
    x = np.asarray(img, dtype=np.float32) / 255.0
    # convert pillow image to numpy array single precision decimal, recale to 0 -1.0
    contrast = rng.uniform(*contrast_range)
    brightness = rng.uniform(*brightness_range)
    noise = rng.uniform(*noise_range)
    # random image contrast, brightness and noise level.
    x = (x - 0.5) * contrast + brightness + 0.5
    # squash or stretch every pixel around mid grey 0.5. brightnes, shift up or down.
    x = x + rng.normal(0, noise, x.shape)
    # ND noise
    x = np.clip(x, 0, 1)
    # force back to 0-1 range. <0 = 0 >1 = 1
    out = Image.fromarray((x * 225).astype(np.uint8))
    # converts back to image
    return out, contrast, brightness, noise


def generate_dataset(out_dir, n=500, seed=0, noise_range=(0.0, 0.02), contrast_range=(0.9, 1.1)):
    '''Generate a dataset+labelse csv, calls previous functions.'''
    out = Path(out_dir)
    # folder name to path object
    (out / "images").mkdir(parents=True, exist_ok=True)
    # creat out/imgase folder. creatses out if not already exist and dont crash if folder already exists
    rng = np.random.default_rng(seed)
    # creatse rng from reporducible seed
    rows = []
    for i in range(n):
        has_defect = i % 2 == 1
        # empty list for pictures, one dictionary per image. 50/50 split of defect or not.
        img, shape, defect_type = draw_part(rng, has_defect)
        # draws part and returns variables
        img, c, b, s = apply_imaging(img, rng, noise_range, contrast_range)
        # applies imaging conditions
        fname = f"img_{i:04d}.png"
        img.save(out / "images" / fname)
        # name and save
        rows.append({
            "filename": fname,
            "label": int(has_defect),  # True/False 1/0
            "shape": shape,
            "defect_type": defect_type,
            "contrast": round(c, 3),  # round to 3 decimal places
            "brightness": round(b, 3),
            "noise": round(s, 3),
        })
        # metatdata as a duictionary.
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {n} images to {out}.")
    # writes files without blank line. headers and data.


# only runs when python generate.py is run
if __name__ == "__main__":
    generate_dataset("data/clean", n=500, seed=0)