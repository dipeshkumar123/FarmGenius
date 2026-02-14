"""Fast cleanup: remove only completely unidentifiable images."""
import os
import glob
from PIL import Image, ImageFile

# Allow truncated images (they can still be used for training)
ImageFile.LOAD_TRUNCATED_IMAGES = True

dataset_path = os.path.join("dataset", "disease_images_split")
extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

all_files = []
for ext in extensions:
    all_files.extend(glob.glob(os.path.join(dataset_path, "**", ext), recursive=True))

print(f"Scanning {len(all_files)} image files...")

removed = 0
for i, fp in enumerate(all_files):
    if (i + 1) % 5000 == 0:
        print(f"  Checked {i+1}/{len(all_files)}... ({removed} removed so far)")
    try:
        sz = os.path.getsize(fp)
        if sz == 0:
            os.remove(fp)
            removed += 1
            continue
        img = Image.open(fp)
        img.verify()  # Quick header check only
    except Exception as e:
        if "cannot identify" in str(e).lower():
            os.remove(fp)
            removed += 1

print(f"\nDone! Removed {removed} unidentifiable files out of {len(all_files)} total.")
