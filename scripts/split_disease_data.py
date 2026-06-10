"""Split the disease image dataset into train/val directories (80/20 split)."""
import os, shutil, random

SRC = r"D:\Projects\FarmGenius\dataset\disease_images"
DST = r"D:\Projects\FarmGenius\dataset\disease_images_split"
SPLIT = 0.8
random.seed(42)

train_dir = os.path.join(DST, "train")
val_dir = os.path.join(DST, "val")

for cls in sorted(os.listdir(SRC)):
    cls_path = os.path.join(SRC, cls)
    if not os.path.isdir(cls_path):
        continue
    imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    random.shuffle(imgs)
    split_idx = int(len(imgs) * SPLIT)
    train_imgs = imgs[:split_idx]
    val_imgs = imgs[split_idx:]

    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy2(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))
    for img in val_imgs:
        shutil.copy2(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))

    print(f"{cls}: {len(train_imgs)} train / {len(val_imgs)} val")

print("\nDone! Split saved to", DST)
