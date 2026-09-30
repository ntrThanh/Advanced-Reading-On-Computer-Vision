import os
import shutil
import zipfile
import random
from pathlib import Path
from ultralytics import YOLO

random.seed(42)

DATA_ZIP = "/root/drone-yolo-detection.zip"
EXTRACT_DIR = "/root/raw_drone_data"
DEST_DIR = "/root/dataset_drone"
yaml_path = Path(DEST_DIR) / "data.yaml"

if not os.path.exists(yaml_path):
    print("1. Extracting zip archive...")
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(DATA_ZIP, "r") as z:
        z.extractall(EXTRACT_DIR)

    print("2. Indexing images and label files...")
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_map = {}
    label_map = {}

    for root, _, files in os.walk(EXTRACT_DIR):
        for f in files:
            p = Path(root) / f
            stem = p.stem.lower()
            if p.suffix.lower() in img_extensions:
                image_map[stem] = p
            elif p.suffix.lower() == ".txt":
                label_map[stem] = p

    common_stems = sorted(list(set(image_map.keys()) & set(label_map.keys())))
    print(f"Total matched image-label pairs: {len(common_stems)}")

    pos_stems = []
    neg_stems = []
    for s in common_stems:
        lbl_file = label_map[s]
        if lbl_file.stat().st_size > 0:
            pos_stems.append(s)
        else:
            neg_stems.append(s)

    print(f"Positive samples: {len(pos_stems)}, Negative samples: {len(neg_stems)}")

    random.shuffle(pos_stems)
    random.shuffle(neg_stems)

    total_needed = 1000 + 100 + 200
    all_selected = common_stems.copy()
    random.shuffle(all_selected)
    selected_stems = all_selected[:total_needed]

    train_stems = selected_stems[:1000]
    val_stems = selected_stems[1000:1100]
    test_stems = selected_stems[1100:1300]

    print(f"Split sizes: Train={len(train_stems)}, Val={len(val_stems)}, Test={len(test_stems)}")

    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)

    splits = {
        "train": train_stems,
        "val": val_stems,
        "test": test_stems
    }

    for split_name, stems in splits.items():
        img_dir = Path(DEST_DIR) / "images" / split_name
        lbl_dir = Path(DEST_DIR) / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for s in stems:
            src_img = image_map[s]
            src_lbl = label_map[s]
            shutil.copy2(src_img, img_dir / f"{s}{src_img.suffix}")
            shutil.copy2(src_lbl, lbl_dir / f"{s}.txt")

    yaml_content = f"""path: {DEST_DIR}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['drone']
"""

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"data.yaml generated at {yaml_path}")
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
else:
    print(f"Dataset already prepared at {yaml_path}. Skipping extraction.")

print("3. Launching YOLOv8 fine-tuning on RTX 3060...")
model = YOLO("yolov8n.pt")

results = model.train(
    data=str(yaml_path),
    epochs=25,
    imgsz=640,
    batch=16,
    lr0=0.01,
    cos_lr=True,
    mosaic=1.0,
    device=0,
    project="/root/runs_drone",
    name="yolov8_drone",
    exist_ok=True
)

print("4. Evaluating on Test Split (200 images)...")
test_metrics = model.val(data=str(yaml_path), split="test")
print("Test mAP50:", test_metrics.box.map50)
print("Test mAP50-95:", test_metrics.box.map)
print("Training & Evaluation COMPLETED successfully!")
