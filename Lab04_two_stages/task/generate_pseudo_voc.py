import os
import json
import time
import torch
import torchvision
from PIL import Image
import xml.etree.ElementTree as ET
from xml.dom import minidom
from torchvision.transforms import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = "/workspace/Advanced-Reading-On-Computer-Vision/Lab04_two_stages/capstone_project"
img_dir = os.path.join(base_dir, "val2017")
ann_file = os.path.join(base_dir, "annotations", "instances_val2017.json")
voc_dir = os.path.join(base_dir, "voc_traffic")
ann_out_dir = os.path.join(voc_dir, "Annotations")
sets_dir = os.path.join(voc_dir, "ImageSets", "Main")

os.makedirs(ann_out_dir, exist_ok=True)
os.makedirs(sets_dir, exist_ok=True)

with open(ann_file) as f:
    coco = json.load(f)

target_names = {"person", "car", "bus", "truck", "motorcycle", "bicycle"}
target_ids = {c["id"] for c in coco["categories"] if c["name"] in target_names}

matching_img_ids = sorted(list({a["image_id"] for a in coco["annotations"] if a["category_id"] in target_ids}))
id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
matching_files = [id_to_file[i] for i in matching_img_ids if i in id_to_file]

all_files = sorted([img["file_name"] for img in coco["images"]])
non_matching = [f for f in all_files if f not in set(matching_files)]

train_files = matching_files[:2000]
test_files = matching_files[2000:] + non_matching[:1000 - len(matching_files[2000:])]

with open(os.path.join(sets_dir, "train.txt"), "w") as f:
    for fn in train_files:
        f.write(os.path.splitext(fn)[0] + "\n")

with open(os.path.join(sets_dir, "test.txt"), "w") as f:
    for fn in test_files:
        f.write(os.path.splitext(fn)[0] + "\n")

print(f"Prepared splits: {len(train_files)} train, {len(test_files)} test.")

label_map = {
    1: "pedestrian",
    2: "vehicle",
    3: "vehicle",
    4: "vehicle",
    6: "vehicle",
    8: "vehicle"
}

weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights).to(device)
model.eval()

score_thresh = 0.5
batch_size = 2
total_images = len(train_files)
total_objects = 0
cat_counts = {"vehicle": 0, "pedestrian": 0}

start_time = time.time()
print("Starting pseudo-labeling on 2,000 images...")

for i in range(0, total_images, batch_size):
    batch_fns = train_files[i:i + batch_size]
    pil_imgs = []
    tensors = []
    valid_fns = []
    
    for fn in batch_fns:
        p = os.path.join(img_dir, fn)
        try:
            img = Image.open(p).convert("RGB")
            pil_imgs.append(img)
            tensors.append(F.to_tensor(img).to(device))
            valid_fns.append(fn)
        except Exception:
            continue
            
    if not tensors:
        continue

    with torch.inference_mode():
        preds = model(tensors)
        
    for fn, img, pred in zip(valid_fns, pil_imgs, preds):
        w, h = img.size
        boxes = pred["boxes"].cpu()
        scores = pred["scores"].cpu()
        labels = pred["labels"].cpu()
        
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = "val2017"
        ET.SubElement(annotation, "filename").text = fn
        
        size_elem = ET.SubElement(annotation, "size")
        ET.SubElement(size_elem, "width").text = str(w)
        ET.SubElement(size_elem, "height").text = str(h)
        ET.SubElement(size_elem, "depth").text = "3"
        
        for box, score, label_id in zip(boxes, scores, labels):
            score_val = score.item()
            lid = label_id.item()
            if score_val >= score_thresh and lid in label_map:
                cat_name = label_map[lid]
                cat_counts[cat_name] += 1
                total_objects += 1
                
                obj_elem = ET.SubElement(annotation, "object")
                ET.SubElement(obj_elem, "name").text = cat_name
                ET.SubElement(obj_elem, "pose").text = "Unspecified"
                ET.SubElement(obj_elem, "truncated").text = "0"
                ET.SubElement(obj_elem, "difficult").text = "0"
                
                bndbox = ET.SubElement(obj_elem, "bndbox")
                xmin = max(1, int(round(box[0].item())))
                ymin = max(1, int(round(box[1].item())))
                xmax = min(w, int(round(box[2].item())))
                ymax = min(h, int(round(box[3].item())))
                
                ET.SubElement(bndbox, "xmin").text = str(xmin)
                ET.SubElement(bndbox, "ymin").text = str(ymin)
                ET.SubElement(bndbox, "xmax").text = str(xmax)
                ET.SubElement(bndbox, "ymax").text = str(ymax)
                
        xml_str = ET.tostring(annotation, encoding="utf-8")
        parsed = minidom.parseString(xml_str)
        pretty_xml = parsed.toprettyxml(indent="  ")
        
        stem = os.path.splitext(fn)[0]
        xml_path = os.path.join(ann_out_dir, f"{stem}.xml")
        with open(xml_path, "w", encoding="utf-8") as xf:
            xf.write(pretty_xml)
            
    if (i + batch_size) % 200 == 0 or (i + batch_size) >= total_images:
        elapsed = time.time() - start_time
        processed = min(i + batch_size, total_images)
        fps = processed / elapsed if elapsed > 0 else 0
        print(f"Processed {processed}/{total_images} images in {elapsed:.1f}s ({fps:.1f} fps) - Objects found: {total_objects} ({cat_counts})")

print("Finished generating pseudo-labels successfully!")
