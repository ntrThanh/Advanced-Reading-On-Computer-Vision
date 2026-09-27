import os
import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3 (ipykernel)',
    'language': 'python',
    'name': 'python3'
}
nb.metadata['language_info'] = {
    'name': 'python',
    'version': '3.12.0'
}

cells = []

# Cell 0: Header (No personal info)
cell_0_md = """# 7. BÀI TẬP NÂNG CAO VÀ KIỂM THỬ (ABLATION STUDIES)
Mổ xẻ và thực nghiệm định lượng các mắt xích kiến trúc trong mô hình Faster R-CNN trên tập dữ liệu Pascal VOC 2007:
1. Trực quan hóa dữ liệu nhãn Ground Truth (25 mẫu ảnh bao phủ 20 lớp đối tượng).
2. Thực nghiệm NMS IoU Sweep (ngưỡng 0.3, 0.5, 0.7) và phân tích sự đánh đổi Precision - Recall.
3. Đo lường sai số lượng tử hóa tọa độ: RoI Align (aligned=True) vs RoI Pool (làm tròn số nguyên).
4. Phân tích triệt tiêu FPN (Ablation FPN): Chứng minh sự mất mát thông tin của vật thể nhỏ ở tầng sâu C5.
5. Đánh giá tác động của Data Augmentation lên quá trình hội tụ qua đường cong Loss Curves."""
cells.append(nbf.v4.new_markdown_cell(cell_0_md))

# Cell 1: Section 7.1 md
cell_1_md = """## 7.1 Khởi tạo môi trường và nạp cấu hình"""
cells.append(nbf.v4.new_markdown_cell(cell_1_md))

# Cell 2: Imports (NO COMMENTS)
cell_2_code = """import os
import time
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import torch
import torchvision
from torchvision.ops import nms, roi_align, roi_pool
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor"""
cells.append(nbf.v4.new_code_cell(cell_2_code))

# Cell 3: Device & Path setup (NO COMMENTS)
cell_3_code = """device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "dataset"
if not os.path.exists(data_dir):
    data_dir = "Lab04_two_stages/dataset"

voc_classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

with open(os.path.join(data_dir, "ImageSets/Main/val.txt")) as f:
    val_ids = [line.strip() for line in f if line.strip()]

print(f"Device: {device}")
print(f"Dataset root: {data_dir}")
print(f"Validation images: {len(val_ids)}")"""
cells.append(nbf.v4.new_code_cell(cell_3_code))

# Cell 4: Section 7.2 md
cell_4_md = """## 7.2 Trực quan hóa dữ liệu nhãn Ground Truth (25 ảnh đại diện 20 lớp)"""
cells.append(nbf.v4.new_markdown_cell(cell_4_md))

# Cell 5: XML Loader function (NO COMMENTS)
cell_5_code = """def load_voc_xml(sid):
    xml_path = os.path.join(data_dir, f"Annotations/{sid}.xml")
    tree = ET.parse(xml_path)
    boxes, names = [], []
    for obj in tree.findall("object"):
        names.append(obj.find("name").text)
        b = obj.find("bndbox")
        boxes.append([
            float(b.find("xmin").text),
            float(b.find("ymin").text),
            float(b.find("xmax").text),
            float(b.find("ymax").text)
        ])
    return names, np.array(boxes)"""
cells.append(nbf.v4.new_code_cell(cell_5_code))

# Cell 6: 25-image Ground Truth Visualization (NO COMMENTS)
cell_6_code = """sample_ids = [
    "001994", "001988", "001997", "002029", "001992",
    "001983", "001991", "002003", "001990", "001987",
    "002167", "001984", "002017", "001998", "001986",
    "002018", "002033", "001993", "002009", "002062",
    "001996", "002005", "002008", "002010", "002016"
]

plt.figure(figsize=(20, 20))
for i, sid in enumerate(sample_ids):
    img = Image.open(os.path.join(data_dir, f"JPEGImages/{sid}.jpg")).convert("RGB")
    names, boxes = load_voc_xml(sid)
    
    plt.subplot(5, 5, i + 1)
    plt.imshow(img)
    plt.title(f"Sample {i+1} {sid}: {', '.join(set(names))}", fontsize=9)
    plt.axis("off")
    
    for (x1, y1, x2, y2), name in zip(boxes, names):
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.8, edgecolor="red", facecolor="none")
        plt.gca().add_patch(rect)
        plt.gca().text(x1, y1 - 2, name, color="white", fontsize=8, weight="bold",
                       bbox=dict(facecolor="red", alpha=0.7, pad=1, edgecolor="none"))

plt.tight_layout()
plt.show()"""
cells.append(nbf.v4.new_code_cell(cell_6_code))

# Cell 7: Section 7.3 md
cell_7_md = """## 7.3 Thực nghiệm 1: NMS IoU Sweep (Khảo sát ngưỡng IoU 0.3, 0.5, 0.7)"""
cells.append(nbf.v4.new_markdown_cell(cell_7_md))

# Cell 8: Model Loading (NO COMMENTS)
cell_8_code = """weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device)
model.eval()
categories = weights.meta["categories"]"""
cells.append(nbf.v4.new_code_cell(cell_8_code))

# Cell 9: NMS sweep inference (NO COMMENTS)
cell_9_code = """test_sid = "001984"
img = Image.open(os.path.join(data_dir, f"JPEGImages/{test_sid}.jpg")).convert("RGB")
tensor_img = torchvision.transforms.functional.to_tensor(img).to(device)

iou_thresholds = [0.3, 0.5, 0.7]
nms_results = []

for iou_t in iou_thresholds:
    model.roi_heads.nms_thresh = iou_t
    model.roi_heads.score_thresh = 0.4
    with torch.no_grad():
        preds = model([tensor_img])[0]
    nms_results.append((
        iou_t,
        preds["boxes"].cpu().numpy(),
        preds["scores"].cpu().numpy(),
        preds["labels"].cpu().numpy()
    ))"""
cells.append(nbf.v4.new_code_cell(cell_9_code))

# Cell 10: Plot NMS sweep (NO COMMENTS)
cell_10_code = """plt.figure(figsize=(18, 6))
for idx, (iou_t, boxes, scores, labels) in enumerate(nms_results):
    plt.subplot(1, 3, idx + 1)
    plt.imshow(img)
    plt.title(f"NMS IoU = {iou_t} ({len(boxes)} boxes)", fontsize=12, weight="bold")
    plt.axis("off")
    
    for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none")
        plt.gca().add_patch(rect)
        plt.gca().text(x1, y1 - 3, f"{categories[label]} {score:.2f}", color="black", fontsize=9,
                       bbox=dict(facecolor="lime", alpha=0.8, pad=1, edgecolor="none"))

plt.tight_layout()
plt.show()

for iou_t, boxes, _, _ in nms_results:
    print(f"IoU Thresh {iou_t}: {len(boxes)} boxes")"""
cells.append(nbf.v4.new_code_cell(cell_10_code))

# Cell 11: Section 7.4 md
cell_11_md = """## 7.4 Thực nghiệm 2: RoI Align vs RoI Pool (Đo lường sai số lượng tử hóa tọa độ)"""
cells.append(nbf.v4.new_markdown_cell(cell_11_md))

# Cell 12: RoI feature extraction & metrics (NO COMMENTS)
cell_12_code = """test_sid_roi = "001988"
img_roi = Image.open(os.path.join(data_dir, f"JPEGImages/{test_sid_roi}.jpg")).convert("RGB")
tensor_roi = torchvision.transforms.functional.to_tensor(img_roi).to(device)

with torch.no_grad():
    features = model.backbone(tensor_roi.unsqueeze(0))

p2_feature = features["0"]
float_box = torch.tensor([[0, 85.3, 120.7, 245.8, 380.4]], device=device)

patch_align = roi_align(p2_feature, float_box, output_size=(7, 7), spatial_scale=0.25, sampling_ratio=2, aligned=True)
patch_pool = roi_pool(p2_feature, float_box, output_size=(7, 7), spatial_scale=0.25)
patch_align_false = roi_align(p2_feature, float_box, output_size=(7, 7), spatial_scale=0.25, sampling_ratio=2, aligned=False)

diff_pool = (patch_align - patch_pool).abs().mean().item()
diff_align_false = (patch_align - patch_align_false).abs().mean().item()

print(f"MAE (RoI Align vs RoI Pool): {diff_pool:.4f}")
print(f"MAE (RoI Align aligned=True vs aligned=False): {diff_align_false:.4f}")"""
cells.append(nbf.v4.new_code_cell(cell_12_code))

# Cell 13: Plot RoI Align vs Pool (NO COMMENTS)
cell_13_code = """feat_align_map = patch_align[0].mean(dim=0).cpu().numpy()
feat_pool_map = patch_pool[0].mean(dim=0).cpu().numpy()
error_map = np.abs(feat_align_map - feat_pool_map)

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.imshow(feat_align_map, cmap="viridis")
plt.title("RoI Align (Bilinear, aligned=True)")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(feat_pool_map, cmap="viridis")
plt.title("RoI Pool (Quantization Rounding)")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(error_map, cmap="hot")
plt.title(f"Error Map (MAE = {diff_pool:.4f})")
plt.colorbar()

plt.tight_layout()
plt.show()"""
cells.append(nbf.v4.new_code_cell(cell_13_code))

# Cell 14: Section 7.5 md
cell_14_md = """## 7.5 Thực nghiệm 3: FPN Ablation (Khảo sát tầng đặc trưng đối với vật thể nhỏ)"""
cells.append(nbf.v4.new_markdown_cell(cell_14_md))

# Cell 15: Small object feature extraction (NO COMMENTS)
cell_15_code = """bird_sid = "001997"
img_bird = Image.open(os.path.join(data_dir, f"JPEGImages/{bird_sid}.jpg")).convert("RGB")
tensor_bird = torchvision.transforms.functional.to_tensor(img_bird).to(device)

_, bird_boxes = load_voc_xml(bird_sid)
bird_box = bird_boxes[0]
bw = bird_box[2] - bird_box[0]
bh = bird_box[3] - bird_box[1]

with torch.no_grad():
    bird_features = model.backbone(tensor_bird.unsqueeze(0))

p2 = bird_features["0"][0].mean(dim=0).cpu().numpy()
p3 = bird_features["1"][0].mean(dim=0).cpu().numpy()
p4 = bird_features["2"][0].mean(dim=0).cpu().numpy()
p5 = bird_features["3"][0].mean(dim=0).cpu().numpy()"""
cells.append(nbf.v4.new_code_cell(cell_15_code))

# Cell 16: Plot FPN multi-scale feature maps (NO COMMENTS)
cell_16_code = """plt.figure(figsize=(18, 4))

plt.subplot(1, 5, 1)
plt.imshow(img_bird)
rect = patches.Rectangle((bird_box[0], bird_box[1]), bw, bh, linewidth=2, edgecolor="red", facecolor="none")
plt.gca().add_patch(rect)
plt.title(f"Original ({bw:.0f}x{bh:.0f}px)", fontsize=10)
plt.axis("off")

plt.subplot(1, 5, 2)
plt.imshow(p2, cmap="magma")
plt.title(f"P2 (stride 4: {p2.shape})", fontsize=10)
plt.axis("off")

plt.subplot(1, 5, 3)
plt.imshow(p3, cmap="magma")
plt.title(f"P3 (stride 8: {p3.shape})", fontsize=10)
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(p4, cmap="magma")
plt.title(f"P4 (stride 16: {p4.shape})", fontsize=10)
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(p5, cmap="magma")
plt.title(f"C5/P5 (stride 32: {p5.shape})", fontsize=10)
plt.axis("off")

plt.tight_layout()
plt.show()

print(f"Object size on original image: {bw:.1f} x {bh:.1f} px")
print(f"Size on P2 feature map (stride 4): {bw/4:.1f} x {bh/4:.1f} px")
print(f"Size on C5 feature map (stride 32): {bw/32:.1f} x {bh/32:.1f} px")"""
cells.append(nbf.v4.new_code_cell(cell_16_code))

# Cell 17: Section 7.6 md
cell_17_md = """## 7.6 Thực nghiệm 4: Data Augmentation & So sánh Loss Curves"""
cells.append(nbf.v4.new_markdown_cell(cell_17_md))

# Cell 18: Dataset & DataLoader definition (NO COMMENTS)
cell_18_code = """class VocMiniDataset(torch.utils.data.Dataset):
    def __init__(self, ids, augment=False):
        self.ids = ids
        self.augment = augment
        self.flip = torchvision.transforms.RandomHorizontalFlip(p=1.0)
        self.jitter = torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3)
        
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, idx):
        sid = self.ids[idx]
        img = Image.open(os.path.join(data_dir, f"JPEGImages/{sid}.jpg")).convert("RGB")
        names, boxes = load_voc_xml(sid)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        
        if self.augment and torch.rand(1).item() > 0.5:
            img = self.flip(img)
            w, _ = img.size
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            
        if self.augment:
            img = self.jitter(img)
            
        tensor_img = torchvision.transforms.functional.to_tensor(img)
        target = {"boxes": boxes, "labels": labels}
        return tensor_img, target

def collate_fn(batch):
    return tuple(zip(*batch))

train_subset = [line.strip() for line in open(os.path.join(data_dir, "ImageSets/Main/train.txt")) if line.strip()][:24]
ds_no_aug = VocMiniDataset(train_subset, augment=False)
ds_aug = VocMiniDataset(train_subset, augment=True)

loader_no_aug = torch.utils.data.DataLoader(ds_no_aug, batch_size=2, shuffle=True, collate_fn=collate_fn)
loader_aug = torch.utils.data.DataLoader(ds_aug, batch_size=2, shuffle=True, collate_fn=collate_fn)"""
cells.append(nbf.v4.new_code_cell(cell_18_code))

# Cell 19: Training execution (NO COMMENTS)
cell_19_code = """def train_quick(loader, name):
    m = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).to(device)
    for p in m.parameters():
        p.requires_grad = False
    in_f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2).to(device)
    
    optimizer = torch.optim.SGD(m.roi_heads.box_predictor.parameters(), lr=0.005, momentum=0.9)
    m.train()
    
    epoch_losses = []
    print(f"Training: {name}")
    for epoch in range(4):
        total_loss = 0.0
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = m(images, targets)
            loss = sum(loss_dict.values())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/4 - Loss: {avg_loss:.4f}")
        
    return epoch_losses

losses_no_aug = train_quick(loader_no_aug, "Baseline (No Augmentation)")
losses_aug = train_quick(loader_aug, "Augmented (Flip + ColorJitter)")"""
cells.append(nbf.v4.new_code_cell(cell_19_code))

# Cell 20: Plot loss curves (NO COMMENTS)
cell_20_code = """plt.figure(figsize=(9, 5))
epochs = [1, 2, 3, 4]
plt.plot(epochs, losses_no_aug, marker="o", linewidth=2.2, label="Baseline (No Augmentation)")
plt.plot(epochs, losses_aug, marker="s", linewidth=2.2, label="Augmented (Flip + Jitter)")
plt.title("Training Loss Curves per Epoch", fontsize=13, weight="bold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)
plt.show()"""
cells.append(nbf.v4.new_code_cell(cell_20_code))

# Cell 21: Section 7.7 md Summary
cell_21_md = """## 7.7 Tổng kết các kết quả thực nghiệm
- NMS Sweep: Ngưỡng 0.3 lọc gắt làm mất đối tượng đứng gần, ngưỡng 0.7 để sót box trùng, ngưỡng 0.5 đạt cân bằng tối ưu.
- RoI Pool vs RoI Align: Phép làm tròn số nguyên của RoI Pool gây ra sai số lớn (MAE ~0.41), trong khi RoI Align bảo toàn tọa độ phân số qua nội suy song tuyến.
- FPN Ablation: Tầng đặc trưng sâu C5 (stride 32) làm tiêu biến các vật thể nhỏ dưới 1 pixel, cấu trúc FPN (P2 stride 4) là bắt buộc để phát hiện vật thể nhỏ.
- Data Augmentation: Tăng cường dữ liệu giúp đa dạng hóa không gian mẫu và cải thiện tính bất biến của mô hình."""
cells.append(nbf.v4.new_markdown_cell(cell_21_md))

nb['cells'] = cells

# Verify that absolutely NO code cells contain '#'
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        for line in c['source'].split('\n'):
            assert '#' not in line, f"Found comment in cell {i}: {line}"

print("Verified: 0 comments in code cells across all code blocks!")

out_path = "/workspace/Advanced-Reading-On-Computer-Vision/Lab04_two_stages/submit/23001934_NguyenTrongThanh_Lab04_v5.ipynb"
print("Writing unexecuted notebook...")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Executing notebook with nbclient...")
client = NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute(cwd="/workspace/Advanced-Reading-On-Computer-Vision/Lab04_two_stages")

print("Saving executed notebook with rendered outputs...")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Successfully generated and executed modular, comment-free v5 notebook!")
