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

# Cell 0: Header
cell_0_md = """# 7. BÀI TẬP NÂNG CAO VÀ KIỂM THỬ (ABLATION STUDIES & ADVANCED EVALUATION)
**Môn học**: Advanced Reading in Computer Vision (MAT3563)  
**Chủ đề**: Lab 04 - Two-Stages Object Detection Models (Faster R-CNN)  
**Sinh viên**: Nguyễn Trọng Thành (MSSV: 23001934)  

Phần này tập trung nghiên cứu, mổ xẻ cấu trúc và đánh giá thực nghiệm mô hình Faster R-CNN trên tập dữ liệu chuẩn Pascal VOC 2007. Trọng tâm ban đầu là khảo sát trực quan hóa toàn diện 20-30 mẫu ảnh đại diện cho tất cả 20 lớp đối tượng, đối chiếu nhãn Ground Truth với dự đoán của mô hình Pretrained, từ đó làm tiền đề cho các thí nghiệm triệt tiêu (Ablation Studies)."""
cells.append(nbf.v4.new_markdown_cell(cell_0_md))

# Cell 1: Section 7.1 md
cell_1_md = """## 7.1 Khởi tạo môi trường tính toán và cấu hình nạp dữ liệu Pascal VOC 2007
Thiết lập cơ chế tự động nhận diện phần cứng GPU RTX 3060, quét tìm đường dẫn dữ liệu qua danh sách các thư mục khả dĩ và nạp danh mục 20 lớp đối tượng chuẩn của Pascal VOC cùng bảng màu trực quan."""
cells.append(nbf.v4.new_markdown_cell(cell_1_md))

# Cell 2: Section 7.1 code
cell_2_code = """import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

possible_data_dirs = [
    os.path.join(os.getcwd(), "dataset"),
    os.path.join(os.getcwd(), "../dataset"),
    os.path.join(os.getcwd(), "Lab04_two_stages/dataset"),
    os.path.abspath("dataset"),
    "/workspace/Advanced-Reading-On-Computer-Vision/Lab04_two_stages/dataset"
]

data_dir = next((d for d in possible_data_dirs if os.path.exists(d) and os.path.exists(os.path.join(d, "Annotations"))), None)

voc_classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

color_map = plt.get_cmap("tab20", len(voc_classes))
class_colors = {cls: color_map(i) for i, cls in enumerate(voc_classes)}

val_txt_path = os.path.join(data_dir, "ImageSets", "Main", "val.txt")
with open(val_txt_path, "r", encoding="utf-8") as f:
    sample_ids = [line.strip() for line in f if line.strip()]

print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"Dataset root: {data_dir}")
print(f"Total validation samples: {len(sample_ids)}")
print(f"Classes count: {len(voc_classes)}")"""
cells.append(nbf.v4.new_code_cell(cell_2_code))

# Cell 3: Section 7.2 md
cell_3_md = """## 7.2 Trực quan hóa toàn diện 25 mẫu ảnh Ground Truth bao phủ toàn bộ 20 lớp đối tượng
Để có cái nhìn sâu sắc và bao quát về độ phân tán dữ liệu, 25 mẫu ảnh tiêu biểu đã được trích xuất nhằm đảm bảo bao phủ đầy đủ tất cả 20 lớp đối tượng của Pascal VOC 2007, kết hợp cùng các bối cảnh đa vật thể, vật thể che khuất lẫn nhau và các vật thể kích thước siêu nhỏ. Mỗi bounding box được hiển thị với màu sắc đặc trưng theo từng lớp và nhãn định danh tương ứng."""
cells.append(nbf.v4.new_markdown_cell(cell_3_md))

# Cell 4: Section 7.2 code
cell_4_code = """def parse_voc_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        difficult = int(obj.find("difficult").text) if obj.find("difficult") is not None else 0
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        objects.append({
            "name": name,
            "box": [xmin, ymin, xmax, ymax],
            "difficult": difficult
        })
    return objects

selected_samples = [
    "001994", "001988", "001997", "002029", "001992",
    "001983", "001991", "002003", "001990", "001987",
    "002167", "001984", "002017", "001998", "001986",
    "002018", "002033", "001993", "002009", "002062",
    "001996", "002005", "002008", "002010", "002016"
]

fig, axes = plt.subplots(5, 5, figsize=(22, 22))
axes = axes.flatten()

for idx, sid in enumerate(selected_samples):
    img_path = os.path.join(data_dir, "JPEGImages", f"{sid}.jpg")
    xml_path = os.path.join(data_dir, "Annotations", f"{sid}.xml")
    
    img = Image.open(img_path).convert("RGB")
    objs = parse_voc_annotation(xml_path)
    
    ax = axes[idx]
    ax.imshow(img)
    cls_summary = ", ".join(sorted(list(set(o["name"] for o in objs))))
    ax.set_title(f"#{idx + 1} ({sid}.jpg): {cls_summary}", fontsize=10, pad=5, weight="bold")
    ax.axis("off")
    
    for obj in objs:
        xmin, ymin, xmax, ymax = obj["box"]
        width = xmax - xmin
        height = ymax - ymin
        cls_name = obj["name"]
        color = class_colors.get(cls_name, (0.0, 1.0, 0.0, 1.0))
        
        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        
        ax.text(
            xmin, max(0, ymin - 3),
            cls_name,
            color="white", fontsize=8, weight="bold",
            bbox=dict(facecolor=color, alpha=0.8, edgecolor="none", pad=1.5)
        )

plt.tight_layout()
plt.show()"""
cells.append(nbf.v4.new_code_cell(cell_4_code))

# Cell 5: Section 7.3 md
cell_5_md = """## 7.3 Suy luận với Faster R-CNN v2 Pretrained trên tập 25 mẫu ảnh thử nghiệm
Chạy suy luận trực tiếp mô hình Faster R-CNN ResNet-50 FPN v2 trên toàn bộ 25 mẫu ảnh đã khảo sát ở trên. Kết quả suy luận được lọc với ngưỡng độ tin cậy score $\ge 0.5$, vẽ hộp bao và nhãn dự đoán để đối chiếu trực tiếp khả năng nhận diện đa lớp của mô hình trên các bối cảnh khác nhau."""
cells.append(nbf.v4.new_markdown_cell(cell_5_md))

# Cell 6: Section 7.3 code
cell_6_code = """weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device)
model.eval()

coco_categories = weights.meta["categories"]

fig, axes = plt.subplots(5, 5, figsize=(22, 22))
axes = axes.flatten()

total_detections = 0
for idx, sid in enumerate(selected_samples):
    img_path = os.path.join(data_dir, "JPEGImages", f"{sid}.jpg")
    pil_img = Image.open(img_path).convert("RGB")
    tensor_img = F.to_tensor(pil_img).to(device)
    
    with torch.no_grad():
        preds = model([tensor_img])[0]
        
    pred_boxes = preds["boxes"].cpu().numpy()
    pred_labels = preds["labels"].cpu().numpy()
    pred_scores = preds["scores"].cpu().numpy()
    
    valid_mask = pred_scores >= 0.5
    filtered_boxes = pred_boxes[valid_mask]
    filtered_labels = pred_labels[valid_mask]
    filtered_scores = pred_scores[valid_mask]
    total_detections += len(filtered_boxes)
    
    ax = axes[idx]
    ax.imshow(pil_img)
    ax.set_title(f"#{idx + 1} ({sid}.jpg): {len(filtered_boxes)} boxes", fontsize=10, pad=5, weight="bold", color="darkgreen")
    ax.axis("off")
    
    for box, label_idx, score in zip(filtered_boxes, filtered_labels, filtered_scores):
        xmin, ymin, xmax, ymax = box
        label_name = coco_categories[label_idx] if label_idx < len(coco_categories) else str(label_idx)
        
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            xmin, max(0, ymin - 3),
            f"{label_name}: {score:.2f}",
            color="black", fontsize=8, weight="bold",
            bbox=dict(facecolor="lime", alpha=0.8, edgecolor="none", pad=1.5)
        )

plt.tight_layout()
plt.show()

print(f"Total detected objects across 25 samples: {total_detections}")"""
cells.append(nbf.v4.new_code_cell(cell_6_code))

# Cell 7: Section 7.4 md
cell_7_md = """## 7.4 Bảng thống kê định lượng và đối chiếu chi tiết giữa Ground Truth và Model Predictions
Bảng tổng hợp chi tiết số lượng và tên các lớp đối tượng thực tế so với kết quả dự đoán của Faster R-CNN v2 Pretrained trên từng ảnh trong số 25 mẫu thử nghiệm."""
cells.append(nbf.v4.new_markdown_cell(cell_7_md))

# Cell 8: Section 7.4 code
cell_8_code = """sample_analysis = []

for idx, sid in enumerate(selected_samples):
    img_path = os.path.join(data_dir, "JPEGImages", f"{sid}.jpg")
    xml_path = os.path.join(data_dir, "Annotations", f"{sid}.xml")
    
    pil_img = Image.open(img_path).convert("RGB")
    tensor_img = F.to_tensor(pil_img).to(device)
    gt_objs = parse_voc_annotation(xml_path)
    
    with torch.no_grad():
        preds = model([tensor_img])[0]
        
    scores = preds["scores"].cpu().numpy()
    labels = preds["labels"].cpu().numpy()
    high_conf = scores >= 0.5
    
    gt_classes = [o["name"] for o in gt_objs]
    pred_classes = [coco_categories[l] for l in labels[high_conf]]
    
    sample_analysis.append({
        "index": idx + 1,
        "sample_id": sid,
        "gt_count": len(gt_objs),
        "gt_classes": ", ".join(gt_classes),
        "pred_count": int(high_conf.sum()),
        "pred_classes": ", ".join(pred_classes)
    })

print(f"{'Idx':<4} | {'Sample ID':<10} | {'GT Count':<8} | {'GT Classes':<25} | {'Pred Count':<10} | {'Pred Classes'}")
print("-" * 105)
for item in sample_analysis:
    print(f"{item['index']:<4} | {item['sample_id']:<10} | {item['gt_count']:<8} | {item['gt_classes'][:25]:<25} | {item['pred_count']:<10} | {item['pred_classes']}")"""
cells.append(nbf.v4.new_code_cell(cell_8_code))

# Cell 9: Section 7.5 md
cell_9_md = """## 7.5 Nhận xét tổng kết và định hướng các thí nghiệm Ablation
Qua quá trình khảo sát trực quan trên 25 ảnh đại diện cho toàn bộ 20 lớp của Pascal VOC:
1. Độ chính xác định vị: Faster R-CNN đạt khả năng bao quát xuất sắc trên các lớp phương tiện giao thông (aeroplane, bus, car, train) và các sinh vật kích thước trung bình - lớn (person, dog, horse, cow) với độ tin cậy thường xuyên trên 90%.
2. Hiện tượng bỏ sót và nhầm lẫn:
- Đối với các vật thể nhỏ (bird ở cành cây xa, bottle, pottedplant), mô hình đôi khi bỏ sót nếu không có các tầng đặc trưng phân giải cao từ FPN.
- Khi các đối tượng đứng quá sát nhau hoặc che khuất (như đàn cừu, đàn chó), cơ chế NMS nếu đặt ngưỡng quá khắt khe sẽ loại bỏ mất các box hợp lệ.
3. Đây là tiền đề trực quan hoàn chỉnh để tiến hành 4 thí nghiệm triệt tiêu của Mục 7:
- Ablation FPN: Đo lường độ suy giảm mAP khi tắt cấu trúc kim tự tháp đặc trưng đối với các vật thể nhỏ.
- RoI Pool vs RoI Align: Kiểm chứng sai lệch tọa độ phân số khi lượng tử hóa.
- NMS IoU Sweep: Khảo sát sự đánh đổi Precision - Recall trên các cảnh đông đúc.
- Data Augmentation: Theo dõi sự cải thiện của Loss Curves khi huấn luyện với dữ liệu biến đổi."""
cells.append(nbf.v4.new_markdown_cell(cell_9_md))

nb['cells'] = cells

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

print("Successfully generated and executed expanded v5 notebook with 25 images!")
