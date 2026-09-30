import os
import time
import math
from collections import OrderedDict
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD, SSDHead
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.ops import box_iou

DATA_DIR = "/root/dataset_drone"
RUNS_DIR = "/root/runs_ssd_resnet50"
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(os.path.join(RUNS_DIR, "weights"), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

class DroneSSDDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, lbl_dir):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.img_paths = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"] and p.stat().st_size > 0
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_p = self.img_paths[idx]
        lbl_p = self.lbl_dir / f"{img_p.stem}.txt"
        try:
            pil_img = Image.open(img_p).convert("RGB")
        except Exception:
            pil_img = Image.new("RGB", (320, 320), (0, 0, 0))
        w, h = pil_img.size
        img_tensor = TF.to_tensor(pil_img)

        boxes = []
        labels = []
        if lbl_p.exists():
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = max(0.0, (cx - bw / 2.0) * w)
                        y1 = max(0.0, (cy - bh / 2.0) * h)
                        x2 = min(float(w), (cx + bw / 2.0) * w)
                        y2 = min(float(h), (cy + bh / 2.0) * h)
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(1)

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx])
        }
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

print("Loading datasets...")
train_dataset = DroneSSDDataset(f"{DATA_DIR}/images/train", f"{DATA_DIR}/labels/train")
val_dataset = DroneSSDDataset(f"{DATA_DIR}/images/val", f"{DATA_DIR}/labels/val")
test_dataset = DroneSSDDataset(f"{DATA_DIR}/images/test", f"{DATA_DIR}/labels/test")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=2)

print(f"Dataset ready: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

class ResNetBackbone(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.body = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4
        )
        self.lateral3 = nn.Conv2d(512, out_channels, 1)
        self.lateral4 = nn.Conv2d(1024, out_channels, 1)
        self.lateral5 = nn.Conv2d(2048, out_channels, 1)
        self.extra6 = Conv2dNormActivation(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.extra7 = Conv2dNormActivation(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        c3, c4, c5 = None, None, None
        for i, m in enumerate(self.body):
            x = m(x)
            if i == 5:
                c3 = x
            elif i == 6:
                c4 = x
            elif i == 7:
                c5 = x
        p3 = self.lateral3(c3)
        p4 = self.lateral4(c4)
        p5 = self.lateral5(c5)
        p6 = self.extra6(p5)
        p7 = self.extra7(p6)
        return OrderedDict([
            ("0", p3),
            ("1", p4),
            ("2", p5),
            ("3", p6),
            ("4", p7),
        ])

print("Building SSD ResNet-50 model...")
backbone = ResNetBackbone(out_channels=256)
box_generator = DefaultBoxGenerator(
    aspect_ratios=[[2, 3, 1/2], [2, 3, 1/2], [2, 3, 1/2], [2, 3, 1/2], [2, 3, 1/2]]
)
num_classes = 2
head = SSDHead(
    in_channels=[256, 256, 256, 256, 256],
    num_anchors=box_generator.num_anchors_per_location(),
    num_classes=num_classes
)

def init_head(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

head.apply(init_head)

model = SSD(
    backbone=backbone,
    anchor_generator=box_generator,
    size=(320, 320),
    num_classes=num_classes,
    head=head
).to(device)

total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"SSD ResNet-50 successfully constructed: {total_params:.2f}M parameters")

for param in model.backbone.body.parameters():
    param.requires_grad = False

head_params = (
    list(model.head.parameters()) +
    list(model.backbone.lateral3.parameters()) +
    list(model.backbone.lateral4.parameters()) +
    list(model.backbone.lateral5.parameters()) +
    list(model.backbone.extra6.parameters()) +
    list(model.backbone.extra7.parameters())
)

optimizer = torch.optim.SGD(head_params, lr=1e-3, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)

NUM_EPOCHS = 30
history = []
best_val_loss = float("inf")

print(f"Starting extensive fine-tuning ({NUM_EPOCHS} epochs) on RTX 3060...")

for epoch in range(1, NUM_EPOCHS + 1):
    t_start = time.time()

    if epoch == 6:
        print(">>> Epoch 6: Unfreezing ResNet-50 layer3 and layer4 for differential fine-tuning...")
        for param in model.backbone.body[6].parameters():
            param.requires_grad = True
        for param in model.backbone.body[7].parameters():
            param.requires_grad = True
        optimizer = torch.optim.SGD([
            {"params": [p for p in model.backbone.body.parameters() if p.requires_grad], "lr": 1e-4},
            {"params": head_params, "lr": 1e-3}
        ], momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)

    model.train()
    train_loss = 0.0
    train_cls = 0.0
    train_reg = 0.0
    num_batches = 0

    for images, targets in train_loader:
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss_cls = loss_dict["classification"]
        loss_reg = loss_dict["bbox_regression"]
        loss_total = loss_cls + loss_reg

        if not torch.isnan(loss_total) and not torch.isinf(loss_total):
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            train_loss += loss_total.item()
            train_cls += loss_cls.item()
            train_reg += loss_reg.item()
            num_batches += 1

    scheduler.step()
    avg_train_loss = train_loss / max(1, num_batches)
    avg_train_cls = train_cls / max(1, num_batches)
    avg_train_reg = train_reg / max(1, num_batches)

    # Validation loss
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss_total = loss_dict["classification"] + loss_dict["bbox_regression"]
            if not torch.isnan(loss_total):
                val_loss += loss_total.item()
                val_batches += 1
    avg_val_loss = val_loss / max(1, val_batches)
    epoch_time = time.time() - t_start

    print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] ({epoch_time:.1f}s) - Train Loss: {avg_train_loss:.4f} (Cls: {avg_train_cls:.4f}, Reg: {avg_train_reg:.4f}) | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f"{RUNS_DIR}/weights/ssd_resnet50_best.pt")
        print(f"  --> Saved new best model (Val Loss: {best_val_loss:.4f})")

    torch.save(model.state_dict(), f"{RUNS_DIR}/weights/ssd_resnet50_last.pt")

    history.append({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "train_cls": avg_train_cls,
        "train_reg": avg_train_reg,
        "val_loss": avg_val_loss,
        "time_sec": epoch_time
    })

# Save results CSV
df_history = pd.DataFrame(history)
df_history.to_csv(f"{RUNS_DIR}/ssd_resnet50_results.csv", index=False)
print("Saved training metrics to ssd_resnet50_results.csv")

# Plot Loss Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
ax1.plot(df_history["epoch"], df_history["train_loss"], marker="o", color="crimson", linewidth=2.5, label="Train Loss")
ax1.plot(df_history["epoch"], df_history["val_loss"], marker="s", color="royalblue", linewidth=2.2, label="Val Loss")
ax1.set_title("SSD ResNet-50: Total Train vs Val Loss (30 Epochs)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Loss", fontsize=11)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(fontsize=10)

ax2.plot(df_history["epoch"], df_history["train_cls"], marker="^", color="forestgreen", linewidth=2, label="Cls Loss")
ax2.plot(df_history["epoch"], df_history["train_reg"], marker="d", color="darkorange", linewidth=2, label="Reg Loss")
ax2.set_title("SSD ResNet-50: Classification vs Box Regression Loss", fontsize=12, fontweight="bold")
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Loss", fontsize=11)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig(f"{RUNS_DIR}/ssd_resnet50_loss_curve.png", dpi=200)
plt.close()

# Evaluate on Test Set & Compute Quantitative Detection Metrics
print("Evaluating trained model on Test Split (200 images)...")
model.load_state_dict(torch.load(f"{RUNS_DIR}/weights/ssd_resnet50_best.pt"))
model.eval()

vis_images = []
total_tp = 0
total_fp = 0
total_fn = 0
inference_times = []

with torch.no_grad():
    for images, targets in test_loader:
        images_dev = [im.to(device) for im in images]
        t0 = time.time()
        predictions = model(images_dev)
        t_infer = (time.time() - t0) / len(images)
        inference_times.append(t_infer)

        for i in range(len(predictions)):
            pred = predictions[i]
            tgt = targets[i]
            pil_img = TF.to_pil_image(images[i].cpu())
            draw = ImageDraw.Draw(pil_img)

            gt_boxes = tgt["boxes"].cpu()
            p_boxes = pred["boxes"].cpu()
            p_scores = pred["scores"].cpu()

            # Filter confident detections
            conf_mask = p_scores >= 0.25
            filtered_boxes = p_boxes[conf_mask]
            filtered_scores = p_scores[conf_mask]

            # IoU matching for evaluation metrics
            if len(gt_boxes) > 0 and len(filtered_boxes) > 0:
                ious = box_iou(filtered_boxes, gt_boxes)
                matched_gt = set()
                tp = 0
                for det_idx in range(len(filtered_boxes)):
                    best_iou, gt_idx = ious[det_idx].max(dim=0)
                    if best_iou >= 0.5 and gt_idx.item() not in matched_gt:
                        tp += 1
                        matched_gt.add(gt_idx.item())
                fp = len(filtered_boxes) - tp
                fn = len(gt_boxes) - len(matched_gt)
                total_tp += tp
                total_fp += fp
                total_fn += fn
            elif len(gt_boxes) > 0 and len(filtered_boxes) == 0:
                total_fn += len(gt_boxes)
            elif len(gt_boxes) == 0 and len(filtered_boxes) > 0:
                total_fp += len(filtered_boxes)

            # Draw Ground Truth in Green
            for b in gt_boxes.numpy():
                draw.rectangle(list(map(int, b)), outline=(0, 255, 0), width=3)

            # Draw Predictions in Red
            for b, s in zip(filtered_boxes.numpy(), filtered_scores.numpy()):
                x1, y1, x2, y2 = map(int, b)
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
                draw.text((x1, max(0, y1 - 10)), f"Drone: {s:.2f}", fill=(255, 255, 0))

            if len(gt_boxes) > 0 and len(vis_images) < 4:
                vis_images.append(pil_img)

precision = total_tp / max(1, total_tp + total_fp)
recall = total_tp / max(1, total_tp + total_fn)
f1_score = 2 * precision * recall / max(1e-6, precision + recall)
mean_latency_ms = (sum(inference_times) / max(1, len(inference_times))) * 1000

print(f"=== SSD ResNet-50 Test Metrics (200 images) ===")
print(f"Total GT Drones: {total_tp + total_fn}")
print(f"True Positives (IoU>=0.5): {total_tp}, False Positives: {total_fp}, False Negatives: {total_fn}")
print(f"Precision@0.5: {precision*100:.2f}%")
print(f"Recall@0.5:    {recall*100:.2f}%")
print(f"F1-Score:      {f1_score:.4f}")
print(f"Mean Latency:  {mean_latency_ms:.1f} ms/image")

with open(f"{RUNS_DIR}/test_metrics.txt", "w") as f:
    f.write(f"Precision@0.5: {precision*100:.2f}%\n")
    f.write(f"Recall@0.5: {recall*100:.2f}%\n")
    f.write(f"F1-Score: {f1_score:.4f}\n")
    f.write(f"Latency_ms: {mean_latency_ms:.1f}\n")

fig, axes = plt.subplots(2, 2, figsize=(14, 14))
for idx, ax in enumerate(axes.flat):
    if idx < len(vis_images):
        ax.imshow(vis_images[idx])
        ax.set_title(f"Test Sample {idx+1} (Green=GT, Red=Pred SSD ResNet-50)", fontsize=11, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
plt.savefig(f"{RUNS_DIR}/ssd_test_detections.jpg", dpi=200)
plt.close()

print(f"All training artifacts and evaluation images saved in {RUNS_DIR}!")
print("COMPLETE!")
