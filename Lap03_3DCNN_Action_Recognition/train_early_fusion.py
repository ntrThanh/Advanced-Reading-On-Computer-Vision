import os
import sys
import time
import json
import random
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

data_root = "./dataset/UCF50"
output_dir = "./outputs_ucf50_early_fusion"
flow_cache_dir = "./flow_cache"
os.makedirs(output_dir, exist_ok=True)

img_size = 224
batch_size = 16
num_workers = 4
val_split = 0.2
flow_stack = 5
num_epochs = 30
learning_rate = 0.001
weight_decay = 0.0001

classes = sorted(os.listdir(data_root))
classes = [c for c in classes if os.path.isdir(os.path.join(data_root, c)) and not c.startswith(".")]
class_to_idx = {c: i for i, c in enumerate(classes)}

train_list, val_list = [], []
for c in classes:
    cdir = os.path.join(data_root, c)
    vids = sorted(os.listdir(cdir))
    vids = [v for v in vids if v.endswith(".avi")]
    random.shuffle(vids)
    split_pt = int(len(vids) * (1.0 - val_split))
    for v in vids[:split_pt]:
        train_list.append({"path": os.path.join(cdir, v), "label": class_to_idx[c], "label_name": c})
    for v in vids[split_pt:]:
        val_list.append({"path": os.path.join(cdir, v), "label": class_to_idx[c], "label_name": c})

print("Train videos:", len(train_list), "Val videos:", len(val_list))

def read_video_frames(path, target_indices, resize_hw=None):
    cap = cv2.VideoCapture(path)
    frames = []
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Cannot open: " + path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for ti in target_indices:
        idx = min(max(int(ti), 0), max(total - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            ok2, frame2 = cap.read()
            frame = frame2 if ok2 else (np.zeros((resize_hw[0], resize_hw[1], 3), dtype=np.uint8) if resize_hw else None)
        if resize_hw is not None and frame is not None:
            frame = cv2.resize(frame, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    return frames

def compute_flow_pair(prev_gray, next_gray):
    return cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def flow_to_uv_stack(frames_gray, flow_stack=5):
    T = len(frames_gray)
    if T < flow_stack + 1:
        while len(frames_gray) < flow_stack + 1:
            frames_gray.append(frames_gray[-1])
    uv_list = []
    for i in range(flow_stack):
        f0 = frames_gray[i]
        f1 = frames_gray[i + 1]
        flow = compute_flow_pair(f0, f1)
        uv_list.append(flow[..., 0].astype(np.float32))
        uv_list.append(flow[..., 1].astype(np.float32))
    uv = np.stack(uv_list, axis=0)
    uv = np.clip(uv, -20.0, 20.0) / 20.0
    return uv

def load_or_compute_flow_stack(video_path, center_idx, resize_hw, flow_stack, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{Path(video_path).stem}_f{center_idx}_s{flow_stack}_farneback_{resize_hw[0]}x{resize_hw[1]}.npy"
    cache_path = cache_dir / key
    if cache_path.exists():
        return np.load(str(cache_path)).astype(np.float32)
    frame_indices = [center_idx + i for i in range(flow_stack + 1)]
    frames = read_video_frames(video_path, frame_indices, resize_hw)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    uv = flow_to_uv_stack(grays, flow_stack=flow_stack)
    np.save(str(cache_path), uv.astype(np.float16))
    return uv

class UCF50TwoStreamDataset(Dataset):
    def __init__(self, items, img_size=224, flow_stack=5, flow_cache_dir=None, mode="train"):
        self.items = items
        self.H = img_size
        self.W = img_size
        self.flow_stack = flow_stack
        self.flow_cache_dir = flow_cache_dir
        self.mode = mode

        self.rgb_train_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.H, self.W)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.rgb_val_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.H, self.W)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.items)

    def _sample_center(self, vpath):
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            return 0, 1
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total <= 1:
            return 0, 1
        ci = min(max(total // 2, 1), max(total - self.flow_stack - 1, 1))
        return ci, total

    def __getitem__(self, idx):
        rec = self.items[idx]
        vpath = rec["path"]
        label = rec["label"]

        ci, total = self._sample_center(vpath)
        rgb_frame = read_video_frames(vpath, [ci], resize_hw=(self.H, self.W))[0]
        rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        rgb_t = self.rgb_train_tf(rgb) if self.mode == "train" else self.rgb_val_tf(rgb)

        uv = load_or_compute_flow_stack(vpath, ci, (self.H, self.W), self.flow_stack, self.flow_cache_dir)
        flow_t = torch.from_numpy(uv).float()
        return rgb_t, flow_t, label

train_ds = UCF50TwoStreamDataset(train_list, img_size=img_size, flow_stack=flow_stack, flow_cache_dir=flow_cache_dir, mode="train")
val_ds = UCF50TwoStreamDataset(val_list, img_size=img_size, flow_stack=flow_stack, flow_cache_dir=flow_cache_dir, mode="val")

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

def inflate_conv1_weight(conv1: nn.Conv2d, new_in_channels: int):
    old_w = conv1.weight.data
    out_c, in_c, kh, kw = old_w.shape
    if new_in_channels == in_c:
        return conv1
    new_w = torch.zeros((out_c, new_in_channels, kh, kw))
    for oc in range(out_c):
        for ic in range(new_in_channels):
            new_w[oc, ic] = old_w[oc, ic % in_c]
    conv1.in_channels = new_in_channels
    conv1.weight = nn.Parameter(new_w)
    return conv1

def build_backbone(name="mobilenet_v3_small", in_channels=3, pretrained=True):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    if in_channels != 3:
        model.features[0][0] = inflate_conv1_weight(model.features[0][0], in_channels)
    feat_dim = model.classifier[0].in_features
    model.classifier = nn.Identity()
    return model, feat_dim

class EarlyChannelResNet(nn.Module):
    def __init__(self, num_classes, in_channels, backbone_name="mobilenet_v3_small"):
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone_name, in_channels=in_channels)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)

flow_in_ch = 2 * flow_stack
model = EarlyChannelResNet(num_classes=len(classes), in_channels=3 + flow_in_ch, backbone_name="mobilenet_v3_small").to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for rgb, flow, y in loader:
        rgb = rgb.to(device, non_blocking=True)
        flow = flow.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        x = torch.cat([rgb, flow], dim=1)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            loss_sum += loss.item() * y.size(0)
    return loss_sum / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for rgb, flow, y in loader:
            rgb = rgb.to(device, non_blocking=True)
            flow = flow.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = torch.cat([rgb, flow], dim=1)
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            loss_sum += loss.item() * y.size(0)
    return loss_sum / total, correct / total

best_acc = -1.0
log_hist = []

print("Starting Early Fusion training...")
for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step()
    elapsed = time.time() - t0

    rec = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "lr": optimizer.param_groups[0]["lr"],
        "time_sec": round(elapsed, 2)
    }
    log_hist.append(rec)
    print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f} | time={elapsed:.2f}s")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
        with open(os.path.join(output_dir, "best_epoch.txt"), "w") as f:
            f.write(f"best_epoch={epoch}\nval_acc={val_acc:.4f}\n")

with open(os.path.join(output_dir, "train_log.json"), "w") as f:
    json.dump(log_hist, f, indent=2)

with open(os.path.join(output_dir, "labels_map.json"), "w") as f:
    json.dump(class_to_idx, f, indent=2)

print("Early Fusion training done. Best val acc:", best_acc)
