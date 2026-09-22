import os
import sys
import time
import json
import importlib
import nbformat as nbf
from nbclient import NotebookClient

work_dir = "/workspace/Advanced-Reading-On-Computer-Vision/Lap03_3DCNN_Action_Recognition"

def create_notebook(filename, title, model_module, model_name, ckpt_name, curve_img):
    nb = nbf.v4.new_notebook()

    header = f"""# {title}
Sinh vien: Nguyen Trong Thanh
MSSV: 23001934
Mon hoc: Advanced Reading on Computer Vision (MAT3563)
Dataset: UCF50
"""
    nb.cells.append(nbf.v4.new_markdown_cell(header))

    c1 = """import os
import sys
import time
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
"""
    nb.cells.append(nbf.v4.new_code_cell(c1))

    c2 = """from dataset import get_dataloaders

data_root = "./dataset/UCF50"
batch_size = 32
num_frames = 16
img_size = 112

train_loader, val_loader, class_to_idx = get_dataloaders(
    data_root=data_root,
    batch_size=batch_size,
    num_frames=num_frames,
    img_size=img_size,
    output_format="CTHW",
    split_ratio=0.8,
    num_workers=4,
    seed=42
)

idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(class_to_idx))]

print("Classes count:", len(class_names))
print("Train samples:", len(train_loader.dataset))
print("Val samples:", len(val_loader.dataset))

x_sample, y_sample = next(iter(val_loader))
print("Batch input shape:", x_sample.shape)
print("Batch target shape:", y_sample.shape)
"""
    nb.cells.append(nbf.v4.new_code_cell(c2))

    c3 = f"""from {model_module} import {model_name}

torch.backends.cudnn.benchmark = True
model = {model_name}(num_classes=len(class_names)).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)
"""
    nb.cells.append(nbf.v4.new_code_cell(c3))

    c4 = f"""checkpoint_path = "{ckpt_name}"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError("Checkpoint not found: " + checkpoint_path)

ckpt = torch.load(checkpoint_path, map_location=device)
state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
clean_state = {{(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}}
model.load_state_dict(clean_state)
model.eval()

model_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
print("Loaded checkpoint:", checkpoint_path)
print("Model size (MB):", round(model_size_mb, 2))
"""
    nb.cells.append(nbf.v4.new_code_cell(c4))

    c5 = """dummy = torch.randn(1, 3, num_frames, img_size, img_size, device=device)
dummy_batch = torch.randn(batch_size, 3, num_frames, img_size, img_size, device=device)

with torch.no_grad():
    for _ in range(20):
        with torch.amp.autocast("cuda"):
            _ = model(dummy)
if torch.cuda.is_available():
    torch.cuda.synchronize()

trials = 100
latencies = []
with torch.no_grad():
    for _ in range(trials):
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda"):
            _ = model(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

latencies = np.array(latencies)
mean_lat = float(np.mean(latencies))
std_lat = float(np.std(latencies))
median_lat = float(np.median(latencies))
p95_lat = float(np.percentile(latencies, 95))
p99_lat = float(np.percentile(latencies, 99))
fps = 1000.0 / mean_lat

batch_trials = 30
batch_latencies = []
with torch.no_grad():
    for _ in range(batch_trials):
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda"):
            _ = model(dummy_batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_latencies.append((time.perf_counter() - t0) * 1000.0)

mean_batch_lat = float(np.mean(batch_latencies))
std_batch_lat = float(np.std(batch_latencies))

print("Average latency per video (ms):", round(mean_lat, 2))
print("Latency std dev (ms):", round(std_lat, 2))
print("Median latency P50 (ms):", round(median_lat, 2))
print("Percentile P95 latency (ms):", round(p95_lat, 2))
print("Percentile P99 latency (ms):", round(p99_lat, 2))
print("Average batch latency (ms):", round(mean_batch_lat, 2))
print("Batch latency std dev (ms):", round(std_batch_lat, 2))
print("Throughput (videos/s):", round(fps, 1))
print("Throughput (frames/s):", round(fps * num_frames, 1))
"""
    nb.cells.append(nbf.v4.new_code_cell(c5))

    c6 = """criterion = nn.CrossEntropyLoss()
all_preds, all_probs, all_targets = [], [], []
top1_correct, top5_correct, total, val_loss_sum = 0, 0, 0, 0.0

start_time = time.time()
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(x)
            loss = criterion(logits, y)

        probs = F.softmax(logits, dim=1)
        _, pred = torch.max(probs, dim=1)
        _, top5 = torch.topk(probs, k=5, dim=1)

        val_loss_sum += loss.item() * y.size(0)
        top1_correct += (pred == y).sum().item()
        top5_correct += (top5 == y.unsqueeze(1)).sum().item()
        total += y.size(0)

        all_preds.extend(pred.cpu().numpy().tolist())
        all_targets.extend(y.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

eval_duration = time.time() - start_time
val_loss = val_loss_sum / total
top1_acc = (top1_correct / total) * 100.0
top5_acc = (top5_correct / total) * 100.0

avg_batch_eval_ms = (eval_duration / len(val_loader)) * 1000.0
avg_sample_eval_ms = (eval_duration / total) * 1000.0

p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)

print("Total evaluation time (s):", round(eval_duration, 2))
print("Average evaluation time per batch (ms):", round(avg_batch_eval_ms, 2))
print("Average evaluation time per sample (ms):", round(avg_sample_eval_ms, 2))
print("Validation Loss:", round(val_loss, 4))
print("Top-1 Accuracy (%):", round(top1_acc, 2))
print("Top-5 Accuracy (%):", round(top5_acc, 2))
print("Macro Precision (%):", round(p_macro * 100, 2))
print("Macro Recall (%):", round(r_macro * 100, 2))
print("Macro F1:", round(f1_macro, 4))
print("Weighted F1:", round(f1_weight, 4))
"""
    nb.cells.append(nbf.v4.new_code_cell(c6))

    c7 = """print(classification_report(all_targets, all_preds, target_names=class_names, digits=4, zero_division=0))
"""
    nb.cells.append(nbf.v4.new_code_cell(c7))

    c8 = """cm = confusion_matrix(all_targets, all_preds)

plt.figure(figsize=(18, 14))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title(f"Confusion Matrix (Top-1: {top1_acc:.2f}%)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()
"""
    nb.cells.append(nbf.v4.new_code_cell(c8))

    c9 = """report_dict = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)
class_f1 = {cls: report_dict[cls]["f1-score"] for cls in class_names if cls in report_dict}
sorted_classes = sorted(class_f1.items(), key=lambda x: x[1], reverse=True)

top_10 = sorted_classes[:10]
worst_10 = sorted_classes[-10:]

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.barh(np.arange(len(top_10)), [x[1] * 100 for x in top_10], color="green")
plt.yticks(np.arange(len(top_10)), [x[0] for x in top_10])
plt.xlabel("F1 (%)")
plt.title("Top 10 Classes")
plt.xlim(0, 105)
plt.gca().invert_yaxis()

plt.subplot(1, 2, 2)
plt.barh(np.arange(len(worst_10)), [x[1] * 100 for x in worst_10], color="red")
plt.yticks(np.arange(len(worst_10)), [x[0] for x in worst_10])
plt.xlabel("F1 (%)")
plt.title("Worst 10 Classes")
plt.xlim(0, 105)
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
"""
    nb.cells.append(nbf.v4.new_code_cell(c9))

    c10 = """confidences = [probs[pred] for probs, pred in zip(all_probs, all_preds)]
correct_mask = np.array(all_preds) == np.array(all_targets)

correct_confs = [conf * 100 for conf, is_corr in zip(confidences, correct_mask) if is_corr]
incorrect_confs = [conf * 100 for conf, is_corr in zip(confidences, correct_mask) if not is_corr]

print("Mean confidence on correct predictions (%):", round(float(np.mean(correct_confs)), 2) if correct_confs else 0.0)
print("Mean confidence on incorrect predictions (%):", round(float(np.mean(incorrect_confs)), 2) if incorrect_confs else 0.0)

plt.figure(figsize=(10, 4))
plt.hist(correct_confs, bins=25, alpha=0.7, color="green", label="Correct", density=True)
if incorrect_confs:
    plt.hist(incorrect_confs, bins=25, alpha=0.7, color="red", label="Incorrect", density=True)
plt.title("Prediction Confidence Distribution")
plt.xlabel("Confidence (%)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()
"""
    nb.cells.append(nbf.v4.new_code_cell(c10))

    c11 = f"""if os.path.exists("{curve_img}"):
    from IPython.display import Image, display
    display(Image(filename="{curve_img}"))
"""
    nb.cells.append(nbf.v4.new_code_cell(c11))

    c12 = """correct_indices = [i for i in range(len(val_loader.dataset)) if all_preds[i] == all_targets[i]][:3]
incorrect_indices = [i for i in range(len(val_loader.dataset)) if all_preds[i] != all_targets[i]][:3]
display_indices = correct_indices + incorrect_indices

mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
frame_steps = [0, 2, 4, 6, 8, 10, 12, 14]

fig, axes = plt.subplots(len(display_indices), len(frame_steps), figsize=(16, 2.2 * len(display_indices)))

for row, idx in enumerate(display_indices):
    x_val, y_val = val_loader.dataset[idx]
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            logits = model(x_val.unsqueeze(0).to(device))
        probs = F.softmax(logits, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    conf = probs[pred_idx].item() * 100
    true_cls = class_names[y_val]
    pred_cls = class_names[pred_idx]
    is_correct = (pred_idx == y_val)
    status_color = "green" if is_correct else "red"

    for col, f_idx in enumerate(frame_steps):
        ax = axes[row, col]
        img = x_val[:, f_idx, :, :].cpu().numpy()
        img = (img * std + mean).transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.axis("off")
        if col == 0:
            title_text = f"True: {true_cls}\\nPred: {pred_cls} ({conf:.1f}%)"
            ax.set_title(title_text, color=status_color, fontsize=9)
        else:
            ax.set_title(f"t={f_idx}", fontsize=8)

plt.tight_layout()
plt.show()
"""
    nb.cells.append(nbf.v4.new_code_cell(c12))

    c13 = """fig, axes = plt.subplots(1, 3, figsize=(15, 4))
test_sample_indices = [5, 25, 45]

for i, s_idx in enumerate(test_sample_indices):
    x_val, y_val = val_loader.dataset[s_idx]
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            logits = model(x_val.unsqueeze(0).to(device))
        probs = F.softmax(logits, dim=1)[0]
    
    top3_probs, top3_indices = torch.topk(probs, k=3)
    top3_probs = [p.item() * 100 for p in top3_probs]
    top3_names = [class_names[idx.item()] for idx in top3_indices]
    
    axes[i].barh(np.arange(3), top3_probs, color="steelblue")
    axes[i].set_yticks(np.arange(3))
    axes[i].set_yticklabels(top3_names)
    axes[i].set_xlabel("Probability (%)")
    axes[i].set_xlim(0, 100)
    axes[i].invert_yaxis()
    axes[i].set_title(f"True: {class_names[y_val]}")

plt.tight_layout()
plt.show()
"""
    nb.cells.append(nbf.v4.new_code_cell(c13))

    out_path = os.path.join(work_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    return out_path

def create_comparison_notebook():
    nb = nbf.v4.new_notebook()

    header = """# So sanh hieu nang cac mo hinh Lab 03
Sinh vien: Nguyen Trong Thanh
MSSV: 23001934
Mon hoc: Advanced Reading on Computer Vision (MAT3563)
Dataset: UCF50
"""
    nb.cells.append(nbf.v4.new_markdown_cell(header))

    c1 = """import os
import sys
import time
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

data_root = "./dataset/UCF50"
train_loader, val_loader, class_to_idx = get_dataloaders(
    data_root=data_root,
    batch_size=32,
    num_frames=16,
    img_size=112,
    output_format="CTHW",
    split_ratio=0.8,
    num_workers=4,
    seed=42
)
idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(class_to_idx))]

models_config = [
    {"name": "3D CNN Baseline", "module": "cnn3d", "class": "SimpleCNN3D", "ckpt": "best_cnn3d.pth"},
    {"name": "3D CNN v2 (5x5)", "module": "cnn3d_v2", "class": "SimpleCNN3D_v2", "ckpt": "best_cnn3d_v2.pth"},
    {"name": "3D CNN Sequential", "module": "cnn3d_sequential", "class": "SimpleCNN3D_Sequential", "ckpt": "best_cnn3d_sequential.pth"},
    {"name": "MobileNetV4 + GRU", "module": "mobilenetv4_gru", "class": "MobileNetV4_GRU", "ckpt": "best_mobilenetv4_gru.pth"}
]

benchmark_results = []
dummy = torch.randn(1, 3, 16, 112, 112, device=device)
dummy_batch = torch.randn(32, 3, 16, 112, 112, device=device)

for cfg in models_config:
    mod = importlib.import_module(cfg["module"])
    cls = getattr(mod, cfg["class"])
    model = cls(num_classes=len(class_names)).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    file_size_mb = os.path.getsize(cfg["ckpt"]) / (1024 * 1024)

    ckpt = torch.load(cfg["ckpt"], map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    clean_state = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state)
    model.eval()

    with torch.no_grad():
        for _ in range(15):
            with torch.amp.autocast("cuda"):
                _ = model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    trials = 60
    latencies = []
    with torch.no_grad():
        for _ in range(trials):
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda"):
                _ = model(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    throughput_fps = 1000.0 / mean_lat

    batch_trials = 20
    batch_latencies = []
    with torch.no_grad():
        for _ in range(batch_trials):
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda"):
                _ = model(dummy_batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_latencies.append((time.perf_counter() - t0) * 1000.0)
    mean_batch_lat = float(np.mean(batch_latencies))

    top1_correct, top5_correct, total = 0, 0, 0
    all_preds, all_targets = [], []

    eval_t0 = time.time()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(x)
            probs = F.softmax(logits, dim=1)
            _, pred = torch.max(probs, dim=1)
            _, top5 = torch.topk(probs, k=5, dim=1)

            top1_correct += (pred == y).sum().item()
            top5_correct += (top5 == y.unsqueeze(1)).sum().item()
            total += y.size(0)

            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    total_eval_time = time.time() - eval_t0
    avg_batch_eval_time_ms = (total_eval_time / len(val_loader)) * 1000.0

    top1_acc = (top1_correct / total) * 100.0
    top5_acc = (top5_correct / total) * 100.0
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)

    benchmark_results.append({
        "Model": cfg["name"],
        "Params": total_params,
        "Size (MB)": round(file_size_mb, 2),
        "Mean Latency (ms)": round(mean_lat, 2),
        "Std Latency (ms)": round(std_lat, 2),
        "P95 Latency (ms)": round(p95_lat, 2),
        "Batch Latency (ms)": round(mean_batch_lat, 2),
        "Avg Eval Batch (ms)": round(avg_batch_eval_time_ms, 2),
        "FPS": round(throughput_fps, 1),
        "Top-1 Acc (%)": round(top1_acc, 2),
        "Top-5 Acc (%)": round(top5_acc, 2),
        "Macro F1": round(f1_macro, 4)
    })
    print(f"Evaluated {cfg['name']}: Top-1={top1_acc:.2f}%, Latency={mean_lat:.2f}ms")

df = pd.DataFrame(benchmark_results)
display(df)
"""
    nb.cells.append(nbf.v4.new_code_cell(c1))

    c2 = """fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(df["Model"], df["Top-1 Acc (%)"], color="steelblue")
axes[0].set_title("Top-1 Accuracy (%)")
axes[0].set_ylim(60, 105)
for i, v in enumerate(df["Top-1 Acc (%)"]):
    axes[0].text(i, v + 1, f"{v:.2f}%", ha="center")
axes[0].tick_params(axis="x", rotation=15)

axes[1].bar(df["Model"], df["Top-5 Acc (%)"], color="darkorange")
axes[1].set_title("Top-5 Accuracy (%)")
axes[1].set_ylim(85, 103)
for i, v in enumerate(df["Top-5 Acc (%)"]):
    axes[1].text(i, v + 0.4, f"{v:.2f}%", ha="center")
axes[1].tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(df["Model"], df["Mean Latency (ms)"], yerr=df["Std Latency (ms)"], capsize=5, color="crimson")
axes[0].set_title("Mean Latency per Video (ms) with Std Dev")
for i, v in enumerate(df["Mean Latency (ms)"]):
    axes[0].text(i, v + df["Std Latency (ms)"][i] + 0.1, f"{v:.2f}ms", ha="center")
axes[0].tick_params(axis="x", rotation=15)

axes[1].bar(df["Model"], df["Batch Latency (ms)"], color="darkmagenta")
axes[1].set_title("Batch Latency (batch_size=32) in ms")
for i, v in enumerate(df["Batch Latency (ms)"]):
    axes[1].text(i, v + 1, f"{v:.1f}ms", ha="center")
axes[1].tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(df["Model"], df["FPS"], color="forestgreen")
axes[0].set_title("Throughput (Videos/s)")
for i, v in enumerate(df["FPS"]):
    axes[0].text(i, v + 5, f"{v:.0f}", ha="center")
axes[0].tick_params(axis="x", rotation=15)

axes[1].bar(df["Model"], df["Avg Eval Batch (ms)"], color="slateblue")
axes[1].set_title("Average Validation Batch Time (DataLoader + Forward) in ms")
for i, v in enumerate(df["Avg Eval Batch (ms)"]):
    axes[1].text(i, v + 2, f"{v:.1f}ms", ha="center")
axes[1].tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(df["Model"], df["Params"] / 1e6, color="teal")
axes[0].set_title("Parameters (Millions)")
for i, v in enumerate(df["Params"] / 1e6):
    axes[0].text(i, v + 0.1, f"{v:.2f}M" if v >= 0.1 else f"{v*1000:.0f}K", ha="center")
axes[0].tick_params(axis="x", rotation=15)

axes[1].bar(df["Model"], df["Size (MB)"], color="chocolate")
axes[1].set_title("Checkpoint File Size (MB)")
for i, v in enumerate(df["Size (MB)"]):
    axes[1].text(i, v + 1, f"{v:.2f}MB", ha="center")
axes[1].tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plt.scatter(df["Mean Latency (ms)"], df["Top-1 Acc (%)"], s=120, color="navy")
for i, row in df.iterrows():
    plt.annotate(row["Model"], (row["Mean Latency (ms)"] + 0.15, row["Top-1 Acc (%)"] - 0.2))
plt.title("Speed vs Accuracy Trade-off")
plt.xlabel("Mean Latency (ms)")
plt.ylabel("Top-1 Accuracy (%)")
plt.grid(True)
plt.tight_layout()
plt.show()
"""
    nb.cells.append(nbf.v4.new_code_cell(c2))

    out_path = os.path.join(work_dir, "23001934_NguyenTrongThanh_Lab03_benchmark_comparison.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    return out_path

def run_notebook(filepath):
    print("Running:", filepath)
    t0 = time.time()
    with open(filepath, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="cv-env")
    client.execute(cwd=work_dir)
    with open(filepath, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Completed in", round(time.time() - t0, 2), "s")

configs = [
    ("23001934_NguyenTrongThanh_Lab03.ipynb", "Bai 1: 3D CNN Baseline", "cnn3d", "SimpleCNN3D", "best_cnn3d.pth", "loss_acc_curve.png"),
    ("23001934_NguyenTrongThanh_Lab03_v2.ipynb", "Bai 2: 3D CNN v2 (Filter 5x5)", "cnn3d_v2", "SimpleCNN3D_v2", "best_cnn3d_v2.pth", "loss_acc_curve_v2.png"),
    ("23001934_NguyenTrongThanh_Lab03_v3_sequential.ipynb", "Bai 3: 3D CNN Sequential", "cnn3d_sequential", "SimpleCNN3D_Sequential", "best_cnn3d_sequential.pth", "loss_acc_curve_sequential.png"),
    ("23001934_NguyenTrongThanh_Lab03_v4_mobilenetv4_gru.ipynb", "Phan D: MobileNetV4 + GRU", "mobilenetv4_gru", "MobileNetV4_GRU", "best_mobilenetv4_gru.pth", "loss_acc_curve_mobilenetv4_gru.png")
]

created_files = []
for fname, title, mod, cls, ckpt, curve in configs:
    created_files.append(create_notebook(fname, title, mod, cls, ckpt, curve))

created_files.append(create_comparison_notebook())

for path in created_files:
    run_notebook(path)

print("Done.")
