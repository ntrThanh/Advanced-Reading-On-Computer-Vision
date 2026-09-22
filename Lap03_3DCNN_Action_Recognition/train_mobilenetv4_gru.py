import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from mobilenetv4_gru import MobileNetV4_GRU


def log_msg(file_path, prefix, message):
    text = f"{prefix} {message}"
    print(text, flush=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def main():
    data_root = "./dataset/UCF50"
    checkpoint_path = "best_mobilenetv4_gru.pth"
    log_file = "train_mobilenetv4_gru.log"

    batch_size = 16
    num_frames = 16
    img_size = 112
    learning_rate = 1e-4
    total_epochs = 30
    num_workers = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_msg(log_file, "[+]", f"Using device: {device}")

    train_loader, val_loader, class_to_idx = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_frames=num_frames,
        img_size=img_size,
        output_format="CTHW",
        split_ratio=0.8,
        num_workers=num_workers
    )

    num_classes = len(class_to_idx)
    log_msg(log_file, "[+]", f"Loaded dataset with {num_classes} classes")

    torch.backends.cudnn.benchmark = True
    model = MobileNetV4_GRU(
        num_classes=num_classes,
        model_name="mobilenetv4_conv_small",
        pretrained=True,
        hidden_size=256,
        num_layers=1,
        bidirectional=True,
        dropout=0.3
    )

    if torch.cuda.device_count() > 1:
        log_msg(log_file, "[+]", f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_val_acc = 0.0

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        target = model.module if hasattr(model, "module") else model

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            clean_state = {}
            for k, v in ckpt["model_state_dict"].items():
                name = k[7:] if k.startswith("module.") else k
                clean_state[name] = v
            target.load_state_dict(clean_state)

            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt and use_amp:
                scaler.load_state_dict(ckpt["scaler_state_dict"])

            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_acc = ckpt.get("best_val_acc", 0.0)
            log_msg(log_file, "[+]", f"Resumed training from epoch {start_epoch} with Best Val Acc: {best_val_acc * 100:.2f}%")
        else:
            clean_state = {}
            for k, v in ckpt.items():
                name = k[7:] if k.startswith("module.") else k
                clean_state[name] = v
            target.load_state_dict(clean_state)
            log_msg(log_file, "[+]", f"Loaded existing weights from {checkpoint_path}")

    if start_epoch > total_epochs:
        log_msg(log_file, "[+]", f"Training already completed up to epoch {total_epochs}")
        return

    log_msg(log_file, "[+]", f"Training from epoch {start_epoch} to {total_epochs}")

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            total_train += targets.size(0)

        epoch_train_loss = train_loss / total_train if total_train > 0 else 0.0
        epoch_train_acc = train_correct / total_train if total_train > 0 else 0.0

        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                total_val += targets.size(0)

        epoch_val_loss = val_loss / total_val if total_val > 0 else 0.0
        epoch_val_acc = val_correct / total_val if total_val > 0 else 0.0

        log_msg(
            log_file,
            "[+]",
            f"Epoch [{epoch:02d}/{total_epochs:02d}] Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%"
        )

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            save_model = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": save_model,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_acc": best_val_acc
            }
            torch.save(checkpoint_dict, checkpoint_path)
            log_msg(log_file, "[+]", f"Saved new best checkpoint with Val Acc: {epoch_val_acc * 100:.2f}%")
        else:
            log_msg(log_file, "[-]", f"Val Acc did not improve, keeping current best: {best_val_acc * 100:.2f}%")

    log_msg(log_file, "[+]", f"Training completed. Best Val Acc: {best_val_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
