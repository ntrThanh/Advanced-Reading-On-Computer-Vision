import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class UCF50Dataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=16, img_size=112, is_train=True, output_format="CTHW"):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.img_size = img_size
        self.is_train = is_train
        self.output_format = output_format

    def __len__(self):
        return len(self.video_paths)

    def _load_frames(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return np.zeros((self.num_frames, self.img_size, self.img_size, 3), dtype=np.uint8)

        if total_frames < self.num_frames:
            indices = [i % total_frames for i in range(self.num_frames)]
        else:
            step = total_frames / self.num_frames
            if self.is_train:
                indices = [int(i * step + random.uniform(0, step)) for i in range(self.num_frames)]
            else:
                indices = [int(i * step + step / 2) for i in range(self.num_frames)]
            indices = [min(idx, total_frames - 1) for idx in indices]

        frames = []
        cur_idx = 0
        needed_set = set(indices)
        saved_frames = {}
        max_idx = max(indices)

        while cap.isOpened() and cur_idx <= max_idx:
            if cur_idx in needed_set:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                saved_frames[cur_idx] = frame
            else:
                ret = cap.grab()
                if not ret:
                    break
            cur_idx += 1
        cap.release()

        last_frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        for idx in indices:
            if idx in saved_frames:
                last_frame = saved_frames[idx]
            frames.append(last_frame)

        return np.array(frames, dtype=np.uint8)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self._load_frames(path)

        if self.is_train and random.random() > 0.5:
            frames = np.ascontiguousarray(frames[:, :, ::-1, :])

        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        if self.output_format == "CTHW":
            tensor = tensor.permute(1, 0, 2, 3)

        return tensor, label


def get_dataloaders(data_root="./dataset/UCF50", batch_size=32, num_frames=16, img_size=112, output_format="CTHW", split_ratio=0.8, num_workers=8, seed=42):
    classes = sorted(os.listdir(data_root))
    classes = [c for c in classes if os.path.isdir(os.path.join(data_root, c)) and not c.startswith('.')]
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    random.seed(seed)
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []

    for cls_name in classes:
        cls_dir = os.path.join(data_root, cls_name)
        vids = sorted(glob.glob(os.path.join(cls_dir, "*.avi")))
        random.shuffle(vids)
        split = int(len(vids) * split_ratio)

        train_paths.extend(vids[:split])
        train_labels.extend([class_to_idx[cls_name]] * split)

        val_paths.extend(vids[split:])
        val_labels.extend([class_to_idx[cls_name]] * (len(vids) - split))

    train_dataset = UCF50Dataset(train_paths, train_labels, num_frames=num_frames, img_size=img_size, is_train=True, output_format=output_format)
    val_dataset = UCF50Dataset(val_paths, val_labels, num_frames=num_frames, img_size=img_size, is_train=False, output_format=output_format)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2
    )

    return train_loader, val_loader, class_to_idx


if __name__ == "__main__":
    data_root = "./dataset/UCF50"
    train_loader, val_loader, class_to_idx = get_dataloaders(data_root=data_root, batch_size=32, num_frames=16, img_size=112, num_workers=4)
    print("Classes:", len(class_to_idx))
    print("Train samples:", len(train_loader.dataset))
    print("Val samples:", len(val_loader.dataset))
    x, y = next(iter(train_loader))
    print("Batch shape:", x.shape)
    print("Labels:", y)
