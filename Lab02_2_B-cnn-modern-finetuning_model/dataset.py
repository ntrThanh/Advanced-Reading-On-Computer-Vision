import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class AnimalDataset(Dataset):
    def __init__(self, root_dir, image_paths=None, labels=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        if image_paths is not None and labels is not None:
            self.image_paths = image_paths
            self.labels = labels
            self.classes = sorted(list(set(labels)))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        else:
            self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            self.image_paths = []
            self.labels = []
            valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
            for cls_name in self.classes:
                cls_folder = os.path.join(root_dir, cls_name)
                for fname in sorted(os.listdir(cls_folder)):
                    if fname.lower().endswith(valid_extensions):
                        self.image_paths.append(os.path.join(cls_folder, fname))
                        self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def get_transforms(image_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_transform, val_transform

def get_dataloaders(root_dir, val_size=500, batch_size=32, image_size=224, num_workers=2, seed=42):
    train_tf, val_tf = get_transforms(image_size)
    full_dataset = AnimalDataset(root_dir=root_dir)
    indices = list(range(len(full_dataset)))
    random.seed(seed)
    random.shuffle(indices)

    val_indices = set(indices[:val_size])
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []

    for i in range(len(full_dataset)):
        if i in val_indices:
            val_paths.append(full_dataset.image_paths[i])
            val_labels.append(full_dataset.labels[i])
        else:
            train_paths.append(full_dataset.image_paths[i])
            train_labels.append(full_dataset.labels[i])

    train_ds = AnimalDataset(root_dir, train_paths, train_labels, transform=train_tf)
    val_ds = AnimalDataset(root_dir, val_paths, val_labels, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, full_dataset.classes
