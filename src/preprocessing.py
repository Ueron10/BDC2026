
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


class WasteDataset(Dataset):
    
    def __init__(self, image_paths, labels=None, transform=None, is_test=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        if self.is_test:
            return image, image_path
        else:
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.long)


def get_train_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=30, p=0.5),
        A.OneOf([
            A.MotionBlur(p=0.3),
            A.MedianBlur(p=0.3),
            A.GaussianBlur(p=0.3),
            A.GaussNoise(p=0.3),
        ], p=0.3),
        A.OneOf([
            A.CLAHE(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.RandomGamma(p=0.3),
        ], p=0.4),
        A.OneOf([
            A.GridDistortion(p=0.3),
            A.ElasticTransform(p=0.3),
            A.OpticalDistortion(p=0.3),
        ], p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_tta_transforms():
    transforms = []
    
    transforms.append(A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]))
    
    transforms.append(A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]))
    
    transforms.append(A.Compose([
        A.Resize(224, 224),
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]))
    
    transforms.append(A.Compose([
        A.Resize(224, 224),
        A.RandomRotate90(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]))
    
    transforms.append(A.Compose([
        A.Resize(224, 224),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]))
    
    return transforms


def load_train_data(data_dir):
    image_paths = []
    labels = []
    class_to_idx = {
        'Recyclable': 0,
        'Electronic': 1,
        'Organic': 2
    }
    
    train_dir = os.path.join(data_dir, 'train')
    
    for class_name, class_idx in class_to_idx.items():
        class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    image_paths.append(img_path)
                    labels.append(class_idx)
    
    print(f"Loaded {len(image_paths)} training images")
    print(f"Class distribution:")
    for class_name, class_idx in class_to_idx.items():
        count = labels.count(class_idx)
        print(f"  {class_name}: {count} images")
    
    return image_paths, labels, class_to_idx


def load_test_data(data_dir):
    image_paths = []
    test_dir = os.path.join(data_dir, 'test')
    
    if os.path.exists(test_dir):
        img_files = sorted(os.listdir(test_dir), key=lambda x: int(x.split('.')[0]))
        for img_name in img_files:
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(test_dir, img_name)
                image_paths.append(img_path)
    
    print(f"Loaded {len(image_paths)} test images")
    return image_paths


def split_train_val(image_paths, labels, val_split=0.2, random_state=42):
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, 
        test_size=val_split, 
        random_state=random_state,
        stratify=labels
    )
    
    print(f"Train set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    
    return train_paths, val_paths, train_labels, val_labels


def create_dataloaders(train_paths, train_labels, val_paths, val_labels, 
                      batch_size=32, num_workers=4):
    train_dataset = WasteDataset(
        train_paths, train_labels, 
        transform=get_train_transforms()
    )
    
    val_dataset = WasteDataset(
        val_paths, val_labels, 
        transform=get_val_transforms()
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_test_dataloader(test_paths, batch_size=32, num_workers=4):
    test_dataset = WasteDataset(
        test_paths, 
        transform=get_val_transforms(),
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader
