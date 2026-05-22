from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch

# Custom dataset to replace WebDataset
class UnstructuredImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_filenames = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg"))
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name=self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name
