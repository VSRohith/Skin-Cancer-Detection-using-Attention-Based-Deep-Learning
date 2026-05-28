import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SkinDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform
        
        # Label mapping (HAM10000 standard)
        self.label_map = {
            'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 
            'mel': 4, 'nv': 5, 'vasc': 6
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Extract image ID and clean filename
        img_id = self.df.iloc[idx]['image_id']
        img_name = os.path.join(self.img_dir, f"{img_id}.jpg")
        
        image = Image.open(img_name).convert('RGB')
        label = self.label_map[self.df.iloc[idx]['dx']]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Standard Image Augmentations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])