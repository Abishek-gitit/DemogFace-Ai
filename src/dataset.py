import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class UTKFaceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Parse labels from filename (e.g., 26_0_2_20170116180126.jpg)
        filename = os.path.basename(img_path)
        parts = filename.split('_')
        
        # Fallback values
        age = 30.0
        gender = 0
        
        try:
            age = float(parts[0])
            gender = int(parts[1])
            if gender not in [0, 1]:
                gender = 0
        except Exception:
            pass
            
        if self.transform:
            image = self.transform(image)
            
        # Return image, age label (regression), gender label (classification)
        return image, torch.tensor(age, dtype=torch.float32), torch.tensor(gender, dtype=torch.long)

def get_dataloaders(data_dir, batch_size=64, img_size=128, val_split=0.2, seed=42):
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                if len(file.split('_')) >= 3:
                    image_paths.append(os.path.join(root, file))
                    
    if not image_paths:
        raise ValueError(f"No valid UTKFace images found in {data_dir}. Ensure dataset is downloaded and extracted.")
        
    train_paths, val_paths = train_test_split(image_paths, test_size=val_split, random_state=seed)
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = UTKFaceDataset(train_paths, transform=train_transform)
    val_dataset = UTKFaceDataset(val_paths, transform=val_transform)
    
    # num_workers=0 is safer on mac to avoid multiprocessing issues with fork/spawn
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, len(train_paths), len(val_paths)
