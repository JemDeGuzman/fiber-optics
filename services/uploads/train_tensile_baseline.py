import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

# --- 1. Dataset Class (Tensile Only) ---
class TensileDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        df = pd.read_csv(csv_file, header=None)
        self.samples = []
        groups = df.groupby(1)
        
        for group_id, group_data in groups:
            files = group_data[4].tolist()
            # We only look for the tensile map here
            t_map = next((f for f in files if "tensile_map" in f), None)
            
            if t_map:
                if 493 <= group_id <= 517:
                    label = 0 # Abaca
                elif 518 <= group_id <= 572:
                    label = 1 # Daratex
                else:
                    continue
                self.samples.append({'t': t_map, 'label': label})

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Load tensile map as grayscale
        t_img = Image.open(os.path.join(self.root_dir, 'tensile_map', s['t'])).convert('L')
        if self.transform: t_img = self.transform(t_img)
        return t_img, s['label']

# --- 2. Simplified Model Architecture ---
class TensileCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# --- 3. Training Loop with Class Weights ---
def train_tensile_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(), # Added augmentation to prevent overfitting
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = TensileDataset('db_export.csv', './labeled_dataset', transform=transform)
    train_size = int(0.8 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset)-train_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = TensileCNN().to(device)
    
    # FIX: Applying Class Weights
    # Since Daratex (55) has more samples than Abaca (25), 
    # we tell the model Daratex errors are "cheaper" and Abaca/Missing-Daratex is "expensive"
    # Weight for class 1 (Daratex) is increased to help with the low recall
    weights = torch.tensor([1.5, 1.0]).to(device) 
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(20): # Increased epochs for better convergence
        model.train()
        for t_img, labels in train_loader:
            t_img, labels = t_img.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(t_img), labels)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1} completed.")

    torch.save(model.state_dict(), "tensile_only_model.pth")
    print("Baseline training complete.")

if __name__ == "__main__":
    train_tensile_baseline()