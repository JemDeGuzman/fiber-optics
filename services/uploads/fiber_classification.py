import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
# --- 1. Dataset Class ---
class FiberDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load CSV (0: id, 1: group_id, 2: url, 3: timestamp, 4: filename)
        df = pd.read_csv(csv_file, header=None)
        
        self.samples = []
        # Group entries by the 'group_id' (index 1)
        groups = df.groupby(1)
        
        for group_id, group_data in groups:
            files = group_data[4].tolist()
            
            # Identify the three preprocessed maps for this capture group
            l_map = next((f for f in files if "luster_map" in f), None)
            r_map = next((f for f in files if "roughness_proxy" in f), None)
            t_map = next((f for f in files if "tensile_map" in f), None)
            
            if l_map and r_map and t_map:
                # Labeling Logic
                if 493 <= group_id <= 517:
                    label = 0 # Abaca
                elif 518 <= group_id <= 572:
                    label = 1 # Daratex
                else:
                    continue # Ignore groups outside specified ID ranges
                
                self.samples.append({
                    'l': l_map, 'r': r_map, 't': t_map, 'label': label
                })
        
        print(f"Dataset initialized: {len(self.samples)} valid triplets found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # Load as Grayscale (L)
        l_img = Image.open(os.path.join(self.root_dir, 'luster_map', s['l'])).convert('L')
        r_img = Image.open(os.path.join(self.root_dir, 'roughness_proxy', s['r'])).convert('L')
        t_img = Image.open(os.path.join(self.root_dir, 'tensile_map', s['t'])).convert('L')
        
        if self.transform:
            l_img = self.transform(l_img)
            r_img = self.transform(r_img)
            t_img = self.transform(t_img)
            
        return l_img, r_img, t_img, s['label']

# --- 2. Model Architecture ---
class HierarchicalEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.layer2(self.layer1(x))

class FiberClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_luster = HierarchicalEncoder(1)
        self.enc_rough = HierarchicalEncoder(1)
        self.enc_tensile = HierarchicalEncoder(1)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1) # Reduces to 1x1 spatially
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2) # Binary: Abaca (0) or Daratex (1)
        )

    def forward(self, l, r, t):
        l_f = self.enc_luster(l)
        r_f = self.enc_rough(r)
        t_f = self.enc_tensile(t)
        
        fused = torch.cat([l_f, r_f, t_f], dim=1)
        bottleneck = self.fusion(fused)
        flat = torch.flatten(bottleneck, 1)
        return self.classifier(flat)

# --- 3. Training Logic ---
def train():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = FiberDataset('db_export.csv', './labeled_dataset', transform=transform)
    
    # Split into Train (80%) and Val (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = FiberClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Simple Loop
    for epoch in range(20):
        model.train()
        train_loss = 0
        for l, r, t, labels in train_loader:
            l, r, t, labels = l.to(device), r.to(device), t.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(l, r, t)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}/20 | Loss: {train_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "fiber_model.pth")
    print("Model saved to fiber_model.pth")

if __name__ == "__main__":
    train()