import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

# --- 1. Dataset Class (Remains mostly the same) ---
class FiberDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        df = pd.read_csv(csv_file, header=None)
        self.samples = []
        groups = df.groupby(1)
        
        for group_id, group_data in groups:
            files = group_data[4].tolist()
            l_map = next((f for f in files if "luster_map" in f), None)
            r_map = next((f for f in files if "roughness_proxy" in f), None)
            t_map = next((f for f in files if "tensile_map" in f), None)
            
            if l_map and r_map and t_map:
                if 635 <= group_id <= 734:
                    label = 1 # Daratex
                elif 580 <= group_id <= 634 or 735 <= group_id <= 779:
                    label = 0 # Abaca
                else:
                    continue
                
                self.samples.append({'l': l_map, 'r': r_map, 't': t_map, 'label': label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        l_img = Image.open(os.path.join(self.root_dir, 'luster_map', s['l'])).convert('L')
        r_img = Image.open(os.path.join(self.root_dir, 'roughness_proxy', s['r'])).convert('L')
        t_img = Image.open(os.path.join(self.root_dir, 'tensile_map', s['t'])).convert('L')
        
        if self.transform:
            l_img = self.transform(l_img)
            r_img = self.transform(r_img)
            t_img = self.transform(t_img)
            
        return l_img, r_img, t_img, s['label']

# --- 2. Single-Input Model Architecture ---
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

class SinglePropertyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = HierarchicalEncoder(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        features = self.encoder(x)
        pooled = self.gap(features)
        flat = torch.flatten(pooled, 1)
        return self.classifier(flat)

def train_individual_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.CenterCrop(300),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = FiberDataset('db_export.csv', './labeled_dataset', transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # Define the tasks
    tasks = [
        {'name': 'luster', 'index': 0},    # L map is at index 0 in __getitem__
        {'name': 'roughness', 'index': 1}, # R map is at index 1
        {'name': 'tensile', 'index': 2}    # T map is at index 2
    ]

    for task in tasks:
        print(f"\n--- Training Model for: {task['name'].upper()} ---")
        model = SinglePropertyClassifier().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        for epoch in range(15): # 15 epochs is enough to see a trend
            model.train()
            total_loss = 0
            for data in train_loader:
                # Select only the specific map for this model
                input_img = data[task['index']].to(device)
                labels = data[3].to(device)

                optimizer.zero_grad()
                outputs = model(input_img)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation phase
            model.eval()
            correct = 0
            with torch.no_grad():
                for data in val_loader:
                    input_img = data[task['index']].to(device)
                    labels = data[3].to(device)
                    outputs = model(input_img)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
            
            acc = 100 * correct / len(val_ds)
            print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")

        save_name = f"{task['name']}_model.pth"
        torch.save(model.state_dict(), save_name)
        print(f"Saved: {save_name}")

if __name__ == "__main__":
    train_individual_models()