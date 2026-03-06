import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
import shutil
import random
import time  # Added for time logging
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import re
import glob

# --- 1. Transforms ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. Model Architectures ---

def get_cnn_model(num_classes=3, input_channels=3):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if input_channels != 3:
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return x + (attn @ v)

class CAILN_Model(nn.Module):
    def __init__(self, num_classes=3, input_channels=3):
        super().__init__()
        self.feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor.fc = nn.Identity() 
        self.attention = CrossAttentionBlock(512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x) 
        features = features.unsqueeze(1)    
        attended = self.attention(features)
        return self.classifier(attended.squeeze(1))
    
class PHEFM_Model(nn.Module):
    def __init__(self, num_classes=3, input_channels=3):
        super().__init__()
        self.branch1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.branch2 = nn.Conv2d(input_channels, 32, kernel_size=5, padding=2)
        self.branch3 = nn.Conv2d(input_channels, 32, kernel_size=7, padding=3)
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder.conv1 = nn.Conv2d(96, 64, kernel_size=7, stride=2, padding=3) 
        in_feats = self.encoder.fc.in_features
        self.encoder.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_feats, num_classes)
        )

    def forward(self, x):
        b1 = torch.relu(self.branch1(x))
        b2 = torch.relu(self.branch2(x))
        b3 = torch.relu(self.branch3(x))
        fused = torch.cat([b1, b2, b3], dim=1) 
        return self.encoder(fused)

# --- 3. Dataset Handling ---

class FiberDataset(Dataset):
    def __init__(self, root_dir, feature_type='all', transform=None):
        self.root_dir = root_dir
        self.feature_type = feature_type
        self.transform = transform
        # UPDATED: Included 'Mixed' class for 3-category classification
        self.classes = ['Abaca', 'Daratex', 'Mixed'] 
        self.data = []

        print(f"--- Scanning Dataset at: {root_dir} ---")
        for cls_idx, cls_name in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls_name, 'sample')
            if not os.path.exists(cls_path):
                continue
            
            sample_names = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for name in sample_names:
                category_folder = os.path.join(root_dir, cls_name)
                self.data.append((category_folder, name, cls_idx))
        
        print(f"[INFO] Total samples loaded for {feature_type}: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        category_folder, sample_filename, label = self.data[idx]
        try:
            ts_id = sample_filename.split('-')[-1].split('.')[0]
        except:
            ts_id = sample_filename.split('.')[0]

        def get_feature_path(feature_name):
            search_pattern = os.path.join(category_folder, feature_name, f"*{ts_id}*")
            matching_files = glob.glob(search_pattern)
            if not matching_files:
                raise FileNotFoundError(f"Missing {feature_name} for ID {ts_id}")
            return matching_files[0]

        if self.feature_type == 'all':
            p1 = Image.open(get_feature_path('luster_map')).convert('L')
            p2 = Image.open(get_feature_path('roughness_proxy')).convert('L')
            p3 = Image.open(get_feature_path('tensile_map')).convert('L')
            img = Image.merge('RGB', (p1, p2, p3))
        else:
            img = Image.open(get_feature_path(self.feature_type)).convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img, label

# --- 4. Training Loop ---

def train_model(model, train_loader, val_loader, model_name, epochs=30):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_acc = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        # Start timing the epoch
        start_time = time.time()
        
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"best_{model_name}_test.pth")
            
        # End timing and calculate duration
        epoch_time = time.time() - start_time
        mins, secs = divmod(epoch_time, 60)
            
        print(f'Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | '
              f'Val Acc: {val_acc:.2f}% | Time: {int(mins)}m {int(secs)}s | '
              f'LR: {optimizer.param_groups[0]["lr"]}')

    return best_acc

# --- 5. Main Execution ---

ARCHS = ['CNN','CAILN', 'PHEFM']
FEATURES = ['luster_map', 'roughness_proxy', 'tensile_map', 'all']
RESULTS_LOG = []

if __name__ == "__main__":
    DATA_ROOT = r"C:\Users\yen\Downloads\fiber-optics-backend-dev\services\uploads\labeled_dataset"

    if not os.path.exists(DATA_ROOT):
        print(f"ERROR: Dataset path not found at {DATA_ROOT}")
    else:
        print("--- Starting Training Matrix (3-Class: Abaca, Daratex, Mixed) ---")
        
        for arch_name in ARCHS:
            for feat_name in FEATURES:
                print(f"\n[INIT] {arch_name} + {feat_name}")
                
                train_set = FiberDataset(os.path.join(DATA_ROOT, 'train'), 
                                         feature_type=feat_name, 
                                         transform=train_transform)
                val_set = FiberDataset(os.path.join(DATA_ROOT, 'val'), 
                                       feature_type=feat_name, 
                                       transform=val_transform)
                
                if len(train_set) == 0:
                    print(f"Skipping {feat_name}: No samples found.")
                    continue

                train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
                val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

                # Initialize Models with 3 classes
                if arch_name == 'CAILN':
                    model = CAILN_Model(num_classes=3)
                elif arch_name == 'PHEFM':
                    model = PHEFM_Model(num_classes=3)
                else:
                    model = get_cnn_model(num_classes=3)

                accuracy = train_model(model, train_loader, val_loader, f"{arch_name}_{feat_name}")
                RESULTS_LOG.append((arch_name, feat_name, accuracy))

        print("\n" + "="*45)
        print(f"{'ARCH':<10} | {'FEATURE':<15} | {'ACCURACY':<10}")
        print("-" * 45)
        for res in RESULTS_LOG:
            print(f"{res[0]:<10} | {res[1]:<15} | {res[2]:>8.2f}%")