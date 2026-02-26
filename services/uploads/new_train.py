import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
import shutil
import random
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import re
import glob

class FiberDataset(Dataset):
    def __init__(self, root_dir, feature_type='all', transform=None):
        self.root_dir = root_dir
        self.feature_type = feature_type
        self.transform = transform
        self.classes = ['Abaca', 'Daratex', 'False']
        self.data = []

        print(f"--- Scanning Dataset at: {root_dir} ---")
        for cls_idx, cls_name in enumerate(self.classes):
            # Construct the path to the 'sample' folder
            # We use sample as the ground truth for filenames
            cls_path = os.path.join(root_dir, cls_name, 'sample')
            
            if not os.path.exists(cls_path):
                print(f"[WARNING] Directory not found: {cls_path}")
                continue
            
            sample_names = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if len(sample_names) == 0:
                print(f"[WARNING] No images found in: {cls_path}")
                continue

            for name in sample_names:
                # Store the path to the parent category folder and the filename
                category_folder = os.path.join(root_dir, cls_name)
                self.data.append((category_folder, name, cls_idx))
        
        print(f"[INFO] Total samples loaded for {feature_type}: {len(self.data)}")
        if len(self.data) == 0:
            raise ValueError(f"Dataset is empty! Check path: {os.path.abspath(root_dir)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        category_folder, sample_filename, label = self.data[idx]
        
        # 1. Extract the unique ID from the sample name
        # Example: 'sample-1771937129317.jpg' -> '1771937129317'
        try:
            ts_id = sample_filename.split('-')[-1].split('.')[0]
        except:
            ts_id = sample_filename.split('.')[0]

        def get_feature_path(feature_name):
            # Look in the specific feature folder for any file containing the ts_id
            search_pattern = os.path.join(category_folder, feature_name, f"*{ts_id}*")
            matching_files = glob.glob(search_pattern)
            if not matching_files:
                raise FileNotFoundError(f"Could not find {feature_name} for ID {ts_id} in {category_folder}")
            return matching_files[0]

        if self.feature_type == 'all':
            # Load and stack all three features
            p1 = Image.open(get_feature_path('luster_map')).convert('L')
            p2 = Image.open(get_feature_path('roughness_proxy')).convert('L')
            p3 = Image.open(get_feature_path('tensile_map')).convert('L')
            img = Image.merge('RGB', (p1, p2, p3))
        else:
            # Load the specific feature requested
            img = Image.open(get_feature_path(self.feature_type)).convert('RGB')

        if self.transform:
            img = self.transform(img)
            
        return img, label
    
def get_cnn_model(num_classes=3, input_channels=1):
    # We use ResNet18 as the backbone
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Adjust the first layer to accept 1 channel (L/R/T) or 3 channels (All)
    if input_channels != 3:
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def split_dataset(root_dir, train_ratio=0.8):
    # The categories you currently have
    categories = ['Abaca', 'Daratex', 'False']
    # The feature sub-folders inside each category
    sub_folders = ['luster_map', 'roughness_proxy', 'sample', 'tensile_map']
    
    # Create the top-level train/val folders
    for split in ['train', 'val']:
        for cat in categories:
            for sub in sub_folders:
                os.makedirs(os.path.join(root_dir, split, cat, sub), exist_ok=True)

    for cat in categories:
        cat_path = os.path.join(root_dir, cat)
        if not os.path.exists(cat_path):
            continue
            
        # We use the 'sample' folder to get the list of unique filenames
        # (Assuming filenames are identical across all sub-folders)
        files = [f for f in os.listdir(os.path.join(cat_path, 'sample')) if f.endswith(('.png', '.jpg'))]
        
        # Split the filenames: 80% for training, 20% for validation
        train_files, val_files = train_test_split(files, test_size=(1 - train_ratio), random_state=42)
        
        # Helper to move files
        def move_files(file_list, target_split):
            for f_name in file_list:
                for sub in sub_folders:
                    source = os.path.join(root_dir, cat, sub, f_name)
                    # Note: Adjust filename mapping if your subfolders use prefixes (e.g., luster_map-ts.png)
                    # If they are exactly the same name, this works:
                    destination = os.path.join(root_dir, target_split, cat, sub, f_name)
                    if os.path.exists(source):
                        shutil.move(source, destination)

        move_files(train_files, 'train')
        move_files(val_files, 'val')
        print(f"Finished splitting {cat}: {len(train_files)} train, {len(val_files)} val.")

def split_dataset_by_timestamp(root_dir, train_ratio=0.8):
    categories = ['Abaca', 'Daratex', 'False']
    sub_folders = ['luster_map', 'roughness_proxy', 'sample', 'tensile_map']
    
    # 1. Create Directories
    for split in ['train', 'val']:
        for cat in categories:
            for sub in sub_folders:
                os.makedirs(os.path.join(root_dir, split, cat, sub), exist_ok=True)

    for cat in categories:
        cat_path = os.path.join(root_dir, cat)
        if not os.path.exists(os.path.join(cat_path, 'sample')):
            continue
            
        # 2. Get all files in 'sample' and extract the timestamp/ID
        # Example: if file is 'sample-20231011.jpg', the ID is '20231011'
        sample_files = os.listdir(os.path.join(cat_path, 'sample'))
        
        # We assume your filenames are formatted like: name-{timestamp}.ext
        # We split based on the unique files in the sample folder
        if not sample_files:
            print(f"Skipping {cat}, no files found in sample folder.")
            continue

        train_files, val_files = train_test_split(sample_files, test_size=(1 - train_ratio), random_state=42)

        def move_all_features(file_list, target_split):
            for f_name in file_list:
                # Extract the timestamp part from 'sample-{ts}.jpg'
                # This regex captures everything after the first hyphen
                match = re.search(r"-(.*)\.", f_name)
                if not match: continue
                ts_id = match.group(1)

                for sub in sub_folders:
                    source_sub_path = os.path.join(root_dir, cat, sub)
                    # Find the file in the subfolder that contains the same timestamp
                    all_in_sub = os.listdir(source_sub_path)
                    target_file = next((x for x in all_in_sub if ts_id in x), None)

                    if target_file:
                        src = os.path.join(source_sub_path, target_file)
                        dst = os.path.join(root_dir, target_split, cat, sub, target_file)
                        shutil.move(src, dst)

        move_all_features(train_files, 'train')
        move_all_features(val_files, 'val')
        print(f"Successfully moved {cat}: {len(train_files)} train, {len(val_files)} val.")

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        # x shape: [Batch, Channels, Features]
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
        self.feature_extractor.fc = nn.Identity() # Remove final layer
        
        # Cross-attention layer
        self.attention = CrossAttentionBlock(512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x) # [Batch, 512]
        features = features.unsqueeze(1)    # Add sequence dim for attention
        attended = self.attention(features)
        return self.classifier(attended.squeeze(1))
    
class PHEFM_Model(nn.Module):
    def __init__(self, num_classes=3, input_channels=3):
        super().__init__()
        # Parallel encoders with different receptive fields
        self.branch1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.branch2 = nn.Conv2d(input_channels, 32, kernel_size=5, padding=2)
        self.branch3 = nn.Conv2d(input_channels, 32, kernel_size=7, padding=3)
        
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder.conv1 = nn.Conv2d(96, 64, kernel_size=7, stride=2, padding=3) # Accept fused branches
        self.encoder.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        b1 = torch.relu(self.branch1(x))
        b2 = torch.relu(self.branch2(x))
        b3 = torch.relu(self.branch3(x))
        fused = torch.cat([b1, b2, b3], dim=1) # Combine hierarchical features
        return self.encoder(fused)

# We use these transforms to make the model robust to rotation and slight lighting shifts
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    # Normalization based on ImageNet standards (standard for ResNet backbones)
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def train_matrix(dataset_path, transform=train_transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architectures = ['CNN', 'CAILN', 'PHEFM']
    features = ['luster_map', 'roughness_proxy', 'tensile_map', 'all']
    
    results = {}

    for arch in architectures:
        for feat in features:
            print(f"\n--- Training {arch} using {feat} ---")
            
            # 1. Prepare Data
            # Note: For 'all', input_channels=3. For others, input_channels=1
            in_channels = 3 if feat == 'all' else 1
            dataset = FiberDataset(dataset_path, feature_type=feat, transform=transform)
            loader = DataLoader(dataset, batch_size=16, shuffle=True)
            
            # 2. Initialize correct model
            if arch == 'CNN':
                model = get_cnn_model(input_channels=in_channels).to(device)
            elif arch == 'CAILN':
                model = CAILN_Model(input_channels=in_channels).to(device)
            elif arch == 'PHEFM':
                model = PHEFM_Model(input_channels=in_channels).to(device)
            
            # 3. Training Loop (Abbreviated)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # train(model, loader, optimizer, criterion, epochs=20)
            
            # 4. Save results for comparison
            model_name = f"{arch}_{feat}.pth"
            torch.save(model.state_dict(), model_name)
            results[f"{arch}_{feat}"] = "Evaluation Metrics Here"

    return results

def train_model(model, train_loader, val_loader, model_name, epochs=30):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Lower LR for stability
    best_acc = 0.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
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

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"best_{model_name}.pth")
            
        print(f'Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2f}%')

    return best_acc

# Configuration
import os
DATA_ROOT = r"C:\Users\yen\Downloads\fiber-optics-backend-dev\services\uploads\labeled_dataset" # Use 'r' for raw string

# Debug check
check_path = os.path.join(DATA_ROOT, 'train', 'Abaca', 'sample')
print(f"Checking path: {check_path}")
print(f"Exists: {os.path.exists(check_path)}")
if os.path.exists(check_path):
    print(f"Files found: {len(os.listdir(check_path))}")

FEATURES = ['luster_map', 'roughness_proxy', 'tensile_map', 'all']
ARCHS = ['CNN', 'CAILN', 'PHEFM']
RESULTS_LOG = []

for arch_name in ARCHS:
    for feat_name in FEATURES:
        print(f"\n[STARTING] Model: {arch_name} | Feature: {feat_name}")
        
        # Load Datasets
        train_set = FiberDataset(os.path.join(DATA_ROOT, 'train'), feature_type=feat_name, transform=train_transform)
        val_set = FiberDataset(os.path.join(DATA_ROOT, 'val'), feature_type=feat_name, transform=val_transform)
        
        train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

        # Initialize Model
        in_channels = 3 # Our Dataset class converts single maps to RGB or stacks 3 maps
        if arch_name == 'CNN':
            model = get_cnn_model(num_classes=3, input_channels=in_channels)
        elif arch_name == 'CAILN':
            model = CAILN_Model(num_classes=3, input_channels=in_channels)
        elif arch_name == 'PHEFM':
            model = PHEFM_Model(num_classes=3, input_channels=in_channels)

        # Train
        accuracy = train_model(model, train_loader, val_loader, f"{arch_name}_{feat_name}")
        RESULTS_LOG.append((arch_name, feat_name, accuracy))

# Final Report
print("\n--- FINAL TRAINING REPORT ---")
for res in RESULTS_LOG:
    print(f"{res[0]} with {res[1]}: {res[2]:.2f}% Best Accuracy")