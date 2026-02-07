import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Dataset Class (Tensile Only) ---
class TensileDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        df = pd.read_csv(csv_file, header=None)
        self.samples = []
        groups = df.groupby(1) # Group ID is column 1
        
        for group_id, group_data in groups:
            files = group_data[4].tolist() # Filename is column 4
            t_map = next((f for f in files if "tensile_map" in f), None)
            
            if t_map:
                if 493 <= group_id <= 517:
                    label = 0 # Abaca
                elif 518 <= group_id <= 572:
                    label = 1 # Daratex
                else:
                    continue
                self.samples.append({'t': t_map, 'label': label})
        print(f"Evaluation dataset loaded: {len(self.samples)} samples found.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        t_img = Image.open(os.path.join(self.root_dir, 'tensile_map', s['t'])).convert('L')
        if self.transform: t_img = self.transform(t_img)
        return t_img, s['label']

# --- 2. Model Architecture ---
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

# --- 3. Evaluation Function ---
def test_evaluation(model_path, csv_file, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms (Must match training exactly)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load Data
    dataset = TensileDataset(csv_file, data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Load Model
    model = TensileCNN().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    all_preds = []
    all_labels = []

    print("Analyzing Tensile samples...")
    with torch.no_grad():
        for t_img, labels in loader:
            t_img, labels = t_img.to(device), labels.to(device)
            outputs = model(t_img)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Results
    target_names = ['Abaca', 'Daratex']
    print("\n" + "="*40)
    print("TENSILE-ONLY MODEL PERFORMANCE")
    print("="*40)
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix: Tensile-Only Model')
    plt.show()

if __name__ == "__main__":
    test_evaluation(
        model_path='tensile_only_model.pth', 
        csv_file='db_export.csv', 
        data_dir='./labeled_dataset'
    )