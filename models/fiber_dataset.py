from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class FiberDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load CSV (assuming index 1 is group_id, index 4 is filename)
        df = pd.read_csv(csv_file, header=None)
        
        # Group by group_id (the second column)
        self.samples = []
        groups = df.groupby(1)
        
        for group_id, group_data in groups:
            files = group_data[4].tolist()
            
            # Find the specific maps in the group
            l_map = next((f for f in files if "luster_map" in f), None)
            r_map = next((f for f in files if "roughness_proxy" in f), None)
            t_map = next((f for f in files if "tensile_map" in f), None)
            
            if l_map and r_map and t_map:
                # Assign labels based on your provided ranges
                if 493 <= group_id <= 517:
                    label = 0 # Abaca
                elif 518 <= group_id <= 572:
                    label = 1 # Daratex
                else:
                    continue # Skip IDs outside your range
                
                self.samples.append({
                    'l': l_map, 'r': r_map, 't': t_map, 'label': label
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # Load grayscale images
        l_img = Image.open(os.path.join(self.root_dir, 'luster_map', s['l'])).convert('L')
        r_img = Image.open(os.path.join(self.root_dir, 'roughness_proxy', s['r'])).convert('L')
        t_img = Image.open(os.path.join(self.root_dir, 'tensile_map', s['t'])).convert('L')
        
        if self.transform:
            l_img = self.transform(l_img)
            r_img = self.transform(r_img)
            t_img = self.transform(t_img)
            
        return l_img, r_img, t_img, s['label']