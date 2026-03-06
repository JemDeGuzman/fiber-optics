import os
import pandas as pd
import shutil
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split

# --- 1. Configuration ---
csv_batch = 'batch_16_samples.csv' 
source_folder = '../uploads'        
output_base = './labeled_dataset'
TRAIN_RATIO = 0.8 

CATEGORIES = ['Abaca', 'Daratex', 'False', 'Mixed']
# These must match the prefixes in your 'fileName' column
SUB_FOLDERS = ['luster_map', 'roughness_proxy', 'sample', 'tensile_map']
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def organize_batch_16_by_timestamp():
    if not os.path.exists(csv_batch):
        print(f"Error: {csv_batch} not found.")
        return

    # Load CSV with headers (since your CSV has id, sampleId, etc.)
    df = pd.read_csv(csv_batch)
    
    # 2. Setup Directory Structure
    for split in ['train', 'val']:
        for sub in SUB_FOLDERS:
            os.makedirs(os.path.join(output_base, split, 'Mixed', sub), exist_ok=True)

    # 3. Split by Timestamps (sampleId)
    unique_timestamps = df['sampleId'].unique()
    train_ts, val_ts = train_test_split(unique_timestamps, train_size=TRAIN_RATIO, random_state=42)
    
    split_map = {ts: 'train' for ts in train_ts}
    split_map.update({ts: 'val' for ts in val_ts})

    print(f"Total Unique Samples: {len(unique_timestamps)}")

    success_count = 0
    missing_count = 0

    # 4. Process Rows using exact DB column names
    for _, row in df.iterrows():
        try:
            # Using your DB column names: sampleId, imageUrl, fileName
            ts = row['sampleId']
            split = split_map.get(ts, 'val')
            file_name_val = str(row['fileName'])
            
            # Identify subfolder (e.g., 'luster_map' from 'luster_map-123.jpg')
            img_type = file_name_val.split('-')[0]
            
            if img_type not in SUB_FOLDERS:
                continue

            # Get the raw filename from the URL to find it in ../uploads
            original_filename = os.path.basename(urlparse(str(row['imageUrl'])).path)
            
            source_path = os.path.join(source_folder, original_filename)
            target_path = os.path.join(output_base, split, 'Mixed', img_type, file_name_val)

            if os.path.exists(source_path):
                shutil.move(source_path, target_path)
                success_count += 1
            elif os.path.exists(target_path):
                # Already moved in a previous run
                success_count += 1
            else:
                missing_count += 1
                
        except Exception as e:
            print(f"Error at timestamp {row.get('sampleId')}: {e}")

    # 5. Cleanup Leftover Images
    print("\nCleaning up source folder...")
    deleted = 0
    for f in os.listdir(source_folder):
        f_path = os.path.join(source_folder, f)
        if os.path.isfile(f_path) and f.lower().endswith(IMAGE_EXTENSIONS):
            os.remove(f_path)
            deleted += 1

    print("-" * 30)
    print(f"Process Complete!")
    print(f"Moved to Mixed: {success_count} files")
    print(f"Missing: {missing_count}")
    print(f"Leftovers Deleted: {deleted}")

if __name__ == "__main__":
    organize_batch_16_by_timestamp()