import os
import pandas as pd
import shutil
from urllib.parse import urlparse

# 1. Configuration
csv_file = 'db_export.csv'        # The file you uploaded to me
source_folder = '../uploads'       # Folder where your raw images currently are
output_base = './labeled_dataset' # Where you want them moved/renamed

# Define column indices based on your file structure:
# Column 2: URL (http://.../1770093493431-b8evdp.jpg)
# Column 4: New Name (sample-1770092550849.jpg)
URL_COL = 2
NAME_COL = 4

def organize_dataset():
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please place it in this directory.")
        return

    # Load CSV without headers
    df = pd.read_csv(csv_file, header=None)
    os.makedirs(output_base, exist_ok=True)

    print(f"Starting processing of {len(df)} records...")
    
    success_count = 0
    missing_count = 0

    for index, row in df.iterrows():
        try:
            url = str(row[URL_COL])
            target_name = str(row[NAME_COL])
            
            # Extract the actual filename from the URL path
            # Example: '1770093493431-b8evdp.jpg'
            original_filename = os.path.basename(urlparse(url).path)
            
            # Create subfolders based on type (sample, luster_map, etc.)
            category = target_name.split('-')[0]
            category_dir = os.path.join(output_base, category)
            os.makedirs(category_dir, exist_ok=True)

            source_path = os.path.join(source_folder, original_filename)
            target_path = os.path.join(category_dir, target_name)

            if os.path.exists(source_path):
                shutil.copy2(source_path, target_path)
                success_count += 1
            else:
                missing_count += 1
                
        except Exception as e:
            print(f"Error processing row {index}: {e}")

    print("-" * 30)
    print(f"Organization Complete!")
    print(f"Successfully copied: {success_count} files")
    print(f"Files not found in source: {missing_count}")
    print(f"Your data is now ready in: {os.path.abspath(output_base)}")

if __name__ == "__main__":
    organize_dataset()