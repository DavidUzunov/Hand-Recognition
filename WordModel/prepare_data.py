import os
import zipfile
import requests
from pathlib import Path
import pandas as pd
def download_ms_asl(target_dir="/home/david/Documents/projects/SBHacks-XII/Get-Data"):
    # 1. Setup Paths
    # Using Path.expanduser() or absolute resolve to handle the ".." safely
    base_path = Path(target_dir).expanduser().resolve()
    base_path.mkdir(parents=True, exist_ok=True)
    
    zip_path = base_path / "ms_asl_dataset.zip"
    
    # 2. Download URL (Updated to official MS-ASL link)
    url = "https://download.microsoft.com/download/3/c/a/3ca92c78-1c4a-4a91-a7ee-6980c1d242ec/MS-ASL.zip"
    
    print(f"Downloading dataset to {zip_path}...")
    
    # Use requests for better progress handling and reliability than urllib
    response = requests.get(url, stream=True)
    response.raise_for_status() # Check for download errors
    
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    # 3. Extracting
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_path)
    
    # Optional: Clean up the zip file after extraction
    # os.remove(zip_path)
    print(f"Success! Dataset available at: {base_path}")

if __name__ == "__main__":
    download_ms_asl()
    train_df = pd.read_json("/home/david/Documents/projects/SBHacks-XII/Get-Data/MS-ASL/MSASL_train.json")
    print(train_df.head())