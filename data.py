import os
import requests
from datetime import datetime, timedelta

BASE_URL = "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
TARGET_FOLDER = "data"
DATASETS = [
    "DISPATCHPRICE",
    "DISPATCHREGIONSUM",
    "PREDISPATCHPRICE",
    "PREDISPATCHPRICESENSITIVITIE",
    "PREDISPATCHREGIONSUM",
    "TRADINGPRICE"
]

# Ensure data folder exists
os.makedirs(TARGET_FOLDER, exist_ok=True)

# Loop through each day from 2018-01-01 to 2024-7-31
start_date = datetime(2018, 1, 1)
end_date = datetime(2024, 7, 31)
delta = timedelta(days=1)

current = start_date
while current <= end_date:
    yyyymm = current.strftime("%Y%m")
    yyyymmdd = current.strftime("%Y%m%d")
    for dataset in DATASETS:
        # This naming pattern is based on actual AEMO file format
        if dataset in {"PREDISPATCHPRICE", "PREDISPATCHPRICESENSITIVITIE", "PREDISPATCHREGIONSUM"}:
            filename = f"PUBLIC_DVD_{dataset}_D_{yyyymm}010000.zip"
        else:
            filename = f"PUBLIC_DVD_{dataset}_{yyyymm}010000.zip"
        file_url = f"{BASE_URL}/{current.year}/MMSDM_{current.year}_{current.strftime('%m')}/MMSDM_Historical_Data_SQLLoader/DATA/{filename}"
        local_path = os.path.join(TARGET_FOLDER, filename)

        if os.path.exists(local_path):
            print(f"Already downloaded: {filename}")
            continue

        print(f"Trying {file_url}")
        try:
            response = requests.get(file_url, stream=True, timeout=30)
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✅ Downloaded: {filename}")
            else:
                print(f"❌ Not found: {filename}")
        except Exception as e:
            print(f"⚠️ Error downloading {filename}: {e}")

    current += delta

# Unzip all downloaded files
import zipfile

print("\nStarting extraction of zip files...\n")

for filename in os.listdir(TARGET_FOLDER):
    if filename.endswith(".zip"):
        zip_path = os.path.join(TARGET_FOLDER, filename)
        extract_folder = os.path.join(TARGET_FOLDER, filename.replace(".zip", ""))

        if os.path.exists(extract_folder):
            print(f"Already extracted: {extract_folder}")
            continue

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
                print(f"✅ Extracted: {filename} ➝ {extract_folder}")
        except zipfile.BadZipFile:
            print(f"❌ Bad zip file: {filename}")


# Move all CSV files from subfolders to main data folder and remove empty subfolders
import shutil

print("\nFlattening extracted CSVs and removing subfolders...\n")

for root, dirs, files in os.walk(TARGET_FOLDER, topdown=False):
    for file in files:
        if file.lower().endswith(".csv"):
            src = os.path.join(root, file)
            dst = os.path.join(TARGET_FOLDER, file)
            if src != dst:
                shutil.move(src, dst)
    # Try to remove the folder if it's not the root and is now empty
    if root != TARGET_FOLDER:
        try:
            os.rmdir(root)
            print(f"✅ Removed folder: {root}")
        except OSError:
            print(f"⚠️ Folder not empty or could not be removed: {root}")

print("✅ All CSVs moved. Attempted to remove all empty subfolders.")