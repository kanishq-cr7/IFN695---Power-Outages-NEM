import os
import pandas as pd
from glob import glob

input_folder = 'nem_data/DISPATCHREGIONSUM'
output_folder = 'nem_data/DISPATCHREGIONSUM_CLEANED'

os.makedirs(output_folder, exist_ok=True)

# Match both .csv and .CSV
csv_files = glob(os.path.join(input_folder, '*.csv')) + glob(os.path.join(input_folder, '*.CSV'))

relevant_columns = [
    # Time and Location
    'SETTLEMENTDATE', 'REGIONID',

    # Demand & Supply
    'TOTALDEMAND', 'AVAILABLEGENERATION', 'DEMANDFORECAST', 'DISPATCHABLEGENERATION',
    'DISPATCHABLELOAD', 'INITIALSUPPLY', 'CLEAREDSUPPLY', 'DEMAND_AND_NONSCHEDGEN',
    'TOTALINTERMITTENTGENERATION',

    # Non-null FCAS columns
    'LOWER5MINLOCALDISPATCH', 'LOWER60SECLOCALDISPATCH', 'LOWER6SECLOCALDISPATCH',
    'RAISE5MINLOCALDISPATCH', 'RAISE60SECLOCALDISPATCH', 'RAISE6SECLOCALDISPATCH',
    'AGGREGATEDISPATCHERROR', 'LOWERREGLOCALDISPATCH', 'RAISEREGLOCALDISPATCH',
    'RAISE6SECACTUALAVAILABILITY', 'RAISE60SECACTUALAVAILABILITY', 'RAISE5MINACTUALAVAILABILITY',
    'RAISEREGACTUALAVAILABILITY', 'LOWER6SECACTUALAVAILABILITY', 'LOWER60SECACTUALAVAILABILITY',
    'LOWER5MINACTUALAVAILABILITY', 'LOWERREGACTUALAVAILABILITY'
]

for file in csv_files:
    try:
        print(f"🔍 Processing {os.path.basename(file)}")

        # Read header line (second line in the file)
        with open(file, 'r') as f:
            f.readline()  # Skip first line
            header_line = f.readline().strip()
            header = [val if val != '' else 'NaN' for val in header_line.split(',')]

        # Read from second line as header
        df = pd.read_csv(file, skiprows=1)

        # Filter to retain only relevant columns that exist in this file
        available_cols = [col for col in relevant_columns if col in df.columns]
        df_filtered = df[available_cols]
        if 'SETTLEMENTDATE' in df_filtered.columns:
            df_filtered['SETTLEMENTDATE'] = pd.to_datetime(df_filtered['SETTLEMENTDATE'], errors='coerce')

        # Save cleaned file
        output_path = os.path.join(output_folder, os.path.basename(file))
        df_filtered.to_csv(output_path, index=False)

        print(f"✅ Saved cleaned file to {output_path}")
    except Exception as e:
        print(f"Error processing {file}: {e}")


# Merge all cleaned files
print("\nStarting merge of all cleaned DISPATCHREGIONSUM files...")

# Get all cleaned CSVs
cleaned_files = glob(os.path.join(output_folder, '*.CSV')) + glob(os.path.join(output_folder, '*.csv'))

# Read and store all DataFrames
all_dfs = []
for f in cleaned_files:
    try:
        df = pd.read_csv(f, parse_dates=['SETTLEMENTDATE'], dayfirst=False, infer_datetime_format=True)
        all_dfs.append(df)
    except Exception as e:
        print(f"Error reading {f}: {e}")

if all_dfs:
    # Concatenate and sort
    df_merged = pd.concat(all_dfs, ignore_index=True)
    df_merged.sort_values(by=['REGIONID', 'SETTLEMENTDATE'], inplace=True)

    # Save merged file
    merged_path = os.path.join(output_folder, 'MERGED_DISPATCHREGIONSUM.csv')
    df_merged.to_csv(merged_path, index=False)

    print(f"Merged dataset saved to: {merged_path}")
else:
    print("No cleaned files found to merge.")

# Convert merged file to daily scale
print("\nConverting merged DISPATCHREGIONSUM data to daily scale...")

try:
    df_merged = pd.read_csv(merged_path, parse_dates=['SETTLEMENTDATE'])
    df_merged['DATE'] = df_merged['SETTLEMENTDATE'].dt.date

    df_daily = df_merged.groupby(['DATE', 'REGIONID']).agg({
        'TOTALDEMAND': 'mean',
        'AVAILABLEGENERATION': 'mean',
        'DEMANDFORECAST': 'mean',
        'DISPATCHABLEGENERATION': 'mean',
        'DISPATCHABLELOAD': 'mean',
        'INITIALSUPPLY': 'mean',
        'CLEAREDSUPPLY': 'mean',
        'DEMAND_AND_NONSCHEDGEN': 'mean',
        'TOTALINTERMITTENTGENERATION': 'mean',
        'LOWER5MINLOCALDISPATCH': 'mean',
        'LOWER60SECLOCALDISPATCH': 'mean',
        'LOWER6SECLOCALDISPATCH': 'mean',
        'RAISE5MINLOCALDISPATCH': 'mean',
        'RAISE60SECLOCALDISPATCH': 'mean',
        'RAISE6SECLOCALDISPATCH': 'mean',
        'AGGREGATEDISPATCHERROR': 'mean',
        'LOWERREGLOCALDISPATCH': 'mean',
        'RAISEREGLOCALDISPATCH': 'mean',
        'RAISE6SECACTUALAVAILABILITY': 'mean',
        'RAISE60SECACTUALAVAILABILITY': 'mean',
        'RAISE5MINACTUALAVAILABILITY': 'mean',
        'RAISEREGACTUALAVAILABILITY': 'mean',
        'LOWER6SECACTUALAVAILABILITY': 'mean',
        'LOWER60SECACTUALAVAILABILITY': 'mean',
        'LOWER5MINACTUALAVAILABILITY': 'mean',
        'LOWERREGACTUALAVAILABILITY': 'mean'
    }).reset_index()

    daily_output_path = os.path.join(output_folder, 'DAILY_DISPATCHREGIONSUM.csv')
    df_daily.to_csv(daily_output_path, index=False)
    print(f"Daily aggregated dataset saved to: {daily_output_path}")
except Exception as e:
    print(f"Error during daily aggregation: {e}")