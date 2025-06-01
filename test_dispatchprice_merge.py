import os
import pandas as pd
from glob import glob

input_folder = 'nem_data/DISPATCHPRICE'
output_folder = 'nem_data/DISPATCHPRICE_CLEANED'

os.makedirs(output_folder, exist_ok=True)

csv_files = glob(os.path.join(input_folder, '*.csv')) + glob(os.path.join(input_folder, '*.CSV'))

relevant_columns = [
    'SETTLEMENTDATE', 'REGIONID', 'RRP', 'ROP',
    'RAISE6SECRRP', 'RAISE60SECRRP', 'RAISE5MINRRP', 'RAISEREGRRP', 'RAISE1SECRRP',
    'RAISE6SECROP', 'RAISE60SECROP', 'RAISE5MINROP', 'RAISEREGROP', 'RAISE1SECROP',
    'LOWER6SECRRP', 'LOWER60SECRRP', 'LOWER5MINRRP', 'LOWERREGRRP',
    'LOWER6SECROP', 'LOWER60SECROP', 'LOWER5MINROP', 'LOWERREGROP',
    'PRE_AP_ENERGY_PRICE', 'PRE_AP_RAISE6_PRICE', 'PRE_AP_RAISE60_PRICE', 'PRE_AP_RAISE5MIN_PRICE',
    'PRE_AP_RAISEREG_PRICE', 'PRE_AP_LOWER6_PRICE', 'PRE_AP_LOWER60_PRICE', 'PRE_AP_LOWER5MIN_PRICE',
    'PRE_AP_LOWERREG_PRICE', 'PRE_AP_RAISE1_PRICE',
    'CUMUL_PRE_AP_ENERGY_PRICE', 'CUMUL_PRE_AP_RAISE6_PRICE', 'CUMUL_PRE_AP_RAISE60_PRICE',
    'CUMUL_PRE_AP_RAISE5MIN_PRICE', 'CUMUL_PRE_AP_RAISEREG_PRICE', 'CUMUL_PRE_AP_LOWER6_PRICE',
    'CUMUL_PRE_AP_LOWER60_PRICE', 'CUMUL_PRE_AP_LOWER5MIN_PRICE', 'CUMUL_PRE_AP_LOWERREG_PRICE',
    'CUMUL_PRE_AP_RAISE1_PRICE', 'CUMUL_PRE_AP_LOWER1_PRICE'
]

for file in csv_files:
    try:
        print(f"Processing {os.path.basename(file)}")

        with open(file, 'r') as f:
            f.readline()  # skip first row
            header_line = f.readline().strip()
            header = [val if val else 'NaN' for val in header_line.split(',')]

        df = pd.read_csv(file, skiprows=1)

        available_cols = [col for col in relevant_columns if col in df.columns]
        df_filtered = df[available_cols]

        if 'SETTLEMENTDATE' in df_filtered.columns:
            df_filtered.loc[:, 'SETTLEMENTDATE'] = pd.to_datetime(df_filtered['SETTLEMENTDATE'], errors='coerce')

        output_path = os.path.join(output_folder, os.path.basename(file))
        df_filtered.to_csv(output_path, index=False)

        print(f"Saved cleaned file to {output_path}")
    except Exception as e:
        print(f"Error processing {file}: {e}")

print("\n📦 Starting merge of all cleaned DISPATCHPRICE files...")

cleaned_files = glob(os.path.join(output_folder, '*.csv')) + glob(os.path.join(output_folder, '*.CSV'))

all_dfs = []
for f in cleaned_files:
    try:
        df = pd.read_csv(f, parse_dates=['SETTLEMENTDATE'])
        all_dfs.append(df)
    except Exception as e:
        print(f"Error reading {f}: {e}")

if all_dfs:
    df_merged = pd.concat(all_dfs, ignore_index=True)
    df_merged.sort_values(by=['REGIONID', 'SETTLEMENTDATE'], inplace=True)

    merged_path = os.path.join(output_folder, 'MERGED_DISPATCHPRICE.csv')
    df_merged.to_csv(merged_path, index=False)
    print(f"Merged dataset saved to: {merged_path}")
else:
    print("No cleaned files found to merge.")

print("\n Converting merged DISPATCHPRICE data to daily scale...")

try:
    df_merged = pd.read_csv(merged_path, low_memory=False)
    df_merged['SETTLEMENTDATE'] = pd.to_datetime(df_merged['SETTLEMENTDATE'], errors='coerce')
    df_merged['DATE'] = df_merged['SETTLEMENTDATE'].dt.date

    df_daily = df_merged.groupby(['DATE', 'REGIONID']).agg('mean').reset_index()
    df_daily.drop(columns=[
        'RAISE1SECRRP', 'RAISE1SECROP', 'PRE_AP_RAISE1_PRICE',
        'CUMUL_PRE_AP_RAISE1_PRICE', 'CUMUL_PRE_AP_LOWER1_PRICE'
    ], inplace=True, errors='ignore')

    daily_output_path = os.path.join(output_folder, 'DAILY_DISPATCHPRICE.csv')
    df_daily.to_csv(daily_output_path, index=False)
    print(f"Daily aggregated dataset saved to: {daily_output_path}")
except Exception as e:
    print(f"Error during daily aggregation: {e}")