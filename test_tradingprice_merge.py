import os
import glob
import pandas as pd

input_folder = 'nem_data/TRADINGPRICE'
output_folder = 'nem_data/TRADINGPRICE_CLEANED'
os.makedirs(output_folder, exist_ok=True)

relevant_columns = [
    'SETTLEMENTDATE', 'REGIONID', 'RRP', 'ROP',
    'RAISE6SECRRP', 'RAISE6SECROP', 'RAISE60SECRRP', 'RAISE60SECROP',
    'RAISE5MINRRP', 'RAISE5MINROP', 'RAISEREGRRP', 'RAISEREGROP',
    'LOWER6SECRRP', 'LOWER6SECROP', 'LOWER60SECRRP', 'LOWER60SECROP',
    'LOWER5MINRRP', 'LOWER5MINROP', 'LOWERREGRRP', 'LOWERREGROP',
    'RAISE1SECRRP', 'RAISE1SECROP', 'LOWER1SECRRP', 'LOWER1SECROP'
]

# Find all CSV files (case insensitive)
csv_files = glob.glob(os.path.join(input_folder, '*.csv')) + glob.glob(os.path.join(input_folder, '*.CSV'))

# Clean each file
for file in csv_files:
    df = pd.read_csv(file, skiprows=1)
    cols_to_keep = [col for col in relevant_columns if col in df.columns]
    df = df[cols_to_keep]
    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
    base_name = os.path.basename(file)
    df.to_csv(os.path.join(output_folder, base_name), index=False)

# Merge cleaned files
cleaned_files = glob.glob(os.path.join(output_folder, '*.csv')) + glob.glob(os.path.join(output_folder, '*.CSV'))
merged_dfs = []
for file in cleaned_files:
    df = pd.read_csv(file, parse_dates=['SETTLEMENTDATE'])
    merged_dfs.append(df)

merged_df = pd.concat(merged_dfs, ignore_index=True)
merged_df.sort_values(['REGIONID', 'SETTLEMENTDATE'], inplace=True)
merged_df.to_csv('MERGED_TRADINGPRICE.csv', index=False)

# Convert to daily scale
merged_df['DATE'] = merged_df['SETTLEMENTDATE'].dt.date
daily_df = merged_df.groupby(['DATE', 'REGIONID']).mean().reset_index()
daily_df.drop(columns=['RAISE1SECRRP', 'RAISE1SECROP', 'LOWER1SECRRP', 'LOWER1SECROP'], errors='ignore', inplace=True)
daily_df.to_csv(os.path.join(output_folder, 'DAILY_TRADINGPRICE.csv'), index=False)
