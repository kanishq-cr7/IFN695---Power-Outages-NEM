import os
import pandas as pd
from pprint import pprint

# Set path to your data folder
data_dir = os.path.join(os.path.dirname(__file__), "nem_data")

# Create dictionaries to hold categorized DataFrames
aemo_data = {
    "network_outages": [],
    "dispatch_price": [],
    "trading_price": [],
    "dispatch_regionsum": [],
    "network_status": None,
}
weather_data = {}

# Directly load pre-cleaned daily files
aemo_data["dispatch_regionsum"] = pd.read_csv(os.path.join(data_dir, "DISPATCHREGIONSUM_CLEANED", "DAILY_DISPATCHREGIONSUM.csv"), parse_dates=["DATE"])
aemo_data["dispatch_price"] = pd.read_csv(os.path.join(data_dir, "DISPATCHPRICE_CLEANED", "DAILY_DISPATCHPRICE.csv"), parse_dates=["DATE"])
aemo_data["trading_price"] = pd.read_csv(os.path.join(data_dir, "TRADINGPRICE_CLEANED", "DAILY_TRADINGPRICE.csv"), parse_dates=["DATE"])

# Loop through files in the folder
for file in os.listdir(data_dir):
    file_path = os.path.join(data_dir, file)
    
    if not file.lower().endswith('.csv'):
        continue  # skip non-CSV files

    # Normalize filename for reliable matching
    normalized_name = os.path.basename(file).upper().replace("#", "_")

    print(f"Checking file: {file} → {normalized_name}")

    # AEMO Tables (flexible pattern match)
    # Only load NETWORK_OUTAGEDETAIL once from the full-history file (202503)
    if "NETWORK_OUTAGEDETAIL" in normalized_name:
        print(f"→ Loading full-history outage file: {file_path}")
        correct_headers = [
            "SETROWID", "DVD_NETWORK_OUTAGEDETAIL", "AEMO", "PUBLIC", "OUTAGEID",
            "SUBSTATIONID", "EQUIPMENTTYPE", "EQUIPMENTID", "STARTTIME", "ENDTIME",
            "SUBMITTEDDATE", "OUTAGESTATUSCODE", "SUBMITREASON", "RESUBMITOUTAGEID",
            "RECALLTIMEDAY", "RECALLTIMENIGHT", "LASTCHANGED", "REASON", "ISSECONDARY",
            "ACTUAL_STARTTIME", "ACTUAL_ENDTIME", "COMPANYREFCODE", "ELEMENTID"
        ]
        df_outage = pd.read_csv(file_path, skiprows=1, names=correct_headers)
        aemo_data["network_outages"].append(df_outage)
        # Save cleaned file
        cleaned_path = os.path.join(data_dir, "network_outage_detail_cleaned.csv")
        df_outage.to_csv(cleaned_path, index=False)
        print(f"Saved cleaned network outage detail to {cleaned_path}")
    if "NETWORK_OUTAGESTATUSCODE" in normalized_name:
        print(f"→ Loading outage status code file: {file_path}")
        aemo_data["network_status"] = pd.read_csv(file_path)
        # Remove the first row from the STATUSCODE dataset
        aemo_data["network_status"] = aemo_data["network_status"].iloc[1:]
        print("\nData types in saved network_outage_status_cleaned.csv:")
        status_path = os.path.join(data_dir, "network_outage_status_cleaned.csv")
        print(pd.read_csv(status_path).dtypes)

# Weather Files (label by filename)
weather_dir = os.path.join(data_dir, 'weather')
for file in os.listdir(weather_dir):
    file_path = os.path.join(weather_dir, file)
    if not file.lower().endswith('.csv'):
        continue  # skip non-CSV files
    if any(loc in file.lower() for loc in ["brisbane", "sydney", "melbourne", "adelaide", "hobart"]):
        key = file.replace('.csv', '')
        weather_data[key] = pd.read_csv(file_path)

# Combine AEMO files into single DataFrames
for key in aemo_data:
    if isinstance(aemo_data[key], list) and aemo_data[key]:  # if list not empty
        aemo_data[key] = pd.concat(aemo_data[key], ignore_index=True)
        print(f"{key} loaded: shape = {aemo_data[key].shape}")
    elif aemo_data[key] is None or (isinstance(aemo_data[key], list) and not aemo_data[key]):
        print(f"{key} not found.")

# Example: print weather file keys
print("\nAvailable weather files:", list(weather_data.keys()))

# Filter and merge weather data by city
filtered_weather = {}

for city in ["brisbane", "sydney", "melbourne", "adelaide", "hobart"]:
    try:
        min_key = f"{city}_minTemp"
        max_key = f"{city}_maxTemp"
        rain_key = f"{city}_rainfall"

        min_df = weather_data[min_key]
        max_df = weather_data[max_key]

        # Fix for Adelaide rainfall casing
        if rain_key not in weather_data:
            alt_rain_key = [k for k in weather_data if city.lower() in k.lower() and "rainfall" in k.lower()]
            if alt_rain_key:
                rain_key = alt_rain_key[0]
        rain_df = weather_data[rain_key]

        # Create unified Date column from Year, Month, Day
        for df in [min_df, max_df, rain_df]:
            df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])

        # Filter to 2018-2024
        start = pd.Timestamp("2018-01-01")
        end = pd.Timestamp("2024-12-31")
        min_df = min_df[(min_df["Date"] >= start) & (min_df["Date"] <= end)]
        max_df = max_df[(max_df["Date"] >= start) & (max_df["Date"] <= end)]
        rain_df = rain_df[(rain_df["Date"] >= start) & (rain_df["Date"] <= end)]

        # Rename value columns
        min_temp_col = [c for c in min_df.columns if "temperature" in c.lower()][0]
        max_temp_col = [c for c in max_df.columns if "temperature" in c.lower()][0]
        rainfall_col = [c for c in rain_df.columns if "rainfall" in c.lower()][0]

        min_df = min_df.rename(columns={min_temp_col: "MinTemp"})
        max_df = max_df.rename(columns={max_temp_col: "MaxTemp"})
        rain_df = rain_df.rename(columns={rainfall_col: "Rainfall"})

        # Merge on Date
        merged = min_df[["Date", "MinTemp"]].merge(
            max_df[["Date", "MaxTemp"]], on="Date"
        ).merge(
            rain_df[["Date", "Rainfall"]], on="Date"
        )
        merged.interpolate(method='linear', inplace=True)
        filtered_weather[city] = merged
        print(f"Merged weather for {city}: shape = {merged.shape}")
    except Exception as e:
        print(f"Error merging weather for {city}: {e}")


# Save merged weather data to CSV files
for city, df in filtered_weather.items():
    output_path = os.path.join(data_dir, f"{city}_weather_merged.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved merged weather data for {city} to {output_path}")

# --- Assign RegionID and combine weather data ---
print("\n Assigning RegionID to weather data and merging into one DataFrame...")
city_region_map = {
    "brisbane": "QLD1",
    "sydney": "NSW1",
    "adelaide": "SA1",
    "melbourne": "VIC1",
    "hobart": "TAS1"
}

regional_weather_dfs = []
for city, region_id in city_region_map.items():
    weather_file = os.path.join(data_dir, f"{city}_weather_merged.csv")
    if os.path.exists(weather_file):
        df = pd.read_csv(weather_file, parse_dates=["Date"])
        df["RegionID"] = region_id
        regional_weather_dfs.append(df)
    else:
        print(f"⚠️ Weather file for {city} not found at {weather_file}")

if regional_weather_dfs:
    master_weather_df = pd.concat(regional_weather_dfs, ignore_index=True)
    master_weather_path = os.path.join(data_dir, "master_weather_data.csv")
    master_weather_df.to_csv(master_weather_path, index=False)
    print(f"Combined weather data with RegionID saved to {master_weather_path}")
else:
    print("No regional weather files were combined.")

# Reopen and drop the first row from cleaned outage detail
df_outage_cleaned = pd.read_csv(cleaned_path)
df_outage_cleaned = df_outage_cleaned.iloc[1:]
df_outage_cleaned.drop(columns=[
    "SETROWID", "DVD_NETWORK_OUTAGEDETAIL", "AEMO", "PUBLIC",
    "RESUBMITOUTAGEID", "RECALLTIMEDAY", "RECALLTIMENIGHT",
    "LASTCHANGED", "SUBMITTEDDATE", "COMPANYREFCODE",
    "ACTUAL_STARTTIME", "ACTUAL_ENDTIME"
], inplace=True)
df_outage_cleaned = df_outage_cleaned.iloc[:-1]
df_outage_cleaned.to_csv(cleaned_path, index=False)
print("\nData types in saved network_outage_detail_cleaned.csv:")
print(pd.read_csv(cleaned_path).dtypes)
print(f"Removed specified columns and last row, then re-saved cleaned network outage detail to {cleaned_path}")

# Convert relevant datetime columns to datetime format
for col in ["STARTTIME", "ENDTIME"]:
    if col in df_outage_cleaned.columns:
        df_outage_cleaned[col] = pd.to_datetime(df_outage_cleaned[col], format="%Y/%m/%d %H:%M:%S", errors="coerce")

# Filter to keep only outages from 2018 to 2024
start_date = pd.Timestamp("2018-01-01")
end_date = pd.Timestamp("2024-12-31")
df_outage_cleaned = df_outage_cleaned[
    (df_outage_cleaned["STARTTIME"] >= start_date) &
    (df_outage_cleaned["STARTTIME"] <= end_date)
]
df_outage_cleaned = df_outage_cleaned.sort_values(by="STARTTIME")

# Drop rows where SUBSTATIONID is null or empty
df_outage_cleaned = df_outage_cleaned[df_outage_cleaned["SUBSTATIONID"].notna() & (df_outage_cleaned["SUBSTATIONID"].str.strip() != "")]
print(f"Rows after removing null/empty SUBSTATIONID: {df_outage_cleaned.shape}")

# Save again after datetime conversion and filtering
df_outage_cleaned.to_csv(cleaned_path, index=False)
print(f"Converted datetime columns and re-saved to {cleaned_path}")
print("\nUpdated datatypes for datetime columns:")
print(df_outage_cleaned[["STARTTIME", "ENDTIME"]].dtypes)

# Generate unique substation list for manual REGIONID mapping
unique_substations = df_outage_cleaned["SUBSTATIONID"].dropna().unique()
unique_substations.sort()
substation_df = pd.DataFrame(unique_substations, columns=["SUBSTATIONID"])
substation_map_path = os.path.join(data_dir, "substation_region_map.csv")
substation_df.to_csv(substation_map_path, index=False)
print(f"\n Unique substation list saved to {substation_map_path}")

# --- Substation to REGIONID fuzzy mapping ---
from fuzzywuzzy import process
import re

# Load transmission_substation data
transmission_path = os.path.join(data_dir, "Transmission_Substations.csv")
df_transmission = pd.read_csv(transmission_path)

# Step 1: Filter to only relevant states
relevant_states = ["Queensland", "New South Wales", "South Australia", "Victoria", "Tasmania"]
df_transmission = df_transmission[df_transmission["state"].isin(relevant_states)].copy()

# Step 2: Normalize substation names in transmission data
def normalize_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r'[^a-z0-9]', '', text.lower())

df_transmission["clean_name"] = df_transmission["name"].apply(normalize_text)

# Step 3: Normalize unique SUBSTATIONIDs and fuzzy match
unique_substations = df_outage_cleaned["SUBSTATIONID"].dropna().unique()
substation_matches = []

for sub_id in unique_substations:
    clean_sub_id = normalize_text(sub_id)
    match_result = process.extractOne(clean_sub_id, df_transmission["clean_name"])
    if match_result:
        best_match = match_result[0]
        score = match_result[1] if len(match_result) > 1 else 0
    else:
        best_match, score = None, 0
    match_row = df_transmission[df_transmission["clean_name"] == best_match].iloc[0]
    region_id = match_row["state"]
    if region_id == "Queensland":
        region_id = "QLD1"
    elif region_id == "New South Wales":
        region_id = "NSW1"
    elif region_id == "South Australia":
        region_id = "SA1"
    elif region_id == "Victoria":
        region_id = "VIC1"
    elif region_id == "Tasmania":
        region_id = "TAS1"
    substation_matches.append({
        "SUBSTATIONID": sub_id,
        "Matched_Station_Name": match_row["name"],
        "State": match_row["state"],
        "RegionID": region_id,
        "MatchScore": score
    })

# Step 4: Export mapping to CSV
mapping_df = pd.DataFrame(substation_matches)
mapping_path = os.path.join(data_dir, "substation_region_mapping.csv")
mapping_df.to_csv(mapping_path, index=False)
print(f" Substation to REGIONID mapping saved to {mapping_path}")

# --- Merge REGIONID info into network_outage_detail_cleaned ---
print("\n🔗 Merging substation-region mapping into network outage details...")
region_map_df = pd.read_csv(mapping_path)
merged_outage_df = df_outage_cleaned.merge(region_map_df, on="SUBSTATIONID", how="left")

# Save merged file
merged_outage_path = os.path.join(data_dir, "network_outage_detail_merged.csv")
merged_outage_df.to_csv(merged_outage_path, index=False)
print(f" Merged outage data with REGIONID saved to {merged_outage_path}")

# Drop OUTAGEID, State, and MatchScore from merged file
print("\n🧹 Dropping columns: OUTAGEID, State, MatchScore")
columns_to_drop = ["OUTAGEID", "State", "MatchScore"]
cleaned_merged_outage_df = merged_outage_df.drop(columns=columns_to_drop, errors="ignore")

# Save cleaned version
cleaned_merged_outage_path = os.path.join(data_dir, "network_outage_detail_merged.csv")
cleaned_merged_outage_df.to_csv(cleaned_merged_outage_path, index=False)
print(f" Cleaned merged outage data saved to {cleaned_merged_outage_path}")

# --- Convert outage events to daily scale, including outage duration ---
print("\n Converting outage events to daily scale with duration...")
from datetime import timedelta

# Expand outages to list of dates and assign daily duration
def expand_outage_dates_with_duration(row):
    try:
        start = pd.to_datetime(row["STARTTIME"])
        end = pd.to_datetime(row["ENDTIME"])
        if pd.isna(start) or pd.isna(end):
            return []
        date_range = pd.date_range(start=start.date(), end=end.date(), freq='D')
        duration = (end - start).total_seconds() / 60  # in minutes

        if len(date_range) == 0:
            return []
        
        per_day_duration = duration / len(date_range)
        return [(d, per_day_duration) for d in date_range]
    except Exception:
        return []

expanded_rows = []
for _, row in cleaned_merged_outage_df.iterrows():
    for day, duration_min in expand_outage_dates_with_duration(row):
        expanded_rows.append({
            "Date": day,
            "SUBSTATIONID": row["SUBSTATIONID"],
            "RegionID": row["RegionID"],
            "DurationMinutes": duration_min
        })

daily_outages_df = pd.DataFrame(expanded_rows)
daily_outage_summary = (
    daily_outages_df
    .groupby(["Date", "RegionID"])
    .agg(OutageCount=("SUBSTATIONID", "count"),
         TotalDurationMinutes=("DurationMinutes", "sum"))
    .reset_index()
    .sort_values(by="Date")
)

# Save to CSV
daily_outage_path = os.path.join(data_dir, "daily_outages_by_region.csv")
daily_outage_summary.to_csv(daily_outage_path, index=False)
print(f" Daily outage summary (with duration) saved to {daily_outage_path}")

# --- Merge all data sources into one master dataset ---
print("\n🔗 Merging weather, outage, and energy data into master dataset...")

# Load all daily datasets
weather_df = pd.read_csv(master_weather_path, parse_dates=["Date"])
outage_df = pd.read_csv(daily_outage_path, parse_dates=["Date"])
dispatch_price_df = pd.read_csv(os.path.join(data_dir, "DISPATCHPRICE_CLEANED", "DAILY_DISPATCHPRICE.csv"), parse_dates=["DATE"])
trading_price_df = pd.read_csv(os.path.join(data_dir, "TRADINGPRICE_CLEANED", "DAILY_TRADINGPRICE.csv"), parse_dates=["DATE"])
dispatch_regionsum_df = pd.read_csv(os.path.join(data_dir, "DISPATCHREGIONSUM_CLEANED", "DAILY_DISPATCHREGIONSUM.csv"), parse_dates=["DATE"])

# Rename DATE to Date for consistency
dispatch_price_df.rename(columns={"DATE": "Date"}, inplace=True)
trading_price_df.rename(columns={"DATE": "Date"}, inplace=True)
dispatch_regionsum_df.rename(columns={"DATE": "Date"}, inplace=True)

# Ensure consistent RegionID column
for df in [dispatch_price_df, trading_price_df, dispatch_regionsum_df]:
    if "REGIONID" in df.columns:
        df.rename(columns={"REGIONID": "RegionID"}, inplace=True)

# Merge all dataframes using outer joins on Date and RegionID
master_df = weather_df.merge(outage_df, on=["Date", "RegionID"], how="outer")
master_df = master_df.merge(dispatch_price_df, on=["Date", "RegionID"], how="outer")
master_df = master_df.merge(trading_price_df, on=["Date", "RegionID"], how="outer")
master_df = master_df.merge(dispatch_regionsum_df, on=["Date", "RegionID"], how="outer")

# Save final master dataset
master_dataset_path = os.path.join(data_dir, "master_dataset.csv")
master_df.to_csv(master_dataset_path, index=False)
print(f" Master dataset saved to {master_dataset_path}")

# --- Visual and Statistical Analysis ---
import matplotlib.pyplot as plt
import seaborn as sns

print("\n Starting visual and statistical analysis...")

master_df = pd.read_csv(master_dataset_path, parse_dates=["Date"])
# Filter to project timeframe: 2018-01-01 to 2024-12-31
master_df = master_df[(master_df["Date"] >= "2018-01-01") & (master_df["Date"] <= "2024-12-31")]

# Summary statistics
print(master_df.describe(include='all'))

# Missing data heatmap
plt.figure(figsize=(16, 6))
sns.heatmap(master_df.isnull(), cbar=False)
plt.title("Missing Values in Master Dataset")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "missing_data_heatmap.png"))
plt.close()

# Daily outage count per region
outage_trend = master_df.groupby(["Date", "RegionID"])["OutageCount"].sum().unstack()
outage_trend.plot(figsize=(15, 5))
plt.title("Daily Outage Count per Region")
plt.ylabel("Outage Count")
plt.xlabel("Date")
plt.legend(title="RegionID")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "daily_outage_count_per_region.png"))
plt.close()

# Daily total outage duration per region
duration_trend = master_df.groupby(["Date", "RegionID"])["TotalDurationMinutes"].sum().unstack()
duration_trend.plot(figsize=(15, 5))
plt.title("Daily Total Outage Duration (Minutes) per Region")
plt.ylabel("Duration (min)")
plt.xlabel("Date")
plt.legend(title="RegionID")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "daily_outage_duration_per_region.png"))
plt.close()


# Correlation between weather and outage variables
corr_cols = ["MinTemp", "MaxTemp", "Rainfall", "OutageCount", "TotalDurationMinutes"]
plt.figure(figsize=(8, 6))
sns.heatmap(master_df[corr_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation between Weather and Outage Variables")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "weather_outage_correlation.png"))
plt.close()

# --- Seasonal Correlation Heatmaps: Summer vs Winter ---
master_df["Month"] = master_df["Date"].dt.month
summer_df = master_df[master_df["Month"].isin([12, 1, 2])]
winter_df = master_df[master_df["Month"].isin([6, 7, 8])]

# Summer correlation
plt.figure(figsize=(8, 6))
sns.heatmap(summer_df[corr_cols].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Summer Correlation: Weather vs Outages")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "summer_weather_outage_correlation.png"))
plt.close()

# Winter correlation
plt.figure(figsize=(8, 6))
sns.heatmap(winter_df[corr_cols].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Winter Correlation: Weather vs Outages")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "winter_weather_outage_correlation.png"))
plt.close()


# Rainfall vs OutageCount scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=master_df, x="Rainfall", y="OutageCount", hue="RegionID", alpha=0.6)
plt.title("Rainfall vs Outage Count")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "rainfall_vs_outagecount.png"))
plt.close()

# --- Binned Rainfall vs Average Outage Count ---
import numpy as np

# Define rainfall bins and labels
bins = [0, 5, 10, 20, 30, 50, 75, 100, 150, 250]
labels = ["0-5", "5-10", "10-20", "20-30", "30-50", "50-75", "75-100", "100-150", "150+"]

# Assign bins
master_df["RainfallBin"] = pd.cut(master_df["Rainfall"], bins=bins, labels=labels, right=False)

# Group and average outage count
binned_outage = master_df.groupby("RainfallBin")["OutageCount"].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=binned_outage, x="RainfallBin", y="OutageCount", color="skyblue")
plt.title("Average Outage Count by Rainfall Bin")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Average Outage Count")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "binned_rainfall_vs_outagecount.png"))
plt.close()

print(" Visualizations saved to:", data_dir)

# Spot price (RRP) on outage vs non-outage days
master_df["OutageDay"] = master_df["OutageCount"].fillna(0) > 0
plt.figure(figsize=(8, 6))
sns.boxplot(data=master_df, x="OutageDay", y="RRP_x")
plt.title("Spot Price (RRP_x) on Outage vs Non-Outage Days")
plt.xticks([0, 1], ["No Outage", "Outage"])
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "rrp_on_outage_vs_nonoutage_days.png"))
plt.close()



# --- Monthly Trend Plots by Region ---
master_df["MonthYear"] = master_df["Date"].dt.to_period("M").astype(str)

# Outage Count
monthly_outage = (
    master_df.groupby(["MonthYear", "RegionID"])["OutageCount"]
    .sum().reset_index()
)
plt.figure(figsize=(14, 6))
sns.lineplot(data=monthly_outage, x="MonthYear", y="OutageCount", hue="RegionID", marker="o")
plt.xticks(rotation=45)
plt.xticks(
    ticks=range(0, len(monthly_outage["MonthYear"].unique()), 3),
    labels=monthly_outage["MonthYear"].unique()[::3], rotation=45
)
plt.title("Monthly Total Outage Count by Region")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "monthly_outage_trend_by_region.png"))
plt.close()

# Spot Price (RRP_x)
monthly_rrp = (
    master_df.groupby(["MonthYear", "RegionID"])["RRP_x"]
    .mean().reset_index()
)
plt.figure(figsize=(14, 6))
sns.lineplot(data=monthly_rrp, x="MonthYear", y="RRP_x", hue="RegionID", marker="o")
plt.xticks(rotation=45)
plt.xticks(
    ticks=range(0, len(monthly_rrp["MonthYear"].unique()), 3),
    labels=monthly_rrp["MonthYear"].unique()[::3], rotation=45
)
plt.title("Monthly Average Spot Price (RRP_x) by Region")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "monthly_rrp_trend_by_region.png"))
plt.close()


# Demand
monthly_demand = (
    master_df.groupby(["MonthYear", "RegionID"])["TOTALDEMAND"]
    .mean().reset_index()
)
plt.figure(figsize=(14, 6))
sns.lineplot(data=monthly_demand, x="MonthYear", y="TOTALDEMAND", hue="RegionID", marker="o")
plt.xticks(rotation=45)
plt.xticks(
    ticks=range(0, len(monthly_demand["MonthYear"].unique()), 3),
    labels=monthly_demand["MonthYear"].unique()[::3], rotation=45
)
plt.title("Monthly Average Demand by Region")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "monthly_demand_trend_by_region.png"))
plt.close()


# --- Extreme Weather Flagging ---
from scipy.stats import ttest_ind

# Define heatwave: MaxTemp > 35°C for 3+ consecutive days
master_df = master_df.sort_values(["RegionID", "Date"])
master_df["HeatwaveFlag"] = False

for region in master_df["RegionID"].unique():
    regional_df = master_df[master_df["RegionID"] == region]
    heatwave_mask = (regional_df["MaxTemp"] > 35).astype(int)
    heatwave_streak = heatwave_mask.rolling(3).sum()
    master_df.loc[heatwave_streak.index, "HeatwaveFlag"] = (heatwave_streak >= 3).values

# Define heavy rain: Rainfall > 90th percentile per region
rainfall_thresholds = master_df.groupby("RegionID")["Rainfall"].quantile(0.9)
master_df["HeavyRainFlag"] = master_df.apply(
    lambda row: row["Rainfall"] > rainfall_thresholds[row["RegionID"]], axis=1
)

# Flag extreme weather: either heatwave or heavy rain
master_df["ExtremeWeather"] = master_df["HeatwaveFlag"] | master_df["HeavyRainFlag"]

# Save this version of master_df
master_df.to_csv(os.path.join(data_dir, "master_dataset_with_extreme_flags.csv"), index=False)

# --- Boxplot Comparison ---
plt.figure(figsize=(8, 6))
sns.boxplot(data=master_df, x="ExtremeWeather", y="OutageCount")
plt.xticks([0, 1], ["Non-Extreme", "Extreme"])
plt.title("Outage Count on Extreme vs Non-Extreme Weather Days")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "boxplot_outagecount_extremeweather.png"))
plt.close()

plt.figure(figsize=(8, 6))
sns.boxplot(data=master_df, x="ExtremeWeather", y="TotalDurationMinutes")
plt.xticks([0, 1], ["Non-Extreme", "Extreme"])
plt.title("Outage Duration on Extreme vs Non-Extreme Weather Days")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "boxplot_outageduration_extremeweather.png"))
plt.close()

# --- T-Test Statistical Comparison ---
non_extreme = master_df[master_df["ExtremeWeather"] == False]
extreme = master_df[master_df["ExtremeWeather"] == True]

ttest_outage = ttest_ind(non_extreme["OutageCount"].dropna(), extreme["OutageCount"].dropna(), equal_var=False)
ttest_duration = ttest_ind(non_extreme["TotalDurationMinutes"].dropna(), extreme["TotalDurationMinutes"].dropna(), equal_var=False)

print("\nT-Test Results:")
print("Outage Count (Extreme vs Non-Extreme):", ttest_outage)
print("Total Duration Minutes (Extreme vs Non-Extreme):", ttest_duration)


# --- Lagged Analysis (1 to 3 days before outage) ---
for lag in [1, 2, 3]:
    master_df[f"Rainfall_Lag{lag}"] = master_df.groupby("RegionID")["Rainfall"].shift(lag)
    master_df[f"MaxTemp_Lag{lag}"] = master_df.groupby("RegionID")["MaxTemp"].shift(lag)

# Correlation between lagged values and outages
lagged_corr = master_df[
    [f"Rainfall_Lag{lag}" for lag in [1, 2, 3]] +
    [f"MaxTemp_Lag{lag}" for lag in [1, 2, 3]] +
    ["OutageCount", "TotalDurationMinutes"]
].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(lagged_corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation: Lagged Weather vs Outage Metrics")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "lagged_weather_outage_correlation.png"))
plt.close()

# --- RRP_x Lag Analysis After Outages ---
for lag in [1, 2]:
    master_df[f"Outage_Lag{lag}"] = master_df.groupby("RegionID")["OutageCount"].shift(lag)

plt.figure(figsize=(10, 6))
for region in master_df["RegionID"].dropna().unique():
    regional_data = master_df[master_df["RegionID"] == region]
    for lag in [1, 2]:
        corr_val = regional_data[[f"Outage_Lag{lag}", "RRP_x"]].corr().iloc[0, 1]
        print(f"Correlation between Outage_Lag{lag} and RRP_x in {region}: {corr_val:.3f}")

# --- Segment by Severity (High vs Low Duration Days) ---
threshold = master_df["TotalDurationMinutes"].quantile(0.75)
master_df["HighDurationDay"] = master_df["TotalDurationMinutes"] > threshold

plt.figure(figsize=(8, 6))
sns.boxplot(data=master_df, x="HighDurationDay", y="RRP_x")
plt.xticks([0, 1], ["Low Duration", "High Duration"])
plt.title("Spot Price on High vs Low Outage Duration Days")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "boxplot_rrp_by_outage_duration.png"))
plt.close()

# --- Volatility Plots ---
master_df = master_df.sort_values(["RegionID", "Date"])
master_df["RRP_x_volatility"] = master_df.groupby("RegionID")["RRP_x"].transform(lambda x: x.rolling(window=3, min_periods=2).std())
volatility_df = master_df[["Date", "RegionID", "RRP_x_volatility", "OutageCount"]].copy()
volatility_df["OutageOccurred"] = volatility_df["OutageCount"].fillna(0) > 0

plt.figure(figsize=(8, 6))
sns.boxplot(data=volatility_df.dropna(subset=["RRP_x_volatility"]), x="OutageOccurred", y="RRP_x_volatility")
plt.xticks([0, 1], ["No Outage", "Outage"])
plt.title("Daily Spot Price Volatility on Outage vs Non-Outage Days")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "boxplot_rrp_volatility_outage.png"))
plt.close()

# --- Demand Drop on Outage Days ---
demand_stats = (
    master_df.groupby("OutageDay")["TOTALDEMAND"]
    .describe()[["mean", "std", "count"]]
)
print("\nAverage Demand Comparison (Outage vs Non-Outage Days):")
print(demand_stats)

plt.figure(figsize=(8, 6))
sns.boxplot(data=master_df, x="OutageDay", y="TOTALDEMAND")
plt.xticks([0, 1], ["No Outage", "Outage"])
plt.title("TOTALDEMAND on Outage vs Non-Outage Days")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "boxplot_demand_outage_vs_nonoutage.png"))
plt.close()