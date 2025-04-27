import pandas as pd

# === Step 1: Read the Master Scribe ===
df_yearly = pd.read_csv("H1M.csv")

# === Step 2: Clean the column names (just in case of trailing spaces or encoding issues) ===
df_yearly.columns = df_yearly.columns.str.strip()

# === Step 3: Convert coordinate columns to numeric (ensure clean values) ===
df_yearly["Lat"] = pd.to_numeric(df_yearly["Lat"], errors="coerce")
df_yearly["Long"] = pd.to_numeric(df_yearly["Long"], errors="coerce")
df_yearly["Delay"] = pd.to_numeric(df_yearly["Delay"], errors="coerce")

# === Step 4: Group by Station Info and Calculate Average Delay ===
df_station_summary = df_yearly.groupby(
    ["Station_Name", "Station_Code", "Lat", "Long"],
    as_index=False
)["Delay"].mean()

# === Step 5: Rename for clarity ===
df_station_summary.rename(columns={"Delay": "Average_Delay"}, inplace=True)

# === Step 6: Save the newly forged scribe ===
df_station_summary.to_csv("Stationwise_H1M.csv", index=False)

print("🪄 The new scribe 'Stationwise_Delay_Summary_Yearly.csv' has been successfully crafted.")
