import pandas as pd

# --------------------------------------------------------
# Load data
# --------------------------------------------------------
DEVICE_FILE = r"C:\Users\megau\Downloads\9-13.xlsx"     # make sure to change the file location
OUTPUT_FILE = r"C:\Users\megau\Downloads\9-13-new.xlsx" # this one too

df_raw = pd.read_excel(DEVICE_FILE)
print(f"📥 Total devices loaded: {len(df_raw)}")

# --------------------------------------------------------
# Core sensor columns ONLY (battery ignored)
# --------------------------------------------------------
CRITICAL_COLS = [
    "Device_Type",
    "Last_Heard",
    "LineCurrent",
    "LineTemperatrue",
]

# --------------------------------------------------------
# Filter out devices with null core sensor data
# --------------------------------------------------------
broken_mask = df_raw[CRITICAL_COLS].isnull().any(axis=1)

df_clean = df_raw[~broken_mask].copy()

# --------------------------------------------------------
# Save clean data to NEW file
# --------------------------------------------------------
df_clean.to_excel(OUTPUT_FILE, index=False)

print(f"✅ Clean devices saved: {len(df_clean)}")

print(f"📁 New file created: {OUTPUT_FILE}")
