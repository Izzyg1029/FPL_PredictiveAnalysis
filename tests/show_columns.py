
"""
TEST CASE: DASH-001 - Power BI Export Column Validation
Author: [Your Name]
Date: 2026-03-10

Requirements Tested:
  - 4.1.1: Final export contains all required columns
  - 4.1.2: Risk score columns present (risk_score, risk_reason)
  - 4.1.3: Device identification columns present (Serial, Device_Type)
  - 4.1.4: Flag columns present (overheat_flag, zero_current_flag, etc.)

Success Criteria:
  - All 96 expected columns present in export
  - risk_score and risk_reason columns exist
  - Device_Type and Serial columns exist
  - All flag columns (overheat_flag, zero_current_flag, etc.) exist
  - No unexpected columns missing
"""
# show_columns.py
import pandas as pd

df = pd.read_csv("../powerbi_exports/FCI_Device_Health_Export.csv")
print("YOUR ACTUAL COLUMN NAMES:")
print("="*50)
for i, col in enumerate(sorted(df.columns)):
    print(f"{i+1:3d}. {col}")
