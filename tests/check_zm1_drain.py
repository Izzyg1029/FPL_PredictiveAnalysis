"""
TEST CASE: UT-001 - ZM1 Battery Drain Rate Validation
Author: [Your Name]
Date: 2026-03-10

Requirements Tested:
  - 3.5.1: Non-linear battery drain calculation for ZM1 devices
  - 3.5.2: Drain rates must be positive and realistic
  - 3.5.3: Drain rate per day must be properly calculated

Success Criteria:
  - All ZM1 devices have battery_drain_rate > 0 (no zeros)
  - Drain rates between 0.1% and 20% per year (realistic range)
  - battery_drain_rate_per_day > 0 for all devices
  - Mean drain rate between 5-15%/year (typical for 10-year battery)
  - Sample devices show reasonable values
"""

import pandas as pd

print('=' * 60)
print('ZM1 DRAIN RATE CHECK')
print('=' * 60)

df = pd.read_csv('data/processed/daily/2025-10-13-FPL-device-export-health.csv')
zm1 = df[df['Device_Type'] == 'ZM1']

print(f'ZM1 devices: {len(zm1)}')

print('\nbattery_drain_rate stats:')
print(f'  min: {zm1["battery_drain_rate"].min()}')
print(f'  max: {zm1["battery_drain_rate"].max()}')
print(f'  mean: {zm1["battery_drain_rate"].mean()}')
print(f'  zeros: {(zm1["battery_drain_rate"] == 0).sum()}')

print('\nbattery_drain_rate_per_day stats:')
print(f'  min: {zm1["battery_drain_rate_per_day"].min()}')
print(f'  max: {zm1["battery_drain_rate_per_day"].max()}')
print(f'  mean: {zm1["battery_drain_rate_per_day"].mean()}')
print(f'  zeros: {(zm1["battery_drain_rate_per_day"] == 0).sum()}')

print('\nSample ZM1 devices:')
sample = zm1[['Serial', 'battery_drain_rate', 'battery_drain_rate_per_day', 'pct_life_used']].head(5)
print(sample.to_string())

print('\n' + '=' * 60)