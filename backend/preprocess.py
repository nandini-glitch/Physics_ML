import pandas as pd
import numpy as np
import os

def preprocess_physics():
    raw_path = os.path.join('data', 'raw_weather.csv')
    df = pd.read_csv(raw_path, skiprows=10)
    
    # Clean-up
    df = df[pd.to_numeric(df['YEAR'], errors='coerce').notnull()]
    df = df.replace(-999, np.nan).dropna()
    df['Date'] = pd.to_datetime(df[['YEAR', 'MO', 'DY']].astype(str).agg('-'.join, axis=1))

    # --- PHYSICS FEATURE ENGINEERING ---
    # 1. Net Radiation (Rn) proxy
    df['Rn'] = 0.77 * df['ALLSKY_SFC_SW_DWN']

    # 2. Vapor Pressure Deficit (VPD)
    # T2M is Air Temp
    es = 0.6108 * np.exp((17.27 * df['T2M']) / (df['T2M'] + 237.3))
    ea = (df['RH2M'] / 100) * es
    df['VPD'] = es - ea

    # 3. Sensible Heat Proxy (H_proxy)
    df['T_lag1'] = df['T2M'].shift(1)
    df['H_proxy'] = df['WS2M'] * (df['ALLSKY_SFC_SW_DWN'] - df['T_lag1'])
    
    # 4. Energy Imbalance
    df['Energy_Imbalance'] = df['Rn'] - (df['H_proxy'] + (df['RH2M'] * 0.1))

    df = df.dropna()
    processed_path = os.path.join('data', 'processed_weather.csv')
    df.to_csv(processed_path, index=False)
    print(f"🧪 Physics features added. Saved to: {processed_path}")

if __name__ == "__main__":
    preprocess_physics()