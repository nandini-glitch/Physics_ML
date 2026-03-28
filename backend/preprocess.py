import pandas as pd
import numpy as np
import os

def preprocess_physics():
    raw_path = os.path.join('data', 'raw_weather.csv')
    
    # 1. FIND THE HEADER
    skip_count = 0
    with open(raw_path, 'r') as f:
        for i, line in enumerate(f):
            if 'YEAR' in line:
                skip_count = i
                break
    
    print(f"🔍 Found data header at line {skip_count+1}. Loading...")
    df = pd.read_csv(raw_path, skiprows=skip_count)
    
    # Clean non-numeric metadata rows
    df = df[pd.to_numeric(df['YEAR'], errors='coerce').notnull()]
    df = df.replace("-999", np.nan).replace(-999, np.nan).dropna()

    # 2. CONVERT YEAR + DOY TO DATE
    # Since NASA gave us 'DOY', we use this specific logic:
    df['Date'] = pd.to_datetime(df['YEAR'].astype(int).astype(str) + 
                                df['DOY'].astype(int).astype(str), format='%Y%j')
    
    print(f"✅ Successfully converted DOY to Dates. Start: {df['Date'].min()} End: {df['Date'].max()}")

    # 3. PHYSICS FEATURE ENGINEERING
    # Net Radiation (Rn) proxy
    df['Rn'] = 0.77 * df['ALLSKY_SFC_SW_DWN']

    # Vapor Pressure Deficit (VPD)
    # Formula: Saturation VP (es) - Actual VP (ea)
    es = 0.6108 * np.exp((17.27 * df['T2M']) / (df['T2M'] + 237.3))
    ea = (df['RH2M'] / 100) * es
    df['VPD'] = es - ea

    # Sensible Heat Proxy (H_proxy)
    df = df.sort_values('Date')
    df['T_lag1'] = df['T2M'].shift(1)
    df = df.dropna() # Drop first row (no lag)
    
    # H_proxy ≈ Wind * (Solar - Yesterday's Temp)
    df['H_proxy'] = df['WS2M'] * (df['ALLSKY_SFC_SW_DWN'] - df['T_lag1'])
    
    # Energy Imbalance (Simplified SEB)
    df['Energy_Imbalance'] = df['Rn'] - (df['H_proxy'] + (df['RH2M'] * 0.1))

    # 4. SAVE
    os.makedirs('data', exist_ok=True)
    processed_path = os.path.join('data', 'processed_weather.csv')
    df.to_csv(processed_path, index=False)
    print(f"🚀 Physics features added! File ready at: {processed_path}")

if __name__ == "__main__":
    preprocess_physics()