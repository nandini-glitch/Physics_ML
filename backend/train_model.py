import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Set Plotting Style for professional aesthetics
plt.style.use('seaborn-v0_8-whitegrid')

def run_advanced_analysis():
    # 1. LOAD DATA
    if not os.path.exists('data/processed_weather.csv'):
        print("❌ Error: processed_weather.csv not found. Run preprocess.py first.")
        return

    df = pd.read_csv('data/processed_weather.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Define Feature Sets
    raw_feats = ['ALLSKY_SFC_SW_DWN', 'RH2M', 'WS2M', 'PS']
    phys_feats = raw_feats + ['Rn', 'VPD', 'H_proxy', 'Energy_Imbalance']
    target = 'T2M'

    # Split Data (80% Train, 20% Test)
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    # 2. TRAIN MODELS
    # We use 4 versions to prove the value of Physics-Informed ML
    models = {
        "RF Pure": (raw_feats, RandomForestRegressor(n_estimators=100, random_state=42)),
        "RF Physics": (phys_feats, RandomForestRegressor(n_estimators=100, random_state=42)),
        "GBM Pure": (raw_feats, GradientBoostingRegressor(random_state=42)),
        "GBM Physics": (phys_feats, GradientBoostingRegressor(random_state=42))
    }

    results = []
    predictions = {}

    print("🚀 Training models and calculating metrics...")
    for name, (feats, model) in models.items():
        model.fit(train[feats], train[target])
        preds = model.predict(test[feats])
        rmse = np.sqrt(mean_squared_error(test[target], preds))
        r2 = r2_score(test[target], preds)
        
        results.append({"Model": name, "RMSE": rmse, "R2": r2})
        predictions[name] = preds
        
    res_df = pd.DataFrame(results)
    print("\n📊 FINAL COMPARISON:\n", res_df)

    # IDENTIFY BEST MODEL FOR DETAILED PLOTS
    best_model_name = "GBM Physics"
    best_preds = predictions[best_model_name]
    pure_preds = predictions["GBM Pure"]
    actual = test[target].values

    print("\n🖼️ Generating 300 DPI high-resolution plots...")

    # ---------------------------------------------------------
    # PLOT 1: Actual vs Predicted (Ideal 45-degree check)
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, best_preds, alpha=0.5, color='teal', edgecolors='white')
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2, label="Ideal (Zero Error)")
    plt.title(f"Actual vs Predicted Temperature ({best_model_name})", fontsize=14)
    plt.xlabel("Actual Temperature (°C)")
    plt.ylabel("Predicted Temperature (°C)")
    plt.legend()
    plt.savefig('plot_1_actual_vs_pred.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # PLOT 2: Time Series Comparison (Tracking Trends)
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(test['Date'], actual, label='Actual', color='black', alpha=0.6, lw=1.5)
    plt.plot(test['Date'], best_preds, label=f'Predicted ({best_model_name})', color='red', linestyle='--', alpha=0.8)
    plt.title("Temperature Forecast vs Reality Over Time", fontsize=14)
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.savefig('plot_2_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # PLOT 3: Model Comparison Bar Graph (The "Win" Chart)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='RMSE', data=res_df, palette='viridis', hue='Model', legend=False)
    plt.title("Model Performance: Impact of Physics on RMSE", fontsize=14)
    plt.ylabel("RMSE (Lower is Better)")
    plt.savefig('plot_3_rmse_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # PLOT 4: Feature Importance (Scientific Validation)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    best_model_obj = models[best_model_name][1]
    importances = pd.Series(best_model_obj.feature_importances_, index=phys_feats).sort_values()
    importances.plot(kind='barh', color='skyblue')
    plt.title(f"Top Drivers of Temperature Prediction ({best_model_name})", fontsize=14)
    plt.xlabel("Relative Importance Score")
    plt.savefig('plot_4_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # PLOT 5: Residual Plot (Evaluating Bias/Stability)
    # ---------------------------------------------------------
    residuals = actual - best_preds
    plt.figure(figsize=(10, 6))
    plt.scatter(best_preds, residuals, alpha=0.5, color='purple', edgecolors='white')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residual Analysis: Are Errors Random?", fontsize=14)
    plt.xlabel("Predicted Temperature (°C)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.savefig('plot_5_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # PLOT 6: Physics vs Pure Error Comparison (Daily Error)
    # ---------------------------------------------------------
    pure_error = np.abs(actual - pure_preds)
    phys_error = np.abs(actual - best_preds)
    plt.figure(figsize=(12, 5))
    plt.fill_between(test['Date'], pure_error, color='grey', alpha=0.2, label='Pure Model Error Area')
    plt.plot(test['Date'], phys_error, label='Physics Model Error', color='green', lw=1)
    plt.title("Daily Prediction Error: Physics-Informed vs Pure Data", fontsize=14)
    plt.ylabel("Absolute Error (°C)")
    plt.legend()
    plt.savefig('plot_6_error_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # PLOT 7: Extreme Temperature Analysis (Hottest Days)
    # ---------------------------------------------------------
    threshold = np.percentile(actual, 85)
    extreme_mask = actual >= threshold
    ext_pure_rmse = np.sqrt(mean_squared_error(actual[extreme_mask], pure_preds[extreme_mask]))
    ext_phys_rmse = np.sqrt(mean_squared_error(actual[extreme_mask], best_preds[extreme_mask]))
    
    plt.figure(figsize=(7, 6))
    plt.bar(['Pure (No Physics)', 'Physics-Informed'], [ext_pure_rmse, ext_phys_rmse], color=['darkgrey', '#e74c3c'])
    plt.title(f"Performance During Extreme Heat (Days > {threshold:.1f}°C)", fontsize=13)
    plt.ylabel("RMSE (Error Magnitude)")
    plt.savefig('plot_7_extreme_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # PLOT 8: Seasonal Performance (Summer vs Winter)
    # ---------------------------------------------------------
    test_copy = test.copy()
    test_copy['Error_Sq'] = (actual - best_preds)**2
    def get_season(month):
        if month in [3, 4, 5, 6]: return 'Summer'
        if month in [7, 8, 9]: return 'Monsoon'
        return 'Winter'
    
    test_copy['Season'] = test_copy['Date'].dt.month.apply(get_season)
    seasonal_rmse = test_copy.groupby('Season')['Error_Sq'].apply(lambda x: np.sqrt(np.mean(x)))
    
    plt.figure(figsize=(8, 6))
    seasonal_rmse.plot(kind='bar', color=['#f39c12', '#3498db', '#2ecc71'])
    plt.title("Model Error (RMSE) by Season", fontsize=14)
    plt.ylabel("RMSE (°C)")
    plt.xticks(rotation=0)
    plt.savefig('plot_8_seasonal.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # PLOT 9: Error Distribution (Narrowness Check)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.kdeplot(actual - pure_preds, label='Pure Model Error', fill=True, color='grey', alpha=0.3)
    sns.kdeplot(actual - best_preds, label='Physics Model Error', fill=True, color='green', alpha=0.5)
    plt.title("Error Distribution: Reliability of Predictions", fontsize=14)
    plt.xlabel("Prediction Error (°C)")
    plt.ylabel("Density (Frequency)")
    plt.legend()
    plt.savefig('plot_9_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Success! All 9 high-res plots saved in your backend folder.")

if __name__ == "__main__":
    run_advanced_analysis()