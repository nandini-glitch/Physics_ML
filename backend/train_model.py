import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

def run_experiments():
    df = pd.read_csv('data/processed_weather.csv')
    
    # Define Feature Sets
    raw_feats = ['ALLSKY_SFC_SW_DWN', 'RH2M', 'WS2M', 'PS', 'T_lag1']
    phys_feats = raw_feats + ['Rn', 'VPD', 'H_proxy', 'Energy_Imbalance']
    target = 'T2M'

    # Time-Series Split
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    # Model Dictionary: (Name, Features, ModelType)
    experiments = [
        ("RF Pure", raw_feats, RandomForestRegressor()),
        ("RF Physics", phys_feats, RandomForestRegressor()),
        ("GBM Pure", raw_feats, GradientBoostingRegressor()),
        ("GBM Physics", phys_feats, GradientBoostingRegressor())
    ]

    results = []
    for name, feats, model in experiments:
        model.fit(train[feats], train[target])
        preds = model.predict(test[feats])
        rmse = np.sqrt(mean_squared_error(test[target], preds))
        r2 = r2_score(test[target], preds)
        results.append({"Model": name, "RMSE": rmse, "R2": r2})

    # Output Results
    res_df = pd.DataFrame(results)
    print("\n📊 FINAL COMPARISON:")
    print(res_df)

    # Plot Feature Importance for Physics Model
    best_model = experiments[1][2] # RF Physics
    feat_importances = pd.Series(best_model.feature_importances_, index=phys_feats)
    feat_importances.nlargest(10).plot(kind='barh', color='teal')
    plt.title("Importance of Physical Proxies")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiments()