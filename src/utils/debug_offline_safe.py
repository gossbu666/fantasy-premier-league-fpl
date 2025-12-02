#!/usr/bin/env python3
import pandas as pd
import joblib

POS = "FWD"  # ลองเปลี่ยนเป็น MID / DEF / GK ได้

df = pd.read_csv(f"data/processed/{POS}_features_enhanced_safe.csv")
model = joblib.load(f"models/{POS}_202425_safe_final.pkl")
with open(f"models/{POS}_202425_features.txt") as f:
    feature_cols = [c for c in f.read().strip().split("\n") if c != "target"]

print(f"{POS}: rows={len(df)}, features={len(feature_cols)}")

# test บน 2024-25 GW12+ (เหมือน validation)
mask = (df["season"] == "2024-25") & (df["round"] >= 12)
test = df[mask].copy()
X = test[feature_cols].fillna(0).to_numpy()
y = test["target"].to_numpy()

y_pred = model.predict(X)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)

print(f"{POS} 2024-25 GW12+  R2={r2:.3f}, RMSE={rmse:.3f}, n={len(test)}")
