"""
train_model.py
==============
Run this script ONCE to train all models and save artefacts.
Usage:  python train_model.py
Output: models/gold_model.pkl  and  static/data/app_data.json
"""

import pandas as pd
import numpy as np
import json, os, warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import joblib

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = 'data/gold_price_data.csv'
MODEL_OUT   = 'models/gold_model.pkl'
STATS_OUT   = 'static/data/app_data.json'
FEATURES    = ['SPX', 'USO', 'SLV', 'EUR/USD']
TARGET      = 'GLD'
TEST_SIZE   = 0.2
RANDOM_SEED = 42

os.makedirs('models', exist_ok=True)
os.makedirs('static/data', exist_ok=True)

# ── Custom JSON encoder (handles numpy types) ─────────────────────────────────
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# ── EDA statistics (passed to templates) ─────────────────────────────────────
print("Computing EDA statistics …")
cols = FEATURES + [TARGET]
stats = {
    'shape': list(df.shape),
    'missing': {k: int(v) for k,v in df[cols].isnull().sum().items()},
    'describe': {col: {k: float(v) for k,v in vals.items()}
                 for col, vals in df[cols].describe().round(3).to_dict().items()},
    'skewness': {k: float(v) for k,v in df[cols].skew().items()},
    'corr':     {col: {k: float(v) for k,v in vals.items()}
                 for col, vals in df[cols].corr().round(4).to_dict().items()},
    'dtypes':   {c: str(df[c].dtype) for c in cols},
    'sample':   df[cols].head(10).round(4).to_dict(orient='records'),
    'hist_data': {c: [float(v) for v in df[c].dropna()] for c in cols},
    'boxplot_data': {c: {
        'q1':  float(df[c].quantile(0.25)),
        'median': float(df[c].median()),
        'q3':  float(df[c].quantile(0.75)),
        'min': float(df[c].min()),
        'max': float(df[c].max()),
        'mean': float(df[c].mean()),
        'std':  float(df[c].std()),
    } for c in cols}
}

# ── Pre-processing ────────────────────────────────────────────────────────────
print("Pre-processing …")
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Model training ────────────────────────────────────────────────────────────
results = {}

# 1. Lasso Regression
print("Training Lasso …")
lasso = Lasso(alpha=0.01, max_iter=10_000)
lasso.fit(X_train_sc, y_train)
y_pred_l = lasso.predict(X_test_sc)
results['Lasso'] = {
    'r2':   round(float(r2_score(y_test, y_pred_l)), 4),
    'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_pred_l))), 4),
    'mae':  round(float(mean_absolute_error(y_test, y_pred_l)), 4),
    'coef': {f: round(float(c), 4) for f, c in zip(FEATURES, lasso.coef_)}
}

# 2. Random Forest + GridSearchCV
print("Training Random Forest (GridSearchCV) …")
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [None, 10]}
gs_rf = GridSearchCV(RandomForestRegressor(random_state=RANDOM_SEED),
                     param_grid_rf, cv=3, scoring='r2', n_jobs=-1)
gs_rf.fit(X_train, y_train)
best_rf = gs_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
results['Random Forest'] = {
    'r2':   round(float(r2_score(y_test, y_pred_rf)), 4),
    'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_pred_rf))), 4),
    'mae':  round(float(mean_absolute_error(y_test, y_pred_rf)), 4),
    'best_params': {k: (int(v) if v is not None else None)
                    for k, v in gs_rf.best_params_.items()},
    'best_score': round(float(gs_rf.best_score_), 4),
    'feature_importance': {f: round(float(v), 4)
                           for f, v in zip(FEATURES, best_rf.feature_importances_)}
}

# 3. XGBoost
print("Training XGBoost …")
xgb = XGBRegressor(random_state=RANDOM_SEED, n_estimators=200,
                   learning_rate=0.05, max_depth=5, verbosity=0)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
results['XGBoost'] = {
    'r2':   round(float(r2_score(y_test, y_pred_xgb)), 4),
    'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_pred_xgb))), 4),
    'mae':  round(float(mean_absolute_error(y_test, y_pred_xgb)), 4),
    'feature_importance': {f: round(float(v), 4)
                           for f, v in zip(FEATURES, xgb.feature_importances_)}
}

# ── Save models ───────────────────────────────────────────────────────────────
print("Saving artefacts …")
joblib.dump({'rf': best_rf, 'lasso': lasso, 'xgb': xgb,
             'scaler': scaler, 'features': FEATURES}, MODEL_OUT)

# Time-series sample for charts
idx = np.linspace(0, len(df)-1, 300, dtype=int)
ts_data = {
    'dates': df['Date'].iloc[idx].tolist(),
    'gld':   [float(v) for v in df['GLD'].iloc[idx]]
}

all_data = {'stats': stats, 'results': results,
            'best_model': 'Random Forest', 'ts': ts_data}
with open(STATS_OUT, 'w') as f:
    json.dump(all_data, f, cls=NpEncoder)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Model Performance Summary ===")
for name, m in results.items():
    print(f"  {name:<20} R²={m['r2']}  RMSE={m['rmse']}  MAE={m['mae']}")
print(f"\n✓ Best model: Random Forest  (R²={results['Random Forest']['r2']})")
print(f"✓ Saved to  : {MODEL_OUT}")
print(f"✓ Stats to  : {STATS_OUT}")
