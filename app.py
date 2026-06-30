"""
Gold Price Prediction — Flask Application
==========================================
9-feature ML pipeline: SPX, USO, SLV, EUR/USD,
SLV_USO_Ratio, GLD_lag1, GLD_MA7, SPX_Return, SLV_Return
"""

from flask import Flask, render_template, request, jsonify
import joblib, numpy as np, json, os

app = Flask(__name__)

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'gold_model.pkl')
DATA_PATH  = os.path.join(BASE_DIR, 'static', 'data', 'app_data.json')

art       = joblib.load(MODEL_PATH)
rf_model  = art['rf']
lasso     = art['lasso']

scaler    = art['scaler']
FEATURES  = art['features']
# ['SPX','USO','SLV','EUR/USD','SLV_USO_Ratio','GLD_lag1','GLD_MA7','SPX_Return','SLV_Return']

with open(DATA_PATH) as f:
    APP_DATA = json.load(f)


@app.route('/')
def index():
    return render_template('index.html',
                           stats=APP_DATA['stats'],
                           results=APP_DATA['results'],
                           best=APP_DATA['best_model'])

@app.route('/analysis')
def analysis():
    return render_template('analysis.html',
                           stats=APP_DATA['stats'],
                           ts=APP_DATA['ts'])

@app.route('/models')
def models():
    return render_template('models.html',
                           results=APP_DATA['results'],
                           best=APP_DATA['best_model'])

@app.route('/insights')
def insights():
    return render_template('insights.html',
                           ms=APP_DATA['market_summary'],
                           ts_full=APP_DATA['ts_full'],
                           yearly=APP_DATA['yearly'],
                           monthly=APP_DATA['monthly'],
                           returns=APP_DATA['returns'],
                           volatility=APP_DATA['volatility'],
                           corr_rolling=APP_DATA['corr_rolling'])

@app.route('/compare')
def compare():
    return render_template('compare.html',
                           compare=APP_DATA['compare'],
                           results=APP_DATA['results'],
                           best=APP_DATA['best_model'])


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    all_preds  = None
    error      = None
    form_data  = {}

    if request.method == 'POST':
        try:
            spx       = float(request.form['SPX'])
            uso       = float(request.form['USO'])
            slv       = float(request.form['SLV'])
            eurusd    = float(request.form['EURUSD'])
            gld_lag1  = float(request.form['GLD_lag1'])
            gld_ma7   = float(request.form['GLD_MA7'])
            spx_ret   = float(request.form['SPX_Return'])
            slv_ret   = float(request.form['SLV_Return'])

            slv_uso_ratio = slv / uso if uso != 0 else 0

            form_data = {
                'SPX': spx, 'USO': uso, 'SLV': slv, 'EURUSD': eurusd,
                'GLD_lag1': gld_lag1, 'GLD_MA7': gld_ma7,
                'SPX_Return': spx_ret, 'SLV_Return': slv_ret,
                'SLV_USO_Ratio': round(slv_uso_ratio, 4)
            }

            X = np.array([[spx, uso, slv, eurusd, slv_uso_ratio,
                           gld_lag1, gld_ma7, spx_ret, slv_ret]])
            X_sc = scaler.transform(X)

            pred_rf    = round(float(rf_model.predict(X)[0]), 4)
            pred_lasso = round(float(lasso.predict(X_sc)[0]), 4)
           

            prediction = pred_rf
            all_preds  = {
                'Random Forest': pred_rf,
                'Lasso':         pred_lasso,
                
                'Ensemble':      round((pred_rf + pred_lasso + pred_xgb) / 3, 4)
            }

        except (ValueError, KeyError) as e:
            error = f"Invalid input: {e}"

    return render_template('predict.html',
                           prediction=prediction,
                           all_preds=all_preds,
                           error=error,
                           form_data=form_data)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        b = request.get_json()
        spx, uso, slv  = float(b['SPX']), float(b['USO']), float(b['SLV'])
        eurusd         = float(b.get('EURUSD', 1.2))
        gld_lag1       = float(b['GLD_lag1'])
        gld_ma7        = float(b['GLD_MA7'])
        spx_ret        = float(b.get('SPX_Return', 0))
        slv_ret        = float(b.get('SLV_Return', 0))
        slv_uso_ratio  = slv / uso if uso else 0

        X    = np.array([[spx, uso, slv, eurusd, slv_uso_ratio, gld_lag1, gld_ma7, spx_ret, slv_ret]])
        X_sc = scaler.transform(X)
        pred_rf    = round(float(rf_model.predict(X)[0]), 4)
        pred_lasso = round(float(lasso.predict(X_sc)[0]), 4)
        pred_xgb   = round(float(xgb_model.predict(X)[0]), 4)

        return jsonify({'status': 'success', 'predictions': {
            'Random Forest': pred_rf, 'Lasso': pred_lasso,
            
            'Ensemble': round((pred_rf+pred_lasso+pred_xgb)/3, 4)
        }})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
