from flask import Flask, render_template, jsonify
import pandas as pd
import os
from scipy.stats import zscore

app = Flask(__name__)

# データ読み込みと処理
data_path = os.path.join("data", "sensitivity_results_sample.csv")
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])
df['month_year'] = df['date'].dt.to_period('M')

resistance_rate = df.groupby(['bacteria', 'antibiotic', 'month_year'])['resistance'].apply(lambda x: (x == 'R').mean()).reset_index(name='resistance_rate')

def detect_outbreak_zscore(resistance_df, z_threshold=2):
    outbreak_alerts = []
    resistance_df['z_score'] = resistance_df.groupby(['bacteria', 'antibiotic'])['resistance_rate'].transform(lambda x: zscore(x, nan_policy='omit'))
    
    for _, row in resistance_df.iterrows():
        if row['z_score'] > z_threshold:
            outbreak_alerts.append({
                'bacteria': row['bacteria'],
                'antibiotic': row['antibiotic'],
                'month_year': str(row['month_year']),
                'resistance_rate': row['resistance_rate'],
                'z_score': row['z_score']
            })
    return outbreak_alerts

outbreak_alerts = detect_outbreak_zscore(resistance_rate)

@app.route('/')
def dashboard():
    alerts = df.to_dict(orient="records")
    return render_template("dashboard.html", alerts=alerts, outbreak_alerts=outbreak_alerts)

@app.route('/api/data')
def api_data():
    alerts = df.to_dict(orient="records")
    return jsonify({"alerts": alerts, "outbreak_alerts": outbreak_alerts})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
