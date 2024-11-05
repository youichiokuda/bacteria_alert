
from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
from scipy.stats import zscore
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def detect_outbreak_zscore(df, z_threshold=2):
    df['month_year'] = df['date'].dt.to_period('M')
    resistance_rate = df.groupby(['bacteria', 'antibiotic', 'month_year'])['resistance'].apply(lambda x: (x == 'R').mean()).reset_index(name='resistance_rate')
    resistance_rate['z_score'] = resistance_rate.groupby(['bacteria', 'antibiotic'])['resistance_rate'].transform(lambda x: zscore(x, nan_policy='omit'))
    
    outbreak_alerts = resistance_rate[resistance_rate['z_score'] > z_threshold].to_dict(orient='records')
    return outbreak_alerts

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        
        outbreak_alerts = detect_outbreak_zscore(df)
        
        return render_template('results.html', alerts=outbreak_alerts)
    
    return redirect(request.url)

@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        
        outbreak_alerts = detect_outbreak_zscore(df)
        
        return jsonify({"outbreak_alerts": outbreak_alerts})
    
    return jsonify({"error": "Invalid file format. Only CSV files are allowed."}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
