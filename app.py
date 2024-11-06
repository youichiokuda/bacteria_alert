from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import pandas as pd
from scipy.stats import zscore
import os
import openai
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import io

# .envファイルの読み込み
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# OpenAI APIキーの取得
openai.api_key = os.getenv("OPENAI_API_KEY")

def detect_outbreak_zscore(df, z_threshold=2):
    # 日付列の型変換
    df['date'] = pd.to_datetime(df['date'])
    df['month_year'] = df['date'].dt.to_period('M')
    
    # "I" を "S" と同じ扱いにして耐性率を計算
    resistance_rate = df.assign(
        resistance=df['resistance'].replace('I', 'S')  # "I" を "S" に置換
    ).groupby(['bacteria', 'antibiotic', 'month_year'])['resistance'].apply(lambda x: (x == 'R').mean()).reset_index(name='resistance_rate')
    
    # zスコアの計算
    resistance_rate['z_score'] = resistance_rate.groupby(['bacteria', 'antibiotic'])['resistance_rate'].transform(lambda x: zscore(x, nan_policy='omit'))
    
    # zスコアが閾値を超えるものをアウトブレイクアラートとして検出
    outbreak_alerts = resistance_rate[resistance_rate['z_score'] > z_threshold].to_dict(orient='records')
    return outbreak_alerts

def generate_comment(alert):
    prompt = (
        f"以下の耐性データに基づいてコメントを作成してください：\n\n"
        f"細菌種: {alert['bacteria']}\n"
        f"抗生物質: {alert['antibiotic']}\n"
        f"月: {alert['month_year']}\n"
        f"耐性率: {alert['resistance_rate']}\n"
        f"Zスコア: {alert['z_score']}\n\n"
        "この結果を解釈し、簡単なコメントを作成してください。"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.RateLimitError:
        return "クォータ制限に達しました。コメントの生成ができませんでした。"

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
        
        for alert in outbreak_alerts:
            alert['comment'] = generate_comment(alert)
        
        return render_template('results.html', alerts=outbreak_alerts)
    
    return redirect(request.url)

@app.route('/details/<bacteria>/<antibiotic>/<month>')
def show_outbreak_details(bacteria, antibiotic, month):
    month_period = pd.Period(month, 'M')
    
    # アップロードされたデータを読み込む
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], os.listdir(app.config['UPLOAD_FOLDER'])[0])
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df['month_year'] = df['date'].dt.to_period('M')
    
    # 該当する細菌と薬剤のデータをフィルタリングし、"I" を "S" に置換
    df_filtered = df[(df['bacteria'] == bacteria) & (df['antibiotic'] == antibiotic)]
    df_filtered['resistance'] = df_filtered['resistance'].replace('I', 'S')  # "I" を "S" に置換
    
    # 月別の検査数、陽性数、陽性率を計算
    monthly_summary = df_filtered.groupby('month_year').agg(
        test_count=('resistance', 'size'),
        positive_count=('resistance', lambda x: (x == 'R').sum()),
        positive_rate=('resistance', lambda x: (x == 'R').mean())
    ).reset_index()
    
    # グラフの作成
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_title(f'Outbreak Details for {bacteria} with {antibiotic} - {month}')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Count')
    
    ax1.bar(monthly_summary['month_year'].astype(str), monthly_summary['test_count'], color='lightblue', label='Test Count')
    ax1.bar(monthly_summary['month_year'].astype(str), monthly_summary['positive_count'], color='salmon', label='Positive Count', alpha=0.7)
    
    ax2 = ax1.twinx()
    ax2.plot(monthly_summary['month_year'].astype(str), monthly_summary['positive_rate'], color='green', marker='o', label='Positive Rate')
    ax2.set_ylabel('Positive Rate')
    
    fig.tight_layout()
    fig.legend(loc="upper left")
    
    # グラフをバイトストリームとして保存
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)
    
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
