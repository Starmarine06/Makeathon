import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import requests
import warnings
import pickle
import seaborn as sns
warnings.filterwarnings('ignore')

# ==========================
# CONFIG
# ==========================
SYMBOL = "AAPL"
TRAIN_START = "2020-01-01"
TRAIN_END = "2025-10-12"
TEST_START = "2018-01-01"
TEST_END = "2019-12-30"
MAX_ARTICLES = 250

analyzer = SentimentIntensityAnalyzer()

# ==========================
# Helper functions
# ==========================
def get_company_name(symbol):
    us_stocks = {
        "AAPL": "Apple",
        "GOOGL": "Google",
        "MSFT": "Microsoft",
        "TSLA": "Tesla",
        "AMZN": "Amazon",
        "META": "Meta",
        "NVDA": "NVIDIA"
    }
    
    return us_stocks.get(symbol) or symbol

def fetch_gdelt_articles(symbol, start_dt, end_dt):
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    company_name = get_company_name(symbol)
    params = {
        "query": f"{company_name} stock",
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(MAX_ARTICLES),
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
    }
    
    try:
        r = requests.get(base_url, params=params, timeout=30)
        articles = r.json().get("articles", [])
        return [a.get("title", "") for a in articles]
    except:
        return []

def compute_sentiment(df, articles):
    daily_sent = {}
    for date in df.index:
        relevant = []
        for offset in range(-3, 4):
            d = date + pd.Timedelta(days=offset)
            relevant.extend([t for t in articles if t])
        
        if relevant:
            scores = [analyzer.polarity_scores(t)['compound'] for t in relevant[:20]]
            daily_sent[date] = float(np.mean(scores))
        else:
            daily_sent[date] = 0.0
    
    df['sentiment'] = pd.Series(daily_sent)
    df['sentiment_ma3'] = df['sentiment'].rolling(3).mean().fillna(0)
    df['sentiment_ma7'] = df['sentiment'].rolling(7).mean().fillna(0)
    df['sentiment_slope'] = df['sentiment'] - df['sentiment'].shift(3).fillna(0)
    return df

def prepare_features_advanced(df, symbol=None, market='US'):
    """Prepare advanced features matching web app"""
    df["returns"] = df["Close"].pct_change().fillna(0)
    df["volatility"] = df["returns"].rolling(10).std().fillna(0)
    df["momentum"] = df["Close"] / df["Close"].shift(10) - 1

    df["sma10"] = df["Close"].rolling(10).mean()
    df["sma20"] = df["Close"].rolling(20).mean()
    df["sma50"] = df["Close"].rolling(50).mean()
    df["ema10"] = df["Close"].ewm(span=10).mean()
    df["ema20"] = df["Close"].ewm(span=20).mean()

    df["bb_middle"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + (2 * bb_std)
    df["bb_lower"] = df["bb_middle"] - (2 * bb_std)
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_diff"] = df["macd"] - df["macd_signal"]

    df["volume_sma"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_sma"]

    df["high_low_pct"] = (df["High"] - df["Low"]) / df["Close"]
    df["close_open_pct"] = (df["Close"] - df["Open"]) / df["Open"]

    df["sentiment"] = 0.0
    df["sentiment_ma3"] = 0.0
    df["sentiment_ma7"] = 0.0
    df["sentiment_slope"] = 0.0

    # Add profitability ratios if symbol is provided
    if symbol is not None:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            # Profitability ratios
            profitability_map = {
                'roe': 'returnOnEquity',
                'roa': 'returnOnAssets',
                'profit_margin': 'profitMargins',
                'operating_margin': 'operatingMargins',
                'gross_margin': 'grossMargins'
            }
            for feature, info_key in profitability_map.items():
                if info_key in info and info[info_key] is not None:
                    df[feature] = info[info_key]
                # If not available, leave as NaN (will be filled with 0 later)
        except Exception as e:
            print(f"⚠️ Could not fetch profitability ratios for {symbol}: {e}")

    df.fillna(0, inplace=True)
    return df

def plot_test_results(df_test, y_test, y_test_pred, y_test_proba, symbol):
    """Create comprehensive visualization of test results"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # 1. Price with Actual vs Predicted Direction
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_test.index, df_test['Close'], 'b-', linewidth=2, label='Actual Price', alpha=0.7)
    
    # Mark correct predictions
    correct_up = (y_test == 1) & (y_test_pred == 1)
    correct_down = (y_test == 0) & (y_test_pred == 0)
    wrong_up = (y_test == 1) & (y_test_pred == 0)
    wrong_down = (y_test == 0) & (y_test_pred == 1)
    
    ax1.scatter(df_test.index[correct_up], df_test['Close'][correct_up], 
               color='green', marker='^', s=100, label='Correct UP prediction', alpha=0.6, zorder=5)
    ax1.scatter(df_test.index[correct_down], df_test['Close'][correct_down], 
               color='darkgreen', marker='v', s=100, label='Correct DOWN prediction', alpha=0.6, zorder=5)
    ax1.scatter(df_test.index[wrong_up], df_test['Close'][wrong_up], 
               color='red', marker='x', s=100, label='Wrong prediction (should be UP)', alpha=0.8, zorder=5)
    ax1.scatter(df_test.index[wrong_down], df_test['Close'][wrong_down], 
               color='orange', marker='x', s=100, label='Wrong prediction (should be DOWN)', alpha=0.8, zorder=5)
    
    ax1.set_title(f'{symbol} - Test Phase: Actual Price vs Predictions', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Prediction Probability
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(df_test.index, y_test_proba * 100, 'purple', linewidth=2, label='Predicted Probability')
    ax2.axhline(50, color='gray', linestyle='--', linewidth=1, label='Decision Threshold (50%)')
    ax2.axhline(55, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Strong BUY (55%)')
    ax2.axhline(45, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Strong SELL (45%)')
    ax2.fill_between(df_test.index, 50, y_test_proba * 100, 
                     where=(y_test_proba * 100 >= 50), color='green', alpha=0.2)
    ax2.fill_between(df_test.index, 50, y_test_proba * 100, 
                     where=(y_test_proba * 100 < 50), color='red', alpha=0.2)
    ax2.set_title('Model Prediction Probability Over Time', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Probability (%)', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Actual vs Predicted (Bar comparison)
    ax3 = fig.add_subplot(gs[2, 0])
    comparison_df = pd.DataFrame({
        'Actual UP': [sum(y_test == 1)],
        'Predicted UP': [sum(y_test_pred == 1)],
        'Actual DOWN': [sum(y_test == 0)],
        'Predicted DOWN': [sum(y_test_pred == 0)]
    })
    comparison_df.T.plot(kind='bar', ax=ax3, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    ax3.set_title('Actual vs Predicted Counts', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.legend().remove()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Confusion Matrix
    ax4 = fig.add_subplot(gs[2, 1])
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4, 
                xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'],
                cbar_kws={'label': 'Count'})
    ax4.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Actual', fontsize=12)
    ax4.set_xlabel('Predicted', fontsize=12)
    
    # 5. Prediction Accuracy Over Time (Rolling Window)
    ax5 = fig.add_subplot(gs[3, 0])
    rolling_window = 50
    rolling_accuracy = pd.Series(y_test == y_test_pred).rolling(rolling_window).mean() * 100
    ax5.plot(df_test.index, rolling_accuracy, 'darkblue', linewidth=2)
    ax5.axhline(50, color='red', linestyle='--', linewidth=1, label='Random Guess (50%)')
    ax5.fill_between(df_test.index, 50, rolling_accuracy, 
                     where=(rolling_accuracy >= 50), color='green', alpha=0.3)
    ax5.fill_between(df_test.index, 50, rolling_accuracy, 
                     where=(rolling_accuracy < 50), color='red', alpha=0.3)
    ax5.set_title(f'Rolling Accuracy (Window: {rolling_window} days)', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Accuracy (%)', fontsize=12)
    ax5.set_ylim(0, 100)
    ax5.legend(loc='lower left', fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Probability Distribution
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.hist(y_test_proba[y_test == 1] * 100, bins=30, alpha=0.6, color='green', 
            label=f'Actual UP (n={sum(y_test == 1)})', edgecolor='black')
    ax6.hist(y_test_proba[y_test == 0] * 100, bins=30, alpha=0.6, color='red', 
            label=f'Actual DOWN (n={sum(y_test == 0)})', edgecolor='black')
    ax6.axvline(50, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    ax6.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Predicted Probability (%)', fontsize=12)
    ax6.set_ylabel('Frequency', fontsize=12)
    ax6.legend(loc='upper left', fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{symbol} - Test Phase Analysis ({TEST_START} to {TEST_END})', 
                fontsize=18, fontweight='bold', y=0.995)
    
    # Save figure
    filename = f"{symbol}_test_analysis_{TEST_START}_to_{TEST_END}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Test analysis chart saved: {filename}")
    
    plt.show()

# ==========================
# MAIN EXECUTION
# ==========================
print("\n" + "="*70)
print(f"TRAINING MODEL FOR {SYMBOL}")
print("="*70)

# Load training data
print("\nLoading training data...")
df_train = yf.download(SYMBOL, start=TRAIN_START, end=TRAIN_END, progress=False, auto_adjust=True)

if isinstance(df_train.columns, pd.MultiIndex):
    df_train.columns = df_train.columns.get_level_values(0)

print("Preparing features...")
df_train = prepare_features_advanced(df_train, symbol=SYMBOL, market='US')

print("Fetching news sentiment...")
train_articles = fetch_gdelt_articles(SYMBOL, pd.to_datetime(TRAIN_START), pd.to_datetime(TRAIN_END))
df_train = compute_sentiment(df_train, train_articles)

df_train["target"] = (df_train["Close"].shift(-1) > df_train["Close"]).astype(int)
df_train.dropna(subset=["target"], inplace=True)

ALL_FEATURES = [
    'Close', 'High', 'Open', 'Low', 'Volume',
    'returns', 'volatility', 'momentum',
    'sma10', 'sma20', 'sma50', 'ema10', 'ema20',
    'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
    'rsi', 'macd', 'macd_signal', 'macd_diff',
    'volume_sma', 'volume_ratio',
    'high_low_pct', 'close_open_pct',
    'sentiment', 'sentiment_ma3', 'sentiment_ma7', 'sentiment_slope',
    'roe', 'roa', 'profit_margin', 'operating_margin', 'gross_margin'
]

FEATURES = [f for f in ALL_FEATURES if f in df_train.columns]
print(f"\n✅ Using {len(FEATURES)} features")

X_train = df_train[FEATURES]
y_train = df_train["target"]

print("\n🔧 Scaling features...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("\n🤖 Training XGBoost model...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train_scaled, y_train)
print("✅ Model trained!")

y_train_pred = model.predict(X_train_scaled)
y_train_proba = model.predict_proba(X_train_scaled)[:, 1]

train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_roc_auc = roc_auc_score(y_train, y_train_proba)

print(f"\nTRAINING PERFORMANCE:")
print(f"   Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"   F1-Score:  {train_f1:.4f}")
print(f"   ROC-AUC:   {train_roc_auc:.4f}")

# Load and evaluate on test data
print("\nLoading test data...")
df_test = yf.download(SYMBOL, start=TEST_START, end=TEST_END, progress=False, auto_adjust=True)

if isinstance(df_test.columns, pd.MultiIndex):
    df_test.columns = df_test.columns.get_level_values(0)

df_test = prepare_features_advanced(df_test, symbol=SYMBOL, market='US')

test_articles = fetch_gdelt_articles(SYMBOL, pd.to_datetime(TEST_START), pd.to_datetime(TEST_END))
df_test = compute_sentiment(df_test, test_articles)

X_test = df_test[FEATURES]
y_test = (df_test["Close"].shift(-1) > df_test["Close"]).astype(int)
y_test.fillna(0, inplace=True)

X_test_scaled = scaler.transform(X_test)
y_test_pred = model.predict(X_test_scaled)
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_proba)

print(f"\n📊 TEST PERFORMANCE:")
print(f"   Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   F1-Score:  {test_f1:.4f}")
print(f"   ROC-AUC:   {test_roc_auc:.4f}")

# Plot comprehensive test results
print("\n📊 Generating comprehensive test analysis...")
plot_test_results(df_test, y_test.values, y_test_pred, y_test_proba, SYMBOL)

# Save model
print("\n💾 Saving model...")

model_data = {
    'model': model,
    'scaler': scaler,
    'features': FEATURES,
    'config': {
        'model_type': 'xgboost',
        'symbol': SYMBOL,
        'train_start': TRAIN_START,
        'train_end': TRAIN_END
    },
    'metrics': {
        'accuracy': test_accuracy,
        'f1_score': test_f1,
        'roc_auc': test_roc_auc,
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0)
    },
    'composite_score': (test_accuracy + test_f1 + test_roc_auc) / 3,
    'trained_date': datetime.now()
}

from sklearn.metrics import precision_recall_curve, matthews_corrcoef

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_test_proba)
pr_auc = np.trapz(recall_vals, precision_vals)
mcc = matthews_corrcoef(y_test, y_test_pred)

model_data['metrics']['pr_auc'] = pr_auc
model_data['metrics']['mcc'] = mcc

model_filename = f"{SYMBOL}_best_model.pkl"
with open(model_filename, 'wb') as f:
    pickle.dump(model_data, f)

print(f"✅ Model saved: {model_filename}")

print("\n" + "="*70)
print("📊 FINAL SUMMARY")
print("="*70)
print(f"\n✅ Model trained and saved successfully!")
print(f"\nFile: {model_filename}")
print(f"Symbol: {SYMBOL}")
print(f"Features: {len(FEATURES)}")
print(f"\nMetrics:")
print(f"   Accuracy:  {test_accuracy*100:.2f}%")
print(f"   F1-Score:  {test_f1:.4f}")
print(f"   ROC-AUC:   {test_roc_auc:.4f}")
print(f"   PR-AUC:    {pr_auc:.4f}")
print(f"   MCC:       {mcc:.4f}")
print(f"\n💡 Use this model in the web app by placing it in the same directory.")
print("="*70 + "\n")