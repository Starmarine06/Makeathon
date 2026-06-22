# trainModel_indian.py
# Complete training script with Times of India sentiment scraping
# Handles inf/NaN values, division by zero, and missing headlines

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import requests
from bs4 import BeautifulSoup
import warnings
import pickle
import seaborn as sns
import time

warnings.filterwarnings('ignore')

# ==========================
# CONFIGURATION
# ==========================
SYMBOL       = "RELIANCE.NS"
TRAIN_START  = "2020-01-01"
TRAIN_END    = "2025-10-12"
TEST_START   = "2018-01-01"
TEST_END     = "2019-12-30"
USER_AGENT   = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
analyzer     = SentimentIntensityAnalyzer()

# ==========================
# HELPER FUNCTIONS
# ==========================

def get_company_name(symbol):
    """Get company name from symbol"""
    base = symbol.replace('.NS','').replace('.BO','')
    names = {
        'RELIANCE': 'Reliance',
        'TCS': 'TCS',
        'INFY': 'Infosys',
        'HDFCBANK': 'HDFC-Bank',
        'ICICIBANK': 'ICICI-Bank',
        'SBIN': 'SBI',
        'BHARTIARTL': 'Airtel',
        'ITC': 'ITC',
        'HINDUNILVR': 'Hindustan-Unilever',
        'LT': 'LT'
    }
    return names.get(base, base)

def fetch_toi_headlines(symbol, start_dt, end_dt):
    """
    Scrape Times of India archive for company headlines.
    Returns list of {'date': 'YYYY-MM-DD', 'text': headline}
    """
    company = get_company_name(symbol)
    results = []
    current = pd.to_datetime(start_dt)
    end_date = pd.to_datetime(end_dt)
    
    print(f"\n📰 Fetching TOI headlines for {company}...")
    print(f"   Date range: {start_dt} to {end_dt}")
    
    days_fetched = 0
    headlines_found = 0
    
    while current <= end_date:
        date_str = current.strftime("%Y%m%d")
        
        # Try multiple URL patterns for TOI
        urls = [
            f"https://timesofindia.indiatimes.com/topic/{company}/{date_str}",
            f"https://timesofindia.indiatimes.com/business/{company}/articlelist/{date_str}.cms"
        ]
        
        headers = {"User-Agent": USER_AGENT}
        
        for url in urls:
            try:
                r = requests.get(url, headers=headers, timeout=10)
                if r.status_code == 200:
                    soup = BeautifulSoup(r.text, "html.parser")
                    
                    # Find headline links
                    for a in soup.find_all('a', href=True):
                        text = a.get_text(strip=True)
                        if len(text) > 20 and company.lower() in text.lower():
                            results.append({
                                "date": current.strftime("%Y-%m-%d"),
                                "text": text
                            })
                            headlines_found += 1
                    
                    if headlines_found > 0:
                        break  # Found headlines, no need to try other URLs
                        
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                time.sleep(1)
                continue
        
        days_fetched += 1
        if days_fetched % 30 == 0:
            print(f"   Progress: {days_fetched} days scanned, {headlines_found} headlines found")
        
        current += pd.Timedelta(days=1)
    
    print(f"✅ TOI: {headlines_found} headlines from {days_fetched} days")
    
    # If no headlines found, create dummy neutral sentiment
    if len(results) == 0:
        print("⚠️ No headlines found. Sentiment will be neutral (0).")
    
    return results

def compute_sentiment(df, articles):
    """
    Compute sentiment scores per trading date using VADER.
    Uses ±3-day window around each date.
    """
    if not articles:
        print("⚠️ No articles - using zero sentiment")
        df["sentiment"] = 0.0
        df["sentiment_ma3"] = 0.0
        df["sentiment_ma7"] = 0.0
        df["sentiment_slope"] = 0.0
        return df
    
    by_date = {}
    for art in articles:
        by_date.setdefault(art["date"], []).append(art["text"])
    
    daily_sent = {}
    for dt in df.index:
        date_str = dt.strftime("%Y-%m-%d")
        texts = []
        
        # ±3 day window
        for offset in range(-3, 4):
            check_date = (dt + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            texts.extend(by_date.get(check_date, []))
        
        if texts:
            scores = [analyzer.polarity_scores(t)["compound"] for t in texts[:20]]
            daily_sent[dt] = float(np.mean(scores))
        else:
            daily_sent[dt] = 0.0
    
    df["sentiment"] = pd.Series(daily_sent)
    df["sentiment_ma3"] = df["sentiment"].rolling(3, min_periods=1).mean()
    df["sentiment_ma7"] = df["sentiment"].rolling(7, min_periods=1).mean()
    df["sentiment_slope"] = df["sentiment"] - df["sentiment"].shift(3).fillna(0)
    
    # Stats
    non_zero = df["sentiment"][df["sentiment"] != 0]
    if len(non_zero) > 0:
        print(f"✅ Sentiment: {len(non_zero)}/{len(df)} days ({len(non_zero)/len(df)*100:.1f}%)")
        print(f"   Avg: {non_zero.mean():.3f} | Pos: {len(non_zero[non_zero>0])} | Neg: {len(non_zero[non_zero<0])}")
    
    return df

def prepare_features(df):
    """
    Prepare all technical and sentiment features.
    Handles inf/NaN values safely.
    """
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Basic features
    df["returns"] = df["Close"].pct_change().fillna(0)
    df["volatility"] = df["returns"].rolling(10).std().fillna(0)
    df["momentum"] = df["Close"] / df["Close"].shift(10) - 1
    
    # Moving averages
    df["sma10"] = df["Close"].rolling(10).mean()
    df["sma20"] = df["Close"].rolling(20).mean()
    df["sma50"] = df["Close"].rolling(50).mean()
    df["ema10"] = df["Close"].ewm(span=10).mean()
    df["ema20"] = df["Close"].ewm(span=20).mean()
    
    # Bollinger Bands (handle division by zero)
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std().fillna(0)
    
    df["bb_mid"] = bb_mid
    df["bb_up"] = bb_mid + 2.5 * bb_std
    df["bb_low"] = bb_mid - 2.5 * bb_std
    df["bb_width"] = df["bb_up"] - df["bb_low"]
    
    # Safe division for bb_position
    bb_range = df["bb_up"] - df["bb_low"]
    df["bb_pos"] = np.where(bb_range > 0, 
                            (df["Close"] - df["bb_low"]) / bb_range, 
                            0.5)
    
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean().fillna(0)
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().fillna(0)
    rs = np.where(loss > 0, gain / loss, 0)
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_sig"] = df["macd"].ewm(span=9).mean()
    df["macd_diff"] = df["macd"] - df["macd_sig"]
    
    # Volume (safe division)
    vol_sma = df["Volume"].rolling(20).mean().fillna(1)
    df["vol_sma"] = vol_sma
    df["vol_ratio"] = np.where(vol_sma > 0, df["Volume"] / vol_sma, 1)
    
    # Price ranges (safe division)
    df["high_low_pct"] = np.where(df["Close"] > 0,
                                  (df["High"] - df["Low"]) / df["Close"],
                                  0)
    df["close_open_pct"] = np.where(df["Open"] > 0,
                                    (df["Close"] - df["Open"]) / df["Open"],
                                    0)
    
    # Gap and intraday (safe division)
    prev_close = df["Close"].shift(1).fillna(df["Close"])
    df["gap"] = np.where(prev_close > 0,
                         (df["Open"] - prev_close) / prev_close,
                         0)
    
    df["intra_range"] = np.where(df["Open"] > 0,
                                  (df["High"] - df["Low"]) / df["Open"],
                                  0)
    
    # Sentiment placeholders
    for col in ["sentiment", "sentiment_ma3", "sentiment_ma7", "sentiment_slope"]:
        if col not in df.columns:
            df[col] = 0.0
    
    # Final cleanup
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def plot_results(df_test, y_test, y_pred, y_prob, symbol):
    """
    Create comprehensive visualization with sentiment.
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # 1. Price with predictions
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_test.index, df_test["Close"], 'b-', linewidth=2, alpha=0.7, label='Price')
    
    correct = (y_test == y_pred)
    ax1.scatter(df_test.index[correct], df_test["Close"][correct],
                color='green', marker='o', s=30, alpha=0.5, label='Correct')
    ax1.scatter(df_test.index[~correct], df_test["Close"][~correct],
                color='red', marker='x', s=30, alpha=0.5, label='Wrong')
    
    ax1.set_title(f'{symbol} - Test Results with Sentiment', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price (₹)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Prediction probability
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(df_test.index, y_prob * 100, 'purple', linewidth=2)
    ax2.axhline(50, color='gray', linestyle='--')
    ax2.fill_between(df_test.index, 50, y_prob * 100,
                     where=(y_prob * 100 >= 50), color='green', alpha=0.2)
    ax2.fill_between(df_test.index, 50, y_prob * 100,
                     where=(y_prob * 100 < 50), color='red', alpha=0.2)
    ax2.set_title('Prediction Probability', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Probability (%)', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion matrix
    ax3 = fig.add_subplot(gs[2, 0])
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # 4. Rolling accuracy
    ax4 = fig.add_subplot(gs[2, 1])
    rolling_acc = pd.Series(y_test == y_pred).rolling(50, min_periods=1).mean() * 100
    ax4.plot(df_test.index, rolling_acc, 'darkblue', linewidth=2)
    ax4.axhline(50, color='red', linestyle='--', label='Random')
    ax4.fill_between(df_test.index, 50, rolling_acc,
                     where=(rolling_acc >= 50), color='green', alpha=0.3)
    ax4.set_title('Rolling Accuracy (50-day)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Accuracy (%)', fontsize=12)
    ax4.set_ylim(0, 100)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Sentiment vs Price
    ax5 = fig.add_subplot(gs[3, :])
    ax5_twin = ax5.twinx()
    
    ax5.plot(df_test.index, df_test["Close"], 'b-', linewidth=2, alpha=0.7)
    ax5.set_ylabel('Price (₹)', color='blue', fontsize=12)
    ax5.tick_params(axis='y', labelcolor='blue')
    
    sentiment = df_test["sentiment"]
    colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in sentiment]
    ax5_twin.bar(df_test.index, sentiment, alpha=0.4, color=colors, width=1)
    ax5_twin.set_ylabel('Sentiment', color='purple', fontsize=12)
    ax5_twin.tick_params(axis='y', labelcolor='purple')
    ax5_twin.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    ax5.set_title('Price vs Sentiment (TOI)', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle(f'{symbol} - Complete Analysis', fontsize=18, fontweight='bold', y=0.995)
    
    filename = f"{symbol.replace('.', '_')}_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Chart saved: {filename}")
    plt.close()

# ==========================
# MAIN TRAINING WORKFLOW
# ==========================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(f"📊 TRAINING MODEL: {SYMBOL}")
    print("🇮🇳 Indian Markets with Times of India Sentiment")
    print("="*70)
    
    # 1. Load training data
    print(f"\n📊 Loading training data ({TRAIN_START} to {TRAIN_END})...")
    df_train = yf.download(SYMBOL, start=TRAIN_START, end=TRAIN_END, 
                           progress=False, auto_adjust=True)
    
    if df_train.empty:
        print("❌ No data downloaded. Check symbol and dates.")
        exit()
    
    print(f"✅ Loaded {len(df_train)} trading days")
    
    # 2. Prepare technical features
    print("\n📊 Preparing technical features...")
    df_train = prepare_features(df_train)
    
    # 3. Fetch sentiment
    headlines = fetch_toi_headlines(SYMBOL, TRAIN_START, TRAIN_END)
    df_train = compute_sentiment(df_train, headlines)
    
    # 4. Create target
    print("\n📊 Creating target variable...")
    df_train["target"] = (df_train["Close"].shift(-1) > df_train["Close"]).astype(int)
    df_train.dropna(subset=["target"], inplace=True)
    
    # Drop first 50 rows (incomplete rolling stats)
    df_train = df_train.iloc[50:]
    
    print(f"✅ Training samples: {len(df_train)}")
    
    # 5. Features
    features = [
        "returns", "volatility", "momentum",
        "sma10", "sma20", "sma50", "ema10", "ema20",
        "bb_mid", "bb_up", "bb_low", "bb_width", "bb_pos",
        "rsi", "macd", "macd_sig", "macd_diff",
        "vol_sma", "vol_ratio",
        "high_low_pct", "close_open_pct",
        "gap", "intra_range",
        "sentiment", "sentiment_ma3", "sentiment_ma7", "sentiment_slope"
    ]
    
    X_train = df_train[features]
    y_train = df_train["target"]
    
    # Final safety check
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"✅ Using {len(features)} features")
    
    # 6. Scale
    print("\n🔧 Scaling features...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # 7. Train
    print("\n🤖 Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=3,
        gamma=0.1,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )
    
    model.fit(X_scaled, y_train)
    print("✅ Model trained!")
    
    # Training metrics
    y_train_pred = model.predict(X_scaled)
    y_train_prob = model.predict_proba(X_scaled)[:, 1]
    
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    
    print(f"\n📊 TRAINING PERFORMANCE:")
    print(f"   Accuracy: {train_acc*100:.2f}%")
    print(f"   F1-Score: {train_f1:.4f}")
    
    # 8. Test
    print(f"\n📊 Loading test data ({TEST_START} to {TEST_END})...")
    df_test = yf.download(SYMBOL, start=TEST_START, end=TEST_END,
                          progress=False, auto_adjust=True)
    
    if not df_test.empty:
        print(f"✅ Loaded {len(df_test)} test days")
        
        df_test = prepare_features(df_test)
        headlines_test = fetch_toi_headlines(SYMBOL, TEST_START, TEST_END)
        df_test = compute_sentiment(df_test, headlines_test)
        
        df_test = df_test.iloc[50:]  # Drop first 50 rows
        
        y_test = (df_test["Close"].shift(-1) > df_test["Close"]).astype(int).fillna(0)
        X_test = df_test[features].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        
        print(f"\n📊 TEST PERFORMANCE:")
        print(f"   Accuracy: {test_acc*100:.2f}%")
        print(f"   F1-Score: {test_f1:.4f}")
        
        # Plot
        plot_results(df_test, y_test.values, y_pred, y_prob, SYMBOL)
    
    # 9. Save model
    print("\n💾 Saving model...")
    model_filename = f"{SYMBOL.replace('.', '_')}_model.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'features': features,
            'config': {
                'symbol': SYMBOL,
                'train_start': TRAIN_START,
                'train_end': TRAIN_END,
                'sentiment_enabled': True
            }
        }, f)
    
    print(f"✅ Model saved: {model_filename}")
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)