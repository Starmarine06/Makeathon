# test_all_models.py
import yfinance as yf
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import requests
from bs4 import BeautifulSoup
import warnings
import time

warnings.filterwarnings('ignore')
SYMBOL = "RELIANCE.NS"
TRAIN_START, TRAIN_END = "2020-01-01", "2025-10-12"
TEST_START, TEST_END   = "2018-01-01", "2019-12-30"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

analyzer = SentimentIntensityAnalyzer()

def get_company_name(symbol):
    base = symbol.replace('.NS','').replace('.BO','')
    names = {
        'RELIANCE': 'Reliance','TCS': 'TCS','INFY': 'Infosys',
        'HDFCBANK': 'HDFC-Bank','ICICIBANK': 'ICICI-Bank','SBIN': 'SBI',
        'BHARTIARTL': 'Airtel','ITC': 'ITC','HINDUNILVR': 'Hindustan-Unilever','LT': 'LT'
    }
    return names.get(base, base)

def fetch_toi_headlines(symbol, start_dt, end_dt):
    company = get_company_name(symbol)
    results = []
    current = pd.to_datetime(start_dt)
    end_date = pd.to_datetime(end_dt)
    while current <= end_date:
        date_str = current.strftime("%Y%m%d")
        url = f"https://timesofindia.indiatimes.com/topic/{company}/{date_str}"
        headers = {"User-Agent": USER_AGENT}
        try:
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                for a in soup.find_all('a', href=True):
                    text = a.get_text(strip=True)
                    if len(text) > 20:
                        results.append({"date": current.strftime("%Y-%m-%d"), "text": text})
            time.sleep(0.2)
        except:
            pass
        current += pd.Timedelta(days=7)
    return results

def compute_sentiment(df, articles):
    if not articles:
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
        texts = []
        for offset in range(-3, 4):
            check_date = (dt + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            texts.extend(by_date.get(check_date, []))
        if texts:
            scores = [analyzer.polarity_scores(t)["compound"] for t in texts[:15]]
            daily_sent[dt] = float(np.mean(scores))
        else:
            daily_sent[dt] = 0.0
    df["sentiment"] = pd.Series(daily_sent)
    df["sentiment_ma3"] = df["sentiment"].rolling(3, min_periods=1).mean()
    df["sentiment_ma7"] = df["sentiment"].rolling(7, min_periods=1).mean()
    df["sentiment_slope"] = df["sentiment"] - df["sentiment"].shift(3).fillna(0)
    return df

def prepare_features(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["returns"] = df["Close"].pct_change().fillna(0)
    df["volatility"] = df["returns"].rolling(10).std().fillna(0)
    df["momentum"] = df["Close"] / df["Close"].shift(10) - 1
    df["sma10"] = df["Close"].rolling(10).mean()
    df["sma20"] = df["Close"].rolling(20).mean()
    df["sma50"] = df["Close"].rolling(50).mean()
    df["ema10"] = df["Close"].ewm(span=10).mean()
    df["ema20"] = df["Close"].ewm(span=20).mean()
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std().fillna(0)
    df["bb_mid"] = bb_mid
    df["bb_up"] = bb_mid + 2.5 * bb_std
    df["bb_low"] = bb_mid - 2.5 * bb_std
    df["bb_width"] = df["bb_up"] - df["bb_low"]
    bb_range = df["bb_up"] - df["bb_low"]
    df["bb_pos"] = np.where(bb_range > 0, (df["Close"] - df["bb_low"]) / bb_range, 0.5)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean().fillna(0)
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().fillna(0)
    rs = np.where(loss > 0, gain / loss, 0)
    df["rsi"] = 100 - (100 / (1 + rs))
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_sig"] = df["macd"].ewm(span=9).mean()
    df["macd_diff"] = df["macd"] - df["macd_sig"]
    vol_sma = df["Volume"].rolling(20).mean().fillna(1)
    df["vol_sma"] = vol_sma
    df["vol_ratio"] = np.where(vol_sma > 0, df["Volume"] / vol_sma, 1)
    df["high_low_pct"] = np.where(df["Close"] > 0, (df["High"] - df["Low"]) / df["Close"], 0)
    df["close_open_pct"] = np.where(df["Open"] > 0, (df["Close"] - df["Open"]) / df["Open"], 0)
    prev_close = df["Close"].shift(1).fillna(df["Close"])
    df["gap"] = np.where(prev_close > 0, (df["Open"] - prev_close) / prev_close, 0)
    df["intra_range"] = np.where(df["Open"] > 0, (df["High"] - df["Low"]) / df["Open"], 0)
    df["tr"] = df[["High","Low","Close"]].apply(lambda x: max(x["High"]-x["Low"], abs(x["High"]-x["Close"]), abs(x["Low"]-x["Close"])), axis=1)
    df["atr14"] = df["tr"].rolling(14).mean().fillna(0)
    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["stoch_k"] = 100 * (df["Close"]-low14) / (high14-low14).replace(0,1)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean().fillna(0)
    for col in ["sentiment", "sentiment_ma3", "sentiment_ma7", "sentiment_slope"]:
        if col not in df.columns:
            df[col] = 0.0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

features = [
    "returns", "volatility", "momentum",
    "sma10", "sma20", "sma50", "ema10", "ema20",
    "bb_mid", "bb_up", "bb_low", "bb_width", "bb_pos",
    "rsi", "macd", "macd_sig", "macd_diff",
    "vol_sma", "vol_ratio", "high_low_pct","close_open_pct",
    "gap","intra_range","sentiment","sentiment_ma3",
    "sentiment_ma7", "sentiment_slope","atr14","stoch_k","stoch_d"
]

# ========== DATA PREP ===========
print("Loading and processing data...")
df_train = yf.download(SYMBOL, start=TRAIN_START, end=TRAIN_END, progress=False, auto_adjust=True)
df_train = prepare_features(df_train)
headlines = fetch_toi_headlines(SYMBOL, TRAIN_START, TRAIN_END)
df_train = compute_sentiment(df_train, headlines)
df_train["target"] = (df_train["Close"].shift(-1) > df_train["Close"]).astype(int)
df_train.dropna(subset=["target"], inplace=True)
df_train = df_train.iloc[50:]

X_train = df_train[features].replace([np.inf, -np.inf], np.nan).fillna(0)
y_train = df_train["target"]

df_test = yf.download(SYMBOL, start=TEST_START, end=TEST_END, progress=False, auto_adjust=True)
df_test = prepare_features(df_test)
headlines_test = fetch_toi_headlines(SYMBOL, TEST_START, TEST_END)
df_test = compute_sentiment(df_test, headlines_test)
df_test = df_test.iloc[50:]
X_test = df_test[features].replace([np.inf, -np.inf], np.nan).fillna(0)
y_test = (df_test["Close"].shift(-1) > df_test["Close"]).astype(int).fillna(0)

scaler = RobustScaler()
Xtr = scaler.fit_transform(X_train)
Xte = scaler.transform(X_test)

print("\nTesting all models with default parameters...\n")

# ========== MODELS ===========
xgb = XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)
rf  = RandomForestClassifier(random_state=42)
lr  = LogisticRegression(solver="lbfgs", max_iter=300)

models = [
    ('XGBoost', xgb), ('RandomForest', rf), ('LogisticRegression', lr)
]

for name, model in models:
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    print(f"{name:20s}  acc={acc*100:.2f}%   f1={f1:.3f}")

# ========== STACKED ENSEMBLE ===========
print("\nEvaluating stacked ensemble...")
ensemble = StackingClassifier(
    estimators=[('xgb', xgb), ('rf', rf), ('lr', lr)],
    final_estimator=LogisticRegression(max_iter=300),
    n_jobs=-1
)
ensemble.fit(Xtr, y_train)
y_pred_ens = ensemble.predict(Xte)
acc_ens = accuracy_score(y_test, y_pred_ens)
f1_ens  = f1_score(y_test, y_pred_ens)
print(f"{'StackedEnsemble':20s}  acc={acc_ens*100:.2f}%   f1={f1_ens:.3f}")

# ========== SUMMARY ===========
print("\nAll models tested on the same data and split for fair comparison.")