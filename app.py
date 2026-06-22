from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os

app = Flask(__name__)

# Cache for loaded models
model_cache = {}

def load_model_for_symbol(symbol):
    """Load model for specific symbol, use cache if available"""
    if symbol in model_cache:
        return model_cache[symbol]
    
    model_file = f"{symbol}_model.pkl"
    
    if os.path.exists(model_file):
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            model_info = {
                'model': model_data['model'],
                'scaler': model_data['scaler'],
                'features': list(model_data['model'].feature_names_in_) if hasattr(model_data['model'], 'feature_names_in_') else None
            }
            
            model_cache[symbol] = model_info
            print(f"✅ Loaded model for {symbol}: {model_file}")
            return model_info
        except Exception as e:
            print(f"❌ Error loading model for {symbol}: {e}")
            return None
    else:
        print(f"⚠️ No model found for {symbol}, using momentum-based predictions")
        return None

def prepare_features_advanced(df):
    """Prepare all advanced features"""
    df["returns"] = df["Close"].pct_change().fillna(0)
    df["volatility"] = df["returns"].rolling(10).std().fillna(0)
    df["momentum"] = df["Close"] / df["Close"].shift(10) - 1
    
    # Moving averages
    df["sma10"] = df["Close"].rolling(10).mean()
    df["sma20"] = df["Close"].rolling(20).mean()
    df["sma50"] = df["Close"].rolling(50).mean()
    df["ema10"] = df["Close"].ewm(span=10).mean()
    df["ema20"] = df["Close"].ewm(span=20).mean()
    
    # Bollinger Bands
    df["bb_middle"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + (2 * bb_std)
    df["bb_lower"] = df["bb_middle"] - (2 * bb_std)
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_diff"] = df["macd"] - df["macd_signal"]
    
    # Volume
    df["volume_sma"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_sma"]
    
    # Price features
    df["high_low_pct"] = (df["High"] - df["Low"]) / df["Close"]
    df["close_open_pct"] = (df["Close"] - df["Open"]) / df["Open"]
    
    # Sentiment placeholders
    df["sentiment"] = 0.0
    df["sentiment_ma3"] = 0.0
    df["sentiment_ma7"] = 0.0
    df["sentiment_slope"] = 0.0
    
    df.fillna(0, inplace=True)
    return df

def get_predictions(df, df_combined, model_info):
    """Get ML predictions using symbol-specific model"""
    if model_info and model_info['model'] and model_info['scaler'] and model_info['features']:
        try:
            FEATURES = model_info['features']
            missing_features = set(FEATURES) - set(df_combined.columns)
            if missing_features:
                print(f"⚠️ Missing features: {missing_features}")
                return None
            
            X = df_combined[FEATURES].tail(len(df))
            X_scaled = model_info['scaler'].transform(X)
            probs = model_info['model'].predict_proba(X_scaled)[:, 1] * 100
            return probs
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    # Fallback: momentum-based
    probs = []
    for i, row in df.iterrows():
        if i in df_combined.index:
            momentum = df_combined.loc[i, 'momentum']
            prob = 50 + (momentum * 100)
            prob = max(10, min(90, prob))
            probs.append(prob)
    return np.array(probs)

def detect_signals(probs, prices, threshold_buy=52, threshold_sell=48, cooldown=2):
    """Detect BUY and SELL signals"""
    signals = []
    last_signal_index = -cooldown
    
    for i in range(1, len(probs)):
        current_prob = probs[i]
        prev_prob = probs[i-1]
        
        if i - last_signal_index < cooldown:
            continue
        
        if current_prob >= threshold_buy and prev_prob < threshold_buy:
            signals.append({'index': i, 'type': 'BUY', 'prob': current_prob})
            last_signal_index = i
        
        elif current_prob <= threshold_sell and prev_prob > threshold_sell:
            signals.append({'index': i, 'type': 'SELL', 'prob': current_prob})
            last_signal_index = i
        
        elif i > 5:
            prob_change = current_prob - probs[i-3]
            if prob_change > 8 and current_prob > 50:
                signals.append({'index': i, 'type': 'BUY', 'prob': current_prob})
                last_signal_index = i
        
        elif i > 5:
            prob_change = probs[i-3] - current_prob
            if prob_change > 8 and current_prob < 50:
                signals.append({'index': i, 'type': 'SELL', 'prob': current_prob})
                last_signal_index = i
    
    return signals

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/historical/<symbol>')
def get_historical_data(symbol):
    try:
        symbol = symbol.upper().strip()
        
        # Load model for this symbol
        model_info = load_model_for_symbol(symbol)
        
        # Get data
        df = yf.download(symbol, period="2y", interval="1h", progress=False, auto_adjust=True)
        
        if df.empty:
            return jsonify({'error': f'No data available for {symbol}'}), 404
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        
        # Get historical data
        df_hist = yf.download(symbol, period="90d", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df_hist.columns, pd.MultiIndex):
            df_hist.columns = df_hist.columns.get_level_values(0)
        if hasattr(df_hist.index, 'tz') and df_hist.index.tz is not None:
            df_hist.index = df_hist.index.tz_convert(None)
        
        # Combine and prepare
        df_combined = pd.concat([df_hist, df]).drop_duplicates()
        df_combined = prepare_features_advanced(df_combined)
        
        # Get predictions using symbol-specific model
        probs = get_predictions(df, df_combined, model_info)
        
        if probs is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Detect signals
        prices = df['Close'].values
        signal_points = detect_signals(probs, prices, threshold_buy=52, threshold_sell=48, cooldown=2)
        
        print(f"✅ {symbol}: Found {len(signal_points)} signals")
        
        # Prepare candlestick data
        candlestick_data = []
        for idx, row in df.iterrows():
            timestamp = int(idx.timestamp())
            candlestick_data.append({
                'time': timestamp,
                'open': round(float(row['Open']), 2),
                'high': round(float(row['High']), 2),
                'low': round(float(row['Low']), 2),
                'close': round(float(row['Close']), 2)
            })
        
        # Prepare markers with enhanced visuals
        markers = []
        for sig in signal_points:
            idx = sig['index']
            if idx < len(candlestick_data):
                if sig['type'] == 'BUY':
                    markers.append({
                        'time': candlestick_data[idx]['time'],
                        'position': 'belowBar',
                        'color': '#00ff88',  # Bright green
                        'shape': 'arrowUp',
                        'text': f"BUY {sig['prob']:.0f}%",
                        'size': 3  # Larger size
                    })
                else:  # SELL
                    markers.append({
                        'time': candlestick_data[idx]['time'],
                        'position': 'aboveBar',
                        'color': '#ff4444',  # Bright red
                        'shape': 'arrowDown',
                        'text': f"SELL {sig['prob']:.0f}%",
                        'size': 3  # Larger size
                    })
        
        latest_price = float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[0])
        change = latest_price - prev_close
        change_pct = (change / prev_close) * 100
        
        latest_prob = float(probs[-1])
        signal = "BUY" if latest_prob >= 52 else "SELL" if latest_prob <= 48 else "HOLD"
        
        buy_signals = [s for s in signal_points if s['type'] == 'BUY']
        sell_signals = [s for s in signal_points if s['type'] == 'SELL']
        
        return jsonify({
            'symbol': symbol,
            'candlestick_data': candlestick_data,
            'markers': markers,
            'current_price': round(latest_price, 2),
            'change': round(change, 2),
            'change_pct': round(change_pct, 2),
            'day_high': round(float(df['High'].max()), 2),
            'day_low': round(float(df['Low'].min()), 2),
            'volume': int(df['Volume'].sum()),
            'probability': round(latest_prob, 1),
            'signal': signal,
            'total_signals': len(markers),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'has_model': model_info is not None
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/live/<symbol>')
def get_live_update(symbol):
    try:
        symbol = symbol.upper().strip()
        model_info = load_model_for_symbol(symbol)
        
        df = yf.download(symbol, period="1d", interval="1m", progress=False, auto_adjust=True)
        
        if df.empty:
            return jsonify({'error': 'Market closed'}), 404
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        
        latest = df.iloc[-1]
        latest_price = float(latest['Close'])
        
        first_price = float(df['Close'].iloc[0])
        change = latest_price - first_price
        change_pct = (change / first_price) * 100
        
        if model_info and model_info['model'] and model_info['scaler'] and model_info['features']:
            df_hist = yf.download(symbol, period="60d", interval="1d", progress=False, auto_adjust=True)
            if isinstance(df_hist.columns, pd.MultiIndex):
                df_hist.columns = df_hist.columns.get_level_values(0)
            
            df_combined = pd.concat([df_hist, df.tail(50)]).drop_duplicates()
            df_combined = prepare_features_advanced(df_combined)
            
            try:
                X = df_combined[model_info['features']].tail(1)
                X_scaled = model_info['scaler'].transform(X)
                prob = float(model_info['model'].predict_proba(X_scaled)[:, 1][0] * 100)
            except:
                prob = 50 + (change_pct * 2)
                prob = max(10, min(90, prob))
        else:
            prob = 50 + (change_pct * 2)
            prob = max(10, min(90, prob))
        
        signal = "BUY" if prob >= 52 else "SELL" if prob <= 48 else "HOLD"
        
        return jsonify({
            'symbol': symbol,
            'price': round(latest_price, 2),
            'change': round(change, 2),
            'change_pct': round(change_pct, 2),
            'day_high': round(float(df['High'].max()), 2),
            'day_low': round(float(df['Low'].min()), 2),
            'volume': int(df['Volume'].sum()),
            'probability': round(prob, 1),
            'signal': signal,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n🚀 Stock Trading AI - Dynamic Model Loading")
    print("📱 Open: http://localhost:5000")
    print("\n💡 Place model files as: SYMBOL_best_model.pkl")
    print("   Example: AAPL_best_model.pkl, TSLA_best_model.pkl\n")
    app.run(host='0.0.0.0', port=5000, debug=True)