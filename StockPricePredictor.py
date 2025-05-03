"""
stock_predictor: Advanced Stock Prediction System

A professional, modular, and extensible stock price prediction system using LSTM and Random Forest models,
technical indicators, and sentiment analysis.

Author: Aidan Frost
"""

import os
import logging
import warnings  # Import the warnings module
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import mplfinance as mpf
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from transformers import pipeline
from newsapi import NewsApiClient
import seaborn as sns
from ta import add_all_ta_features
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, ADXIndicator
from ta.volume import VolumeWeightedAveragePrice
from dotenv import load_dotenv
from typing import Optional, Tuple, Dict, Any

__version__ = "1.0.0"

# ========================
# CONFIGURATION
# ========================
CONFIG = {
    'MODEL': {
        'LSTM_EPOCHS': 100,
        'LSTM_BATCH_SIZE': 32,
        'LSTM_LOOKBACK': 60,
        'RF_ESTIMATORS': 200,
        'RF_MAX_DEPTH': 15,
        'PREDICTION_DAYS_LIMIT': 90,
        'TRAIN_TEST_SPLIT': 0.8
    },
    'DATA': {
        'HISTORY_PERIOD': '5y',
        'INTERVAL': '1d',
        'MAX_NEWS_ARTICLES': 100,
        'MIN_DATA_POINTS': 100
    },
    'VISUALIZATION': {
        'HISTORY_PLOT_DAYS': 500
    }
}

# ========================
# LOGGING SETUP
# ========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ========================
# ENVIRONMENT SETUP
# ========================
load_dotenv()
sns.set_theme(style="whitegrid")
sns.set_palette("deep")
warnings.filterwarnings('ignore')

# ========================
# EXTERNAL SERVICES
# ========================
try:
    sentiment_analyzer = pipeline("text-classification", model="ProsusAI/finbert")
    newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY', 'NEWS_API_KEY'))
    if not newsapi:
        raise ValueError("News API key is not set. Please set the NEWS_API_KEY environment variable.")
    logging.info("Sentiment analysis and news API services initialized successfully.")
except Exception as e:
    logging.warning(f"Error initializing services: {str(e)}")
    sentiment_analyzer = None
    newsapi = None

# =======================
# DATA FUNCTIONS
# =======================

def fetch_stock_data(
    ticker: str,
    period: str = CONFIG['DATA']['HISTORY_PERIOD'],
    interval: str = CONFIG['DATA']['INTERVAL']
) -> Optional[pd.DataFrame]:
    """Fetch historical stock data with optional news sentiment."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data available for ticker {ticker}")
        data.index = pd.to_datetime(data.index)
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        today = pd.Timestamp(datetime.now().date())
        data = data[data.index <= today]
        # Add news sentiment if available
        if newsapi and sentiment_analyzer:
            try:
                news = newsapi.get_everything(
                    q=ticker, language='en', sort_by='publishedAt',
                    page_size=CONFIG['DATA']['MAX_NEWS_ARTICLES']
                )
                sentiments = []
                for article in news.get('articles', [])[:CONFIG['DATA']['MAX_NEWS_ARTICLES']]:
                    try:
                        text = f"{article.get('title', '')}. {article.get('description', '')}"
                        if not text.strip():
                            continue
                        sentiment = sentiment_analyzer(text[:512])[0]
                        score = sentiment['score'] * (1 if sentiment['label'] == 'positive' else -1)
                        sentiments.append({
                            'date': pd.to_datetime(article['publishedAt']).date(),
                            'sentiment': score
                        })
                    except Exception:
                        continue
                if sentiments:
                    sentiment_df = pd.DataFrame(sentiments).groupby('date').mean()
                    data = data.merge(sentiment_df, left_index=True, right_index=True, how='left')
                    data['sentiment'] = data['sentiment'].fillna(0)
                else:
                    data['sentiment'] = 0
            except Exception as e:
                logging.warning(f"Error fetching news sentiment: {str(e)}")
                data['sentiment'] = 0
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def preprocess_data(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Preprocess data and add technical features."""
    if data is None or data.empty:
        return None
    logging.info(f"Raw data shape before preprocessing: {data.shape}")
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    data = data.dropna(subset=required_cols)
    try:
        data = add_all_ta_features(
            data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
        )
    except Exception as e:
        logging.warning(f"Error adding technical indicators: {str(e)}")
    try:
        data['5_day_momentum'] = data['Close'].pct_change(5)
        data['20_day_momentum'] = data['Close'].pct_change(20)
        data['5_day_volatility'] = data['Close'].pct_change().rolling(5).std()
        bb = BollingerBands(data['Close'])
        data['bb_position'] = (data['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility_30D'] = data['Daily_Return'].rolling(window=30).std()
        data['Volume_Change'] = data['Volume'].pct_change()
        for lag in [1, 2, 3, 5, 10, 20]:
            data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
            data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
        for window in [7, 20, 50]:
            data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
            data[f'Volume_MA_{window}'] = data['Volume'].rolling(window=window).mean()
    except Exception as e:
        logging.warning(f"Error adding custom features: {str(e)}")
    data = data.dropna()
    if len(data) < CONFIG['DATA']['MIN_DATA_POINTS']:
        raise ValueError(f"Insufficient data after preprocessing ({len(data)} rows). Need at least {CONFIG['DATA']['MIN_DATA_POINTS']} rows.")
    logging.info(f"Data shape after preprocessing: {data.shape}")
    return data

# ========================
# MODEL FUNCTIONS
# ========================

def build_lstm_model(input_shape: Tuple[int, int]) -> Optional[Sequential]:
    """Build and compile an LSTM model."""
    try:
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True, input_shape=input_shape)),
            Dropout(0.3),
            Bidirectional(LSTM(32, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(16)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mae'])
        return model
    except Exception as e:
        logging.error(f"Error building LSTM model: {str(e)}")
        return None

def prepare_rf_data(data: pd.DataFrame, target_col: str = 'Close') -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Prepare data for Random Forest model."""
    try:
        data = data.copy()
        data['target'] = data[target_col].shift(-1)
        data = data.dropna()
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        data = data[numeric_cols]
        if len(data) < CONFIG['DATA']['MIN_DATA_POINTS']:
            raise ValueError(f"Insufficient data for Random Forest ({len(data)} rows). Need at least {CONFIG['DATA']['MIN_DATA_POINTS']} rows.")
        X = data.drop(['target'], axis=1)
        y = data['target']
        return X, y
    except Exception as e:
        logging.error(f"Error preparing RF data: {str(e)}")
        return None, None

def train_random_forest(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[RandomForestRegressor], Optional[pd.Index]]:
    """Train a Random Forest model."""
    try:
        selector = SelectKBest(score_func=f_regression, k=min(20, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        if len(selected_features) == 0:
            selected_features = X.columns.tolist()
        model = RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2,
            max_features='sqrt', random_state=42, n_jobs=-1, bootstrap=True, max_samples=0.8
        )
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx][selected_features], X.iloc[test_idx][selected_features]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            scores.append(model.score(X_test, y_test))
        logging.info(f"Cross-validation R² scores: {scores}")
        logging.info(f"Mean R²: {np.mean(scores):.2f}")
        model.fit(X[selected_features], y)
        importances = model.feature_importances_
        features_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        logging.info("\nTop 10 Features:")
        logging.info(features_df.head(10))
        return model, selected_features
    except Exception as e:
        logging.error(f"Error training Random Forest: {str(e)}")
        return None, None

def adjust_predictions(predictions: np.ndarray, rsi: float, bb_position: float) -> np.ndarray:
    """Adjust predictions based on technical indicators."""
    try:
        adjustments = np.ones_like(predictions)
        if rsi < 30:
            adjustments *= 1.02
        elif rsi > 70:
            adjustments *= 0.98
        if bb_position < 0.2:
            adjustments *= 1.01
        elif bb_position > 0.8:
            adjustments *= 0.99
        return predictions * adjustments
    except Exception as e:
        logging.error(f"Error adjusting predictions: {str(e)}")
        return predictions

# ====================
# PREDICTION FUNCTIONS
# ====================

def prepare_lstm_data(
    data: pd.DataFrame,
    target_col: str = 'Close',
    n_steps: int = 60,
    test_size: float = 0.2
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[RobustScaler], Optional[list]]:
    """Prepare data for LSTM model."""
    try:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col]
        essential_cols = ['Close', 'Volume', 'sentiment']
        for col in essential_cols:
            if col not in feature_cols and col in data.columns and col != target_col:
                feature_cols.append(col)
        features = data[feature_cols]
        target = data[target_col].values.reshape(-1, 1)
        feature_scaler = RobustScaler()
        target_scaler = RobustScaler()
        scaled_features = feature_scaler.fit_transform(features)
        scaled_target = target_scaler.fit_transform(target)
        X, y = [], []
        for i in range(n_steps, len(scaled_features)):
            X.append(scaled_features[i-n_steps:i, :])
            y.append(scaled_target[i, 0])
        X, y = np.array(X), np.array(y)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, X_test, y_train, y_test, target_scaler, feature_cols
    except Exception as e:
        logging.error(f"Error preparing LSTM data: {str(e)}")
        return None, None, None, None, None, None

def train_lstm_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = CONFIG['MODEL']['LSTM_EPOCHS'],
    batch_size: int = CONFIG['MODEL']['LSTM_BATCH_SIZE']
) -> Tuple[Optional[Sequential], Optional[tf.keras.callbacks.History]]:
    """Train LSTM model."""
    try:
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        return model, history
    except Exception as e:
        logging.error(f"Error training LSTM model: {str(e)}")
        return None, None

def predict_with_lstm(
    model: Sequential,
    data: pd.DataFrame,
    target_scaler: RobustScaler,
    feature_cols: list,
    n_steps: int = 60,
    days_to_predict: int = 30
) -> Optional[pd.DataFrame]:
    """Make predictions using LSTM model."""
    try:
        if not isinstance(feature_cols, list):
            raise ValueError("feature_cols must be a list")
        available_cols = [col for col in feature_cols if col in data.columns]
        if len(available_cols) != len(feature_cols):
            logging.warning(f"{len(feature_cols)-len(available_cols)} features missing from prediction data")
        if 'Close' not in available_cols and 'Close' in data.columns:
            available_cols.append('Close')
        feature_scaler = RobustScaler()
        scaled_features = feature_scaler.fit_transform(data[available_cols])
        predictions = []
        last_sequence = scaled_features[-n_steps:]
        for _ in range(days_to_predict):
            x_input = last_sequence.reshape(1, n_steps, len(available_cols))
            pred = model.predict(x_input, verbose=0)[0, 0]
            predictions.append(pred)
            new_row = np.copy(last_sequence[-1])
            if 'Close' in available_cols:
                new_row[available_cols.index('Close')] = pred
            last_sequence = np.vstack([last_sequence[1:], new_row])
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = target_scaler.inverse_transform(predictions).flatten()
        current_rsi = RSIIndicator(data['Close']).rsi().iloc[-1]
        bb = BollingerBands(data['Close'])
        current_bb_pos = (data['Close'].iloc[-1] - bb.bollinger_lband().iloc[-1]) / (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1])
        predictions = adjust_predictions(predictions, current_rsi, current_bb_pos)
        last_date = data.index[-1]
        prediction_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict+1)]
        return pd.DataFrame({
            'Date': prediction_dates,
            'LSTM_Prediction': predictions
        })
    except Exception as e:
        logging.error(f"Error making LSTM predictions: {str(e)}")
        return None

def predict_with_random_forest(
    model: RandomForestRegressor,
    selected_features: list,
    data: pd.DataFrame,
    days_to_predict: int = 30
) -> Optional[pd.DataFrame]:
    """Make predictions using Random Forest model."""
    try:
        if hasattr(selected_features, 'tolist'):
            selected_features = selected_features.tolist()
        if not isinstance(selected_features, list) or len(selected_features) == 0:
            raise ValueError("Invalid selected_features - must be a non-empty list")
        missing_features = [f for f in selected_features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features[:5]}...")
        last_date = data.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict+1)]
        future_data = pd.DataFrame(
            np.tile(data[selected_features].iloc[-1:].values, (days_to_predict, 1)),
            columns=selected_features,
            index=future_dates
        )
        predictions = []
        close_predictions = []
        for i in range(days_to_predict):
            current_features = future_data.iloc[[i]][selected_features]
            pred = model.predict(current_features)[0]
            predictions.append(pred)
            close_predictions.append(pred)
            if i < days_to_predict - 1:
                for lag in range(1, 4):
                    lag_col = f'Close_Lag_{lag}'
                    if lag_col in selected_features:
                        if lag == 1:
                            future_data.loc[future_dates[i+1], lag_col] = pred
                        else:
                            prev_lag_col = f'Close_Lag_{lag-1}'
                            if prev_lag_col in selected_features:
                                future_data.loc[future_dates[i+1], lag_col] = future_data.loc[future_dates[i], prev_lag_col]
                for window in [7, 20, 50]:
                    ma_col = f'SMA_{window}'
                    if ma_col in selected_features:
                        if len(close_predictions) >= window:
                            future_data.loc[future_dates[i+1], ma_col] = np.mean(close_predictions[-window:])
                        else:
                            hist_data = data['Close'].iloc[-(window-len(close_predictions)):].tolist()
                            future_data.loc[future_dates[i+1], ma_col] = np.mean(hist_data + close_predictions)
        current_rsi = RSIIndicator(data['Close']).rsi().iloc[-1]
        bb = BollingerBands(data['Close'])
        current_bb_pos = (data['Close'].iloc[-1] - bb.bollinger_lband().iloc[-1]) / (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1])
        adjusted_predictions = adjust_predictions(np.array(predictions), current_rsi, current_bb_pos)
        return pd.DataFrame({
            'Date': future_dates,
            'RF_Prediction': adjusted_predictions
        })
    except Exception as e:
        logging.error(f"Error making RF predictions: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ======================
# VISUALIZATION FUNCTIONS
# ======================

def plot_historical_with_predictions(data: pd.DataFrame, predictions: Optional[pd.DataFrame]) -> None:
    """Plot historical data with predictions."""
    try:
        plt.figure(figsize=(16, 8))
        plt.plot(data.index, data['Close'], label='Historical Prices', color='blue', linewidth=2)
        if predictions is not None:
            if 'LSTM_Prediction' in predictions.columns:
                plt.plot(predictions['Date'], predictions['LSTM_Prediction'], label='LSTM Prediction', color='red', linestyle='--', linewidth=2)
            if 'RF_Prediction' in predictions.columns:
                plt.plot(predictions['Date'], predictions['RF_Prediction'], label='Random Forest Prediction', color='green', linestyle='--', linewidth=2)
            if 'Average_Prediction' in predictions.columns:
                plt.plot(predictions['Date'], predictions['Average_Prediction'], label='Ensemble Prediction', color='purple', linestyle='-', linewidth=3)
        plt.scatter(data.index[-1], data['Close'].iloc[-1], color='black', s=150, label='Prediction Start', zorder=5)
        plt.title('Stock Price History with Future Predictions', fontsize=18, pad=20)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price ($)', fontsize=14)
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error generating historical plot: {str(e)}")

def plot_technical_indicators(data: pd.DataFrame) -> None:
    """Plot technical indicators."""
    try:
        fig, axes = plt.subplots(4, 1, figsize=(16, 16), gridspec_kw={'height_ratios': [3, 2, 2, 2]})
        axes[0].plot(data.index, data['Close'], label='Close Price', color='blue', linewidth=2)
        if 'SMA_20' in data.columns:
            axes[0].plot(data.index, data['SMA_20'], label='20-day MA', color='orange', linestyle='--')
        if 'SMA_50' in data.columns:
            axes[0].plot(data.index, data['SMA_50'], label='50-day MA', color='red', linestyle='--')
        try:
            bb = BollingerBands(data['Close'])
            axes[0].fill_between(data.index, bb.bollinger_lband(), bb.bollinger_hband(), color='gray', alpha=0.2, label='Bollinger Bands')
        except:
            pass
        axes[0].set_title('Price and Moving Averages', fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        try:
            rsi = RSIIndicator(data['Close']).rsi()
            axes[1].plot(data.index, rsi, label='RSI (14)', color='purple', linewidth=2)
            axes[1].axhline(70, color='red', linestyle='--', linewidth=1)
            axes[1].axhline(30, color='green', linestyle='--', linewidth=1)
            axes[1].set_title('Relative Strength Index (RSI)', fontsize=14)
            axes[1].set_ylim(0, 100)
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
        except:
            pass
        try:
            macd = MACD(data['Close'])
            axes[2].plot(data.index, macd.macd(), label='MACD', color='blue', linewidth=2)
            axes[2].plot(data.index, macd.macd_signal(), label='Signal Line', color='red', linewidth=2)
            axes[2].bar(data.index, macd.macd_diff(), label='MACD Histogram', color=np.where(macd.macd_diff() > 0, 'g', 'r'))
            axes[2].set_title('Moving Average Convergence Divergence (MACD)', fontsize=14)
            axes[2].legend(fontsize=10)
            axes[2].grid(True, alpha=0.3)
        except:
            pass
        try:
            axes[3].bar(data.index, data['Volume'], label='Volume', color='gray', alpha=0.7)
            if 'Volume_MA_20' in data.columns:
                axes[3].plot(data.index, data['Volume_MA_20'], label='20-day Volume MA', color='blue', linewidth=2)
            axes[3].set_title('Trading Volume', fontsize=14)
            axes[3].legend(fontsize=10)
            axes[3].grid(True, alpha=0.3)
        except:
            pass
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error generating technical indicators plot: {str(e)}")

def plot_model_history(history: tf.keras.callbacks.History) -> None:
    """Plot training history of neural network."""
    try:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error generating model history plot: {str(e)}")

# ======================
# MAIN FUNCTION
# ======================

def analyze_stock(ticker: str, days_to_predict: int = 30) -> Optional[Dict[str, Any]]:
    """Analyze and predict stock performance."""
    logging.info(f"\n{'='*50}")
    tf.keras.backend.clear_session()
    days_to_predict = min(max(1, days_to_predict), CONFIG['MODEL']['PREDICTION_DAYS_LIMIT'])
    logging.info(f"\n📊 Fetching and preprocessing data for {ticker}...")
    data = fetch_stock_data(ticker)
    if data is None or data.empty:
        logging.error(f"Error: No data found for ticker '{ticker}'.")
        return None
    try:
        data = preprocess_data(data)
    except Exception as e:
        logging.error(f"Error preprocessing data: {str(e)}")
        return None
    if data is None or data.empty:
        logging.error(f"Error: No usable data after preprocessing for ticker '{ticker}'.")
        return None
    last_data_date = data.index[-1].strftime('%Y-%m-%d')
    logging.info(f"Analyzing {ticker} - Last available data: {last_data_date}")
    logging.info(f"{'='*50}")
    logging.info(f"\n🤖 Generating {days_to_predict}-day predictions...")
    predictions = None
    logging.info("\nTraining LSTM model...")
    try:
        X_train, X_test, y_train, y_test, target_scaler, feature_cols = prepare_lstm_data(
            data, n_steps=CONFIG['MODEL']['LSTM_LOOKBACK']
        )
        if X_train is not None:
            lstm_model = build_lstm_model((CONFIG['MODEL']['LSTM_LOOKBACK'], len(feature_cols)))
            if lstm_model:
                lstm_model, history = train_lstm_model(lstm_model, X_train, y_train, X_test, y_test)
                lstm_predictions = predict_with_lstm(lstm_model, data, target_scaler, feature_cols,
                                                     CONFIG['MODEL']['LSTM_LOOKBACK'], days_to_predict)
                if lstm_predictions is not None:
                    predictions = lstm_predictions.copy()
                    plot_model_history(history)
    except Exception as e:
        logging.error(f"LSTM failed: {str(e)}")
    logging.info("\nTraining Random Forest model...")
    try:
        X_rf, y_rf = prepare_rf_data(data)
        if X_rf is not None and y_rf is not None:
            rf_model, selected_features = train_random_forest(X_rf, y_rf)
            if rf_model:
                rf_predictions = predict_with_random_forest(rf_model, selected_features, data, days_to_predict)
                if predictions is not None and rf_predictions is not None:
                    predictions['RF_Prediction'] = rf_predictions['RF_Prediction']
                    predictions['Average_Prediction'] = predictions[['LSTM_Prediction', 'RF_Prediction']].mean(axis=1)
                elif rf_predictions is not None:
                    predictions = rf_predictions.copy()
                    predictions['Average_Prediction'] = predictions['RF_Prediction']
    except Exception as e:
        logging.error(f"Random Forest failed: {str(e)}")
    if predictions is None:
        logging.error("\n❌ Error: No predictions generated.")
        return None
    logging.info("\n🔮 Price Predictions:")
    logging.info(predictions.head(days_to_predict))
    current_price = data['Close'].iloc[-1]
    predicted_end_price = predictions['Average_Prediction'].iloc[-1]
    potential_return = (predicted_end_price - current_price) / current_price * 100
    logging.info(f"\n📈 Potential {days_to_predict}-day return: {potential_return:.2f}%")
    plot_technical_indicators(data[-CONFIG['VISUALIZATION']['HISTORY_PLOT_DAYS']:])
    plot_historical_with_predictions(data[-CONFIG['VISUALIZATION']['HISTORY_PLOT_DAYS']:], predictions)
    return {
        'data': data,
        'predictions': predictions,
        'models': {
            'lstm': lstm_model if 'lstm_model' in locals() else None,
            'random_forest': rf_model if 'rf_model' in locals() else None
        }
    }

# ======================
# USER INTERFACE
# ======================

def main() -> None:
    """Main user interface."""
    logging.info("📈 Advanced Stock Market Analysis and Prediction System")
    logging.info("----------------------------------------------------")
    while True:
        ticker = input("\nEnter stock ticker (e.g., AAPL, MSFT) or 'quit' to exit: ").strip().upper()
        if ticker.lower() == 'quit':
            logging.info("Exiting the system. Goodbye!")
            break
        try:
            days = int(input(f"Enter number of days to predict (1-{CONFIG['MODEL']['PREDICTION_DAYS_LIMIT']}): "))
            days = max(1, min(CONFIG['MODEL']['PREDICTION_DAYS_LIMIT'], days))
            result = analyze_stock(ticker, days)
            if result:
                save = input("\nWould you like to save the analysis results? (y/n): ").lower()
                if save == 'y':
                    filename = f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    result['data'].to_csv(filename)
                    logging.info(f"Analysis saved to {filename}")
        except ValueError:
            logging.error("Please enter a valid number of days.")
        except Exception as e:
            logging.error(f"Error: {str(e)}. Please try again.")

if __name__ == "__main__":
    main()