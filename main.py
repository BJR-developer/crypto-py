from flask import Flask, jsonify, request
from langchain_google_genai import ChatGoogleGenerativeAI
from binance.client import Client
from binance.exceptions import BinanceAPIException
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from functools import lru_cache
import ta  # Technical Analysis library
import requests
from requests.exceptions import RequestException, Timeout
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize Binance client with proper timeout settings
def create_binance_client(max_retries=3, timeout=30):
    """Create a Binance client with retry logic"""
    for attempt in range(max_retries):
        try:
            # Configure client with longer timeout and testnet
            client = Client(
                None,  # API Key (not required for public endpoints)
                None,  # API Secret (not required for public endpoints)
                requests_params={
                    'timeout': timeout,
                    'verify': True
                }
            )
            
            # Test connection with a simple API call
            client.get_system_status()
            logger.info("Successfully connected to Binance API")
            return client
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
            
        except (RequestException, Timeout) as e:
            logger.error(f"Connection error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
            
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
            
def get_advanced_technical_indicators(df: pd.DataFrame) -> dict:
    """Calculate advanced technical indicators"""
    try:
        # Volume-based indicators
        df['MFI'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
        df['ADI'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])
        df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Momentum indicators
        df['RSI'] = ta.momentum.rsi(df['close'])
        df['Stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['StochRSI'] = ta.momentum.stochrsi(df['close'])
        df['MACD'] = ta.trend.macd_diff(df['close'])
        
        # Trend indicators
        df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'])
        df['CCI'] = ta.trend.cci(df['high'], df['low'], df['close'])
        df['DPO'] = ta.trend.dpo(df['close'])
        
        # Volatility indicators
        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # Keltner Channels
        kc = ta.volatility.KeltnerChannel(high=df['high'], low=df['low'], close=df['close'])
        df['KC_high'] = kc.keltner_channel_hband()
        df['KC_mid'] = kc.keltner_channel_mband()
        df['KC_low'] = kc.keltner_channel_lband()
        
        # Get latest values
        latest = df.iloc[-1]
        
        return {
            'momentum': {
                'rsi': float(latest['RSI']) if not pd.isna(latest['RSI']) else 0,
                'stoch': float(latest['Stoch']) if not pd.isna(latest['Stoch']) else 0,
                'stoch_rsi': float(latest['StochRSI']) if not pd.isna(latest['StochRSI']) else 0,
                'macd': float(latest['MACD']) if not pd.isna(latest['MACD']) else 0
            },
            'volume': {
                'mfi': float(latest['MFI']) if not pd.isna(latest['MFI']) else 0,
                'adi': float(latest['ADI']) if not pd.isna(latest['ADI']) else 0,
                'obv': float(latest['OBV']) if not pd.isna(latest['OBV']) else 0
            },
            'trend': {
                'adx': float(latest['ADX']) if not pd.isna(latest['ADX']) else 0,
                'cci': float(latest['CCI']) if not pd.isna(latest['CCI']) else 0,
                'dpo': float(latest['DPO']) if not pd.isna(latest['DPO']) else 0
            },
            'volatility': {
                'bb': {
                    'upper': float(latest['BB_upper']) if not pd.isna(latest['BB_upper']) else 0,
                    'middle': float(latest['BB_middle']) if not pd.isna(latest['BB_middle']) else 0,
                    'lower': float(latest['BB_lower']) if not pd.isna(latest['BB_lower']) else 0
                },
                'atr': float(latest['ATR']) if not pd.isna(latest['ATR']) else 0,
                'kc': {
                    'high': float(latest['KC_high']) if not pd.isna(latest['KC_high']) else 0,
                    'mid': float(latest['KC_mid']) if not pd.isna(latest['KC_mid']) else 0,
                    'low': float(latest['KC_low']) if not pd.isna(latest['KC_low']) else 0
                }
            }
        }
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        return {
            'momentum': {'rsi': 0, 'stoch': 0, 'stoch_rsi': 0, 'macd': 0},
            'volume': {'mfi': 0, 'adi': 0, 'obv': 0},
            'trend': {'adx': 0, 'cci': 0, 'dpo': 0},
            'volatility': {
                'bb': {'upper': 0, 'middle': 0, 'lower': 0},
                'atr': 0,
                'kc': {'high': 0, 'mid': 0, 'low': 0}
            }
        }

# Initialize Binance client
try:
    binance_client = create_binance_client(max_retries=3, timeout=30)
    logger.info("Binance client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Binance client: {str(e)}")
    binance_client = None

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.7,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def get_market_sentiment(symbol: str) -> dict:
    """Get market sentiment from multiple sources"""
    try:
        # Get fear and greed index
        fear_greed_url = "https://api.alternative.me/fng/"
        fear_greed = requests.get(fear_greed_url).json()
        
        # Get Google Trends data
        from pytrends.request import TrendReq
        pytrends = TrendReq()
        pytrends.build_payload([symbol], timeframe='now 7-d')
        interest = pytrends.interest_over_time()
        
        # Get social media sentiment (placeholder - you'd need API keys for real implementation)
        social_sentiment = {
            'twitter': 0.65,  # Example value
            'reddit': 0.58    # Example value
        }
        
        return {
            'fear_greed': {
                'value': fear_greed['data'][0]['value'],
                'classification': fear_greed['data'][0]['value_classification']
            },
            'google_trends': {
                'current': interest[symbol].iloc[-1] if not interest.empty else 0,
                'trend': interest[symbol].pct_change().mean() if not interest.empty else 0
            },
            'social': social_sentiment
        }
    except Exception as e:
        logger.error(f"Error getting market sentiment: {str(e)}")
        return {}

def predict_price(df: pd.DataFrame) -> dict:
    """Predict future prices using multiple models"""
    try:
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        import statsmodels.api as sm
        
        # Prepare data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['close']])
        
        # LSTM prediction
        sequence_length = 10
        X = []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length), 0])
        X = np.array(X)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Fit model with last 20% of data
        split = int(len(X) * 0.8)
        model.fit(X[split:], scaled_data[sequence_length + split:], epochs=50, batch_size=32, verbose=0)
        
        # Make predictions
        last_sequence = scaled_data[-sequence_length:]
        lstm_pred = model.predict(last_sequence.reshape(1, sequence_length, 1))
        lstm_pred = scaler.inverse_transform(lstm_pred)[0][0]
        
        # ARIMA prediction
        model_arima = sm.tsa.ARIMA(df['close'], order=(5,1,0))
        results_arima = model_arima.fit()
        arima_pred = results_arima.forecast(steps=1)[0]
        
        # Combine predictions
        ensemble_pred = (lstm_pred + arima_pred) / 2
        
        return {
            'lstm': lstm_pred,
            'arima': arima_pred,
            'ensemble': ensemble_pred,
            'confidence': min(1.0, 1.0 - abs(lstm_pred - arima_pred) / df['close'].iloc[-1])
        }
    except Exception as e:
        logger.error(f"Error in price prediction: {str(e)}")
        return {}

def get_market_data(coin_id: str) -> dict:
    """Get comprehensive market data"""
    try:
        # Get historical klines/candlestick data
        symbol = f"{coin_id}USDT"
        klines = binance_client.get_historical_klines(
            symbol,
            Client.KLINE_INTERVAL_1HOUR,
            "30 days ago UTC"
        )
        
        if not klines:
            return f"Error: No data available for {coin_id}"
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert values to float
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        # Get current market data
        ticker = binance_client.get_ticker(symbol=symbol)
        if not ticker:
            return f"Error: Could not get current price for {coin_id}"
        
        # Get advanced indicators
        indicators = get_advanced_technical_indicators(df)
        
        # Get market sentiment
        sentiment = get_market_sentiment(coin_id)
        
        # Get price predictions
        predictions = predict_price(df)
        
        return {
            'current_price': float(ticker['lastPrice']),
            '24h_change': float(ticker['priceChangePercent']),
            '7d_change': (float(ticker['lastPrice']) - float(df['close'].iloc[-168])) / float(df['close'].iloc[-168]) * 100 if len(df) >= 168 else 0,
            'market_cap': float(ticker['quoteVolume']),
            'total_volume': float(ticker['volume']),
            'indicators': indicators,
            'sentiment': sentiment,
            'predictions': predictions
        }
    except BinanceAPIException as e:
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in get_market_data: {str(e)}")
        return f"Error: {str(e)}"

def get_news_and_sentiment(symbol: str) -> str:
    """Get recent trades and market activity as a proxy for news/sentiment"""
    try:
        # Convert symbol to Binance format
        symbol = symbol.upper()
        if not symbol.endswith('USDT'):
            symbol = f"{symbol}USDT"

        # Get recent trades
        trades = binance_client.get_recent_trades(symbol=symbol, limit=100)
        
        # Get 24h ticker
        ticker = binance_client.get_ticker(symbol=symbol)
        
        # Calculate basic sentiment metrics
        buy_volume = sum(float(trade['qty']) for trade in trades if trade['isBuyerMaker'])
        sell_volume = sum(float(trade['qty']) for trade in trades if not trade['isBuyerMaker'])
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            buy_percentage = (buy_volume / total_volume) * 100
        else:
            buy_percentage = 50
            
        sentiment = "Very Positive" if buy_percentage > 70 else \
                   "Positive" if buy_percentage > 55 else \
                   "Neutral" if buy_percentage > 45 else \
                   "Negative" if buy_percentage > 30 else \
                   "Very Negative"
        
        return f"""Market Activity and Sentiment Analysis:
- Recent Trading Activity: {len(trades)} trades in recent period
- Buy vs Sell Ratio: {buy_percentage:.2f}% buys
- Overall Sentiment: {sentiment}
- 24h Price Change: {ticker['priceChangePercent']}%
- 24h Volume: {ticker['volume']} {symbol[:-4]}"""

    except BinanceAPIException as e:
        return f"Error fetching market activity: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def clean_json_response(response: str) -> str:
    """Clean and extract JSON from the AI response"""
    # Remove any text before the first '{'
    start_idx = response.find('{')
    if start_idx == -1:
        raise ValueError("No JSON object found in response")
    
    # Find the matching closing brace
    brace_count = 0
    end_idx = -1
    
    for i in range(start_idx, len(response)):
        if response[i] == '{':
            brace_count += 1
        elif response[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break
    
    if end_idx == -1:
        raise ValueError("Invalid JSON structure")
    
    # Extract and clean the JSON
    json_str = response[start_idx:end_idx].strip()
    return json_str

@app.route('/analyze', methods=['POST'])
def analyze_market():
    try:
        # Check if Binance client is available
        if binance_client is None:
            return jsonify({
                'error': 'Binance API is currently unavailable. Please try again later.'
            }), 503

        data = request.get_json()
        
        # Get coin_id from request
        coin_id = data.get('coin_id', '').upper()
        if not coin_id:
            return jsonify({'error': 'coin_id is required'}), 400
            
        try:
            # Get all trading symbols from Binance
            exchange_info = binance_client.get_exchange_info()
            valid_symbols = [s['baseAsset'] for s in exchange_info['symbols'] if s['quoteAsset'] == 'USDT']
            
            # Check if the coin exists on Binance
            if coin_id not in valid_symbols:
                return jsonify({
                    'error': f'Coin {coin_id} not found on Binance. Available coins: {", ".join(valid_symbols[:10])}...'
                }), 404
        
        except BinanceAPIException as e:
            return jsonify({
                'error': f'Failed to validate coin: {str(e)}'
            }), 503
        
        try:
            # Gather all data with timeout handling
            market_data = get_market_data(coin_id)
            if isinstance(market_data, str) and "Error" in market_data:
                return jsonify({'error': market_data}), 400
                
            news = get_news_and_sentiment(coin_id)
            if isinstance(news, str) and "Error" in news:
                news = "Market sentiment data temporarily unavailable"
        
        except (BinanceAPIException, RequestException, Timeout) as e:
            return jsonify({
                'error': f'Failed to fetch data from Binance: {str(e)}'
            }), 503

        # Create the analysis prompt
        analysis_prompt = f"""You are an expert cryptocurrency analyst with deep knowledge of technical analysis, market fundamentals, and sentiment analysis.
Analyze the following comprehensive data for {coin_id} and provide a detailed trading signal and analysis:

MARKET DATA:
- Current Price: ${market_data['current_price']:,.2f}
- 24h Change: {market_data['24h_change']:.2f}%
- 7d Change: {market_data['7d_change']:.2f}%
- Market Cap: ${market_data['market_cap']:,.2f}
- Trading Volume: ${market_data['total_volume']:,.2f}

TECHNICAL ANALYSIS:
1. Momentum Indicators:
   - RSI (14): {market_data['indicators']['momentum']['rsi']:.2f}
   - Stochastic: {market_data['indicators']['momentum']['stoch']:.2f}
   - StochRSI: {market_data['indicators']['momentum']['stoch_rsi']:.2f}
   - MACD: {market_data['indicators']['momentum']['macd']:.2f}

2. Volume Indicators:
   - Money Flow Index: {market_data['indicators']['volume']['mfi']:.2f}
   - Accumulation/Distribution Index: {market_data['indicators']['volume']['adi']:.2f}
   - On-Balance Volume: {market_data['indicators']['volume']['obv']:.2f}

3. Trend Indicators:
   - ADX: {market_data['indicators']['trend']['adx']:.2f}
   - CCI: {market_data['indicators']['trend']['cci']:.2f}
   - DPO: {market_data['indicators']['trend']['dpo']:.2f}

4. Volatility Indicators:
   - ATR: {market_data['indicators']['volatility']['atr']:.2f}
   - Bollinger Bands:
     * Upper: ${market_data['indicators']['volatility']['bb']['upper']:,.2f}
     * Middle: ${market_data['indicators']['volatility']['bb']['middle']:,.2f}
     * Lower: ${market_data['indicators']['volatility']['bb']['lower']:,.2f}
   - Keltner Channels:
     * High: ${market_data['indicators']['volatility']['kc']['high']:,.2f}
     * Mid: ${market_data['indicators']['volatility']['kc']['mid']:,.2f}
     * Low: ${market_data['indicators']['volatility']['kc']['low']:,.2f}

MARKET SENTIMENT:
1. Fear & Greed Index:
   - Value: {market_data['sentiment'].get('fear_greed', {}).get('value', 'N/A')}
   - Classification: {market_data['sentiment'].get('fear_greed', {}).get('classification', 'N/A')}

2. Social Sentiment:
   - Twitter Sentiment: {market_data['sentiment'].get('social', {}).get('twitter', 'N/A')}
   - Reddit Sentiment: {market_data['sentiment'].get('social', {}).get('reddit', 'N/A')}

3. Google Trends:
   - Current Interest: {market_data['sentiment'].get('google_trends', {}).get('current', 'N/A')}
   - Trend Direction: {market_data['sentiment'].get('google_trends', {}).get('trend', 'N/A')}

PRICE PREDICTIONS:
- LSTM Model: ${market_data['predictions'].get('lstm', 0):,.2f}
- ARIMA Model: ${market_data['predictions'].get('arima', 0):,.2f}
- Ensemble Prediction: ${market_data['predictions'].get('ensemble', 0):,.2f}
- Prediction Confidence: {market_data['predictions'].get('confidence', 0):.2%}

RECENT MARKET ACTIVITY:
{news}

Based on this comprehensive analysis, provide a detailed JSON response in this exact format:
{{
    "signal": "STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL",
    "confidence_score": <0.0-1.0>,
    "price_prediction": {{
        "24h": <predicted_price>,
        "7d": <predicted_price>
    }},
    "risk_level": "LOW/MEDIUM/HIGH",
    "technical_analysis": {{
        "momentum": "Analysis of momentum indicators",
        "volume": "Analysis of volume indicators",
        "trend": "Analysis of trend indicators",
        "volatility": "Analysis of volatility indicators"
    }},
    "fundamental_analysis": "Analysis of market data and fundamentals",
    "sentiment_analysis": {{
        "market_sentiment": "Analysis of fear & greed index",
        "social_sentiment": "Analysis of social media sentiment",
        "trend_analysis": "Analysis of Google Trends data"
    }},
    "key_takeaways": ["List", "of", "key", "points"],
    "risk_factors": ["List", "of", "potential", "risks"],
    "trading_strategy": {{
        "entry_points": ["List", "of", "entry", "points"],
        "exit_points": ["List", "of", "exit", "points"],
        "stop_loss": <stop_loss_price>,
        "take_profit": <take_profit_price>
    }}
}}

IMPORTANT: Return ONLY the JSON object above, with no additional text or formatting."""

        try:
            # Get analysis from Gemini with timeout handling
            response = llm.invoke(analysis_prompt)
            
            # Clean and parse the JSON response
            analysis_text = clean_json_response(response.content)
            analysis_json = json.loads(analysis_text)
            
            return jsonify({
                'symbol': coin_id,
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis_json
            })
        
        except (json.JSONDecodeError, ValueError) as e:
            return jsonify({
                'error': 'Failed to parse AI response as JSON',
                'raw_response': response.content if 'response' in locals() else None,
                'error_details': str(e)
            }), 500
            
        except Exception as e:
            return jsonify({
                'error': f'AI analysis failed: {str(e)}'
            }), 500
    
    except Exception as e:
        return jsonify({
            'error': f'Unexpected error: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)