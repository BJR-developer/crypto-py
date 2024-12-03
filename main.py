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

def get_market_data(symbol: str) -> dict:
    """Get comprehensive market data for a cryptocurrency"""
    try:
        # Convert symbol to Binance format (e.g., 'bitcoin' to 'BTCUSDT')
        symbol = symbol.upper()
        if not symbol.endswith('USDT'):
            symbol = f"{symbol}USDT"

        # Get ticker data
        ticker = binance_client.get_ticker(symbol=symbol)
        
        # Get 24h stats
        stats = binance_client.get_ticker(symbol=symbol)
        
        # Get historical klines (candlesticks) for 7d change
        klines = binance_client.get_historical_klines(
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_1DAY,
            start_str="7 days ago UTC"
        )
        
        # Calculate 7d change
        if len(klines) >= 7:
            seven_day_change = ((float(ticker['lastPrice']) - float(klines[0][4])) / float(klines[0][4])) * 100
        else:
            seven_day_change = 0
            
        return {
            'current_price': float(ticker['lastPrice']),
            '24h_change': float(ticker['priceChangePercent']),
            '7d_change': seven_day_change,
            'market_cap': float(ticker['quoteVolume']),  # Using quote volume as proxy
            'total_volume': float(ticker['volume']),
            'market_cap_rank': None,  # Not available in Binance
            'ath': None,  # Not available in Binance
            'ath_change_percentage': None  # Not available in Binance
        }
    except BinanceAPIException as e:
        return f"Error fetching market data: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def get_technical_indicators(symbol: str) -> dict:
    """Calculate technical indicators for a cryptocurrency"""
    try:
        # Convert symbol to Binance format
        symbol = symbol.upper()
        if not symbol.endswith('USDT'):
            symbol = f"{symbol}USDT"

        # Get historical klines for calculations
        klines = binance_client.get_historical_klines(
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_1HOUR,
            start_str="30 days ago UTC"
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignored'
        ])
        
        # Convert price columns to float
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        
        # Calculate indicators using the ta library
        # RSI
        rsi = ta.momentum.RSIIndicator(df['close'], window=14)
        current_rsi = rsi.rsi().iloc[-1]
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        current_macd = macd.macd().iloc[-1]
        current_signal = macd.macd_signal().iloc[-1]
        
        # SMA
        sma_20 = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator().iloc[-1]
        sma_50 = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator().iloc[-1]
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        bb_upper = bollinger.bollinger_hband().iloc[-1]
        bb_middle = bollinger.bollinger_mavg().iloc[-1]
        bb_lower = bollinger.bollinger_lband().iloc[-1]
        
        # Volatility (using standard deviation of returns)
        returns = df['close'].pct_change()
        volatility = returns.std() * 100  # Convert to percentage
        
        # Determine trend
        trend = "BULLISH" if sma_20 > sma_50 else "BEARISH"
        
        return {
            'rsi': current_rsi,
            'macd': current_macd,
            'signal_line': current_signal,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'bollinger_upper': bb_upper,
            'bollinger_middle': bb_middle,
            'bollinger_lower': bb_lower,
            'volatility': volatility,
            'trend': trend
        }
    except BinanceAPIException as e:
        return f"Error calculating technical indicators: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

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
        symbol = data.get('symbol', 'BTC').upper()  # Default to BTC if not specified
        
        try:
            # Gather all data with timeout handling
            market_data = get_market_data(symbol)
            if isinstance(market_data, str) and "Error" in market_data:
                return jsonify({'error': market_data}), 400
                
            technical_data = get_technical_indicators(symbol)
            if isinstance(technical_data, str) and "Error" in technical_data:
                return jsonify({'error': technical_data}), 400
                
            news = get_news_and_sentiment(symbol)
            if isinstance(news, str) and "Error" in news:
                news = "Market sentiment data temporarily unavailable"
        
        except (BinanceAPIException, RequestException, Timeout) as e:
            return jsonify({
                'error': f'Failed to fetch data from Binance: {str(e)}'
            }), 503
        
        # Create the analysis prompt
        analysis_prompt = f"""You are an expert cryptocurrency analyst with deep knowledge of technical analysis, market fundamentals, and sentiment analysis.
Your task is to analyze the data and provide a JSON response with your analysis.

Analyze the following data for {symbol} and provide a detailed trading signal and analysis:

MARKET DATA:
- Current Price: ${market_data['current_price']:,.2f}
- 24h Change: {market_data['24h_change']:.2f}%
- 7d Change: {market_data['7d_change']:.2f}%
- Market Cap: ${market_data['market_cap']:,.2f}
- Trading Volume: ${market_data['total_volume']:,.2f}

TECHNICAL INDICATORS:
- RSI (14): {technical_data['rsi']:.2f}
- MACD: {technical_data['macd']:.2f}
- Signal Line: {technical_data['signal_line']:.2f}
- SMA 20: ${technical_data['sma_20']:,.2f}
- SMA 50: ${technical_data['sma_50']:,.2f}
- Volatility: {technical_data['volatility']:.2f}%
- Overall Trend: {technical_data['trend']}
- Bollinger Bands:
  * Upper: ${technical_data['bollinger_upper']:,.2f}
  * Middle: ${technical_data['bollinger_middle']:,.2f}
  * Lower: ${technical_data['bollinger_lower']:,.2f}

RECENT MARKET ACTIVITY:
{news}

Return ONLY a JSON object in this exact format (no other text):
{{
    "signal": "STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL",
    "confidence_score": <0.0-1.0>,
    "price_prediction": {{
        "24h": <predicted_price>,
        "7d": <predicted_price>
    }},
    "risk_level": "LOW/MEDIUM/HIGH",
    "technical_analysis": "Detailed technical analysis with key points",
    "fundamental_analysis": "Analysis of market data and fundamentals",
    "sentiment_analysis": "Analysis of market activity and sentiment",
    "key_takeaways": ["List", "of", "key", "points"],
    "risk_factors": ["List", "of", "potential", "risks"],
    "recommendation": "Detailed trading recommendation and strategy"
}}

IMPORTANT: Return ONLY the JSON object above, with no additional text or formatting."""

        try:
            # Get analysis from Gemini with timeout handling
            response = llm.invoke(analysis_prompt)
            
            # Clean and parse the JSON response
            analysis_text = clean_json_response(response.content)
            analysis_json = json.loads(analysis_text)
            
            return jsonify({
                'symbol': symbol,
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