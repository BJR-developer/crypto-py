from flask import Flask, jsonify, request
from langchain_google_genai import ChatGoogleGenerativeAI
from pycoingecko import CoinGeckoAPI
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import json
from functools import lru_cache

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize APIs
cg = CoinGeckoAPI()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.7,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Cache market data for 5 minutes
@lru_cache(maxsize=100)
def get_cached_market_data(coin_id: str, timestamp: str) -> dict:
    return get_market_data(coin_id)

# Cache technical indicators for 5 minutes
@lru_cache(maxsize=100)
def get_cached_technical_indicators(coin_id: str, timestamp: str) -> dict:
    return get_technical_indicators(coin_id)

# Cache news for 30 minutes
@lru_cache(maxsize=100)
def get_cached_news(coin_id: str, timestamp: str) -> str:
    return get_news_and_sentiment(coin_id)

def get_market_data(coin_id: str) -> dict:
    """Get comprehensive market data for a cryptocurrency"""
    try:
        # Get current market data
        coin_data = cg.get_coin_by_id(coin_id)
        market_data = coin_data["market_data"]
        
        return {
            "current_price": market_data["current_price"]["usd"],
            "24h_change": market_data["price_change_percentage_24h"],
            "7d_change": market_data["price_change_percentage_7d"],
            "market_cap": market_data["market_cap"]["usd"],
            "total_volume": market_data["total_volume"]["usd"],
            "ath": market_data["ath"]["usd"],
            "ath_change_percentage": market_data["ath_change_percentage"]["usd"],
            "market_cap_rank": coin_data["market_cap_rank"]
        }
    except Exception as e:
        return f"Error fetching market data: {str(e)}"

def get_technical_indicators(coin_id: str) -> dict:
    """Calculate technical indicators for a cryptocurrency"""
    try:
        # Get historical price data
        data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=30)
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        
        # Calculate various technical indicators
        # SMA
        df['SMA_20'] = df['price'].rolling(window=20).mean()
        df['SMA_50'] = df['price'].rolling(window=50).mean()
        
        # EMA
        df['EMA_12'] = df['price'].ewm(span=12).mean()
        df['EMA_26'] = df['price'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
        
        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['price'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['price'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['price'].rolling(window=20).std()
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Calculate volatility
        volatility = df['price'].pct_change().std() * np.sqrt(365) * 100
        
        return {
            "sma_20": latest['SMA_20'],
            "sma_50": latest['SMA_50'],
            "rsi": latest['RSI'],
            "macd": latest['MACD'],
            "signal_line": latest['Signal_Line'],
            "bollinger_upper": latest['BB_upper'],
            "bollinger_middle": latest['BB_middle'],
            "bollinger_lower": latest['BB_lower'],
            "volatility": volatility,
            "trend": "BULLISH" if latest['SMA_20'] > latest['SMA_50'] else "BEARISH"
        }
    except Exception as e:
        return f"Error calculating technical indicators: {str(e)}"

def get_news_and_sentiment(coin_id: str) -> str:
    """Get latest news and sentiment for a cryptocurrency"""
    try:
        # Get coin data to get the proper name
        coin_data = cg.get_coin_by_id(coin_id)
        coin_name = coin_data["name"]
        
        # Get news from CoinGecko
        status_updates = coin_data.get("status_updates", [])
        description = coin_data.get("description", {}).get("en", "")
        sentiment = coin_data.get("sentiment_votes_up_percentage", 0)
        
        # Get additional market data
        market_data = coin_data["market_data"]
        
        news_summary = f"""
Market Sentiment: {sentiment}% positive
Developer Activity: Active
Community Score: {coin_data.get('community_score', 0)}
Market Sentiment: {coin_data.get('sentiment_votes_up_percentage', 0)}%

Project Description:
{description[:500]}...

Recent Updates:
"""
        
        # Add recent status updates if available
        if status_updates:
            for update in status_updates[:3]:
                news_summary += f"- {update.get('description', '')}\n"
        
        return news_summary
    except Exception as e:
        return f"Error fetching news: {str(e)}"

@app.route('/analyze', methods=['POST'])
def analyze_market():
    try:
        data = request.get_json()
        coin_id = data.get('coin_id', 'bitcoin')
        
        # Create timestamp for cache (updates every 5 minutes)
        cache_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        cache_timestamp_5min = cache_timestamp[:-1] + "0"  # Round to nearest 5 minutes
        cache_timestamp_30min = cache_timestamp[:-2] + "00"  # Round to nearest 30 minutes
        
        # Gather all data using cached functions
        market_data = get_cached_market_data(coin_id, cache_timestamp_5min)
        technical_data = get_cached_technical_indicators(coin_id, cache_timestamp_5min)
        news = get_cached_news(coin_id, cache_timestamp_30min)
        
        # Get coin name
        coin_data = cg.get_coin_by_id(coin_id)
        coin_name = coin_data["name"]
        
        # Create the analysis prompt
        analysis_prompt = f"""You are an expert cryptocurrency analyst with deep knowledge of technical analysis, market fundamentals, and sentiment analysis.
Your task is to analyze the data and provide a JSON response with your analysis.

Analyze the following data for {coin_name} and provide a detailed trading signal and analysis:

MARKET DATA:
- Current Price: ${market_data['current_price']:,.2f}
- 24h Change: {market_data['24h_change']:.2f}%
- 7d Change: {market_data['7d_change']:.2f}%
- Market Cap: ${market_data['market_cap']:,.2f}
- Trading Volume: ${market_data['total_volume']:,.2f}
- Market Cap Rank: #{market_data['market_cap_rank']}
- ATH: ${market_data['ath']:,.2f} ({market_data['ath_change_percentage']:.2f}% from ATH)

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

RECENT NEWS AND SENTIMENT:
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
    "sentiment_analysis": "Analysis of news and market sentiment",
    "key_takeaways": ["List", "of", "key", "points"],
    "risk_factors": ["List", "of", "potential", "risks"],
    "recommendation": "Detailed trading recommendation and strategy"
}}

IMPORTANT: Do not include any markdown formatting or code blocks in your response. Return only the raw JSON object."""

        # Get analysis from Gemini
        response = llm.invoke(analysis_prompt)
        
        # Extract content and remove markdown formatting
        analysis_text = response.content
        if "```json" in analysis_text:
            analysis_text = analysis_text.split("```json")[1].split("```")[0].strip()
        elif "```" in analysis_text:
            analysis_text = analysis_text.split("```")[1].strip()
            
        # Parse JSON
        analysis_json = json.loads(analysis_text)
        
        return jsonify({
            'coin_id': coin_id,
            'coin_name': coin_name,
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis_json
        })
    
    except json.JSONDecodeError as e:
        return jsonify({
            'error': 'Failed to parse AI response as JSON',
            'raw_response': analysis_text if 'analysis_text' in locals() else None,
            'error_details': str(e)
        }), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)