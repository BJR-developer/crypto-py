from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import google.generativeai as genai
from pycoingecko import CoinGeckoAPI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Crypto Trading Signals API",
    description="API for cryptocurrency analysis and trading signals using technical analysis and AI",
    version="1.0.0"
)

# Initialize CoinGecko API client
cg = CoinGeckoAPI()

# Initialize Google Generative AI
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

genai.configure(api_key=GOOGLE_API_KEY)
model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

class CryptoRequest(BaseModel):
    coin_id: str
    days: Optional[int] = 30

class CryptoSignal(BaseModel):
    coin_id: str
    signal: str
    confidence: float
    current_price: float
    price_change_24h: float
    rsi_value: float
    macd_signal: str
    analysis: str
    next_target: float
    stop_loss: float

def calculate_technical_indicators(prices):
    df = pd.DataFrame(prices)
    
    # Calculate RSI
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = df['price'].ewm(span=12, adjust=False).mean()
    exp2 = df['price'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=9, adjust=False).mean()
    
    return {
        'rsi': rsi.iloc[-1],
        'macd': macd.iloc[-1],
        'signal_line': signal_line.iloc[-1]
    }

def get_market_sentiment_prompt(market_data, technical_indicators):
    return f"""
    Analyze the following cryptocurrency data and provide a trading signal:
    
    Technical Indicators:
    - RSI: {technical_indicators['rsi']:.2f}
    - MACD: {technical_indicators['macd']:.2f}
    - Signal Line: {technical_indicators['signal_line']:.2f}
    
    Market Data:
    - Current Price: ${market_data['current_price']['usd']}
    - 24h Change: {market_data['price_change_percentage_24h']}%
    - 7d Change: {market_data['price_change_percentage_7d']}%
    - Market Cap: ${market_data['market_cap']['usd']}
    - Volume: ${market_data['total_volume']['usd']}
    
    Based on this data:
    1. Determine if the signal is BULLISH or BEARISH
    2. Provide a confidence score between 0 and 1
    3. Suggest a reasonable next price target
    4. Recommend a stop-loss level
    5. Give a brief analysis explaining the signal
    
    Format your response exactly like this example:
    SIGNAL: BULLISH
    CONFIDENCE: 0.85
    TARGET: 45000
    STOP_LOSS: 38000
    ANALYSIS: The asset shows strong bullish momentum with RSI...
    """

@app.get("/")
async def root():
    return {"message": "Welcome to the Crypto Trading Signals API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/trading_signal/{coin_id}")
async def get_trading_signal(coin_id: str, days: int = 30):
    try:
        # Fetch historical price data
        try:
            price_data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching price data from CoinGecko: {str(e)}"
            )
        
        try:
            prices_df = pd.DataFrame(price_data['prices'], columns=['timestamp', 'price'])
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing price data: {str(e)}"
            )
        
        # Get current market data
        try:
            coin_data = cg.get_coin_by_id(
                id=coin_id,
                localization=False,
                tickers=False,
                market_data=True,
                community_data=False,
                developer_data=False,
                sparkline=False
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching coin data from CoinGecko: {str(e)}"
            )
        
        market_data = coin_data['market_data']
        
        # Calculate technical indicators
        try:
            technical_indicators = calculate_technical_indicators(prices_df)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error calculating technical indicators: {str(e)}"
            )
        
        # Create analysis prompt
        sentiment_prompt = PromptTemplate(
            input_variables=["data"],
            template="Analyze this cryptocurrency data and provide trading signals: {data}"
        )
        
        # Create chain and get analysis
        try:
            sentiment_chain = LLMChain(llm=model, prompt=sentiment_prompt)
            analysis_result = sentiment_chain.run(data=get_market_sentiment_prompt(market_data, technical_indicators))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating AI analysis: {str(e)}"
            )
        
        # Parse AI response
        try:
            lines = analysis_result.strip().split('\n')
            signal_data = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    signal_data[key.strip()] = value.strip()
            
            return CryptoSignal(
                coin_id=coin_id,
                signal=signal_data.get('SIGNAL', 'NEUTRAL'),
                confidence=float(signal_data.get('CONFIDENCE', '0.5')),
                current_price=market_data['current_price']['usd'],
                price_change_24h=market_data['price_change_percentage_24h'],
                rsi_value=technical_indicators['rsi'],
                macd_signal="BULLISH" if technical_indicators['macd'] > technical_indicators['signal_line'] else "BEARISH",
                analysis=signal_data.get('ANALYSIS', 'No analysis available'),
                next_target=float(signal_data.get('TARGET', market_data['current_price']['usd'])),
                stop_loss=float(signal_data.get('STOP_LOSS', market_data['current_price']['usd'] * 0.95))
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error parsing analysis results: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/supported_coins")
async def get_supported_coins():
    try:
        coins_list = cg.get_coins_list()
        return {"supported_coins": coins_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app = CORSMiddleware(
    app=app,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)

if __name__ == "__main__" and not os.getenv("VERCEL"):
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
