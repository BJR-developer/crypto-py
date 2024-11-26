from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
from pycoingecko import CoinGeckoAPI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
import json
import re

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Crypto Trading Signals API",
    description="API for cryptocurrency analysis and trading signals using technical analysis and AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize CoinGecko API client
cg = CoinGeckoAPI()

# Initialize OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

def get_ai_model():
    """Get the OpenAI model instance"""
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

def get_market_sentiment_prompt(market_data: Dict[str, Any], technical_indicators: Dict[str, float]) -> str:
    """Generate prompt for AI analysis based on market data"""
    return f"""You are a professional cryptocurrency analyst. Analyze the following market data and provide a trading signal response in a specific JSON format.

Market Data:
- Current Price: ${market_data['current_price']['usd']}
- 24h Price Change: {market_data['price_change_percentage_24h']}%
- Market Cap: ${market_data['market_cap']['usd']}
- 24h Trading Volume: ${market_data['total_volume']['usd']}

Technical Indicators:
- RSI (14): {technical_indicators['rsi']:.2f}
- MACD: {technical_indicators['macd']:.2f}
- Signal Line: {technical_indicators['signal_line']:.2f}

Provide your analysis in the following JSON format:
{{
    "signal": "BULLISH or BEARISH",
    "confidence": "number between 0.0 and 1.0",
    "target": "next price target in USD (number only)",
    "stop_loss": "recommended stop loss level in USD (number only)",
    "analysis": "2-3 sentences explaining the rationale"
}}

Ensure your response is ONLY the JSON object, with no additional text before or after."""

class CryptoRequest(BaseModel):
    coin_id: str = Field(..., description="The ID of the cryptocurrency (e.g., 'bitcoin')")
    days: Optional[int] = Field(default=30, description="Number of days of historical data to analyze")

class CryptoSignal(BaseModel):
    coin_id: str = Field(..., description="The ID of the cryptocurrency")
    signal: str = Field(..., description="The trading signal (BULLISH or BEARISH)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="The confidence level of the signal (0.0-1.0)")
    current_price: float = Field(..., gt=0, description="The current price of the cryptocurrency")
    price_change_24h: float = Field(..., description="The 24h price change percentage")
    rsi_value: float = Field(..., ge=0, le=100, description="The RSI value (0-100)")
    macd_signal: str = Field(..., description="The MACD signal (BULLISH or BEARISH)")
    analysis: str = Field(..., min_length=10, description="The detailed market analysis")
    next_target: float = Field(..., gt=0, description="The next price target")
    stop_loss: float = Field(..., gt=0, description="The recommended stop loss level")

def calculate_technical_indicators(prices):
    """Calculate technical indicators from price data"""
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    
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
        'rsi': float(rsi.iloc[-1]),
        'macd': float(macd.iloc[-1]),
        'signal_line': float(signal_line.iloc[-1])
    }

@app.get("/")
async def root():
    return {"message": "Welcome to the Crypto Trading Signals API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/trading_signal/{coin_id}", response_model=CryptoSignal)
async def get_trading_signal(coin_id: str, days: int = 30):
    try:
        # Normalize coin_id
        coin_id = coin_id.lower().strip()
        
        # Common mappings for popular coins
        coin_mappings = {
            "ripple": "xrp",
            "btc": "bitcoin",
            "eth": "ethereum",
            "doge": "dogecoin"
        }
        
        # Use mapping if available
        coin_id = coin_mappings.get(coin_id, coin_id)
        
        try:
            # Get market data
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
            if "could not find coin with the given id" in str(e).lower():
                raise HTTPException(
                    status_code=404,
                    detail=f"Cryptocurrency '{coin_id}' not found. Please check the coin ID and try again."
                )
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching market data: {str(e)}"
            )
        
        market_data = coin_data['market_data']
        
        # Get historical price data
        try:
            price_data = cg.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency='usd',
                days=str(days)
            )
            prices = price_data['prices']
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching historical price data: {str(e)}"
            )
        
        # Calculate technical indicators
        try:
            technical_indicators = calculate_technical_indicators(prices)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error calculating technical indicators: {str(e)}"
            )
        
        # Create analysis prompt
        try:
            model = get_ai_model()
            prompt = PromptTemplate(
                input_variables=["market_data"],
                template="{market_data}"
            )
            
            # Create chain and get analysis
            sentiment_chain = LLMChain(llm=model, prompt=prompt)
            analysis_result = sentiment_chain.run(market_data=get_market_sentiment_prompt(market_data, technical_indicators))
            
            # Parse the analysis result
            try:
                analysis_result = json.loads(analysis_result)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON response from AI model")
            
            if not all(key in analysis_result for key in ['signal', 'confidence', 'target', 'stop_loss', 'analysis']):
                raise ValueError("Invalid response from AI model. Missing required keys.")
            
            return CryptoSignal(
                coin_id=coin_id,
                signal=analysis_result['signal'].upper(),
                confidence=analysis_result['confidence'],
                current_price=market_data['current_price']['usd'],
                price_change_24h=market_data['price_change_percentage_24h'],
                rsi_value=technical_indicators['rsi'],
                macd_signal="BULLISH" if technical_indicators['macd'] > technical_indicators['signal_line'] else "BEARISH",
                analysis=analysis_result['analysis'],
                next_target=analysis_result['target'],
                stop_loss=analysis_result['stop_loss']
            )
            
        except ValueError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error parsing AI response: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating AI analysis: {str(e)}"
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
    """Get a list of supported cryptocurrencies"""
    try:
        coins_list = cg.get_coins_list()
        # Return only the most relevant information
        return {
            "supported_coins": [
                {
                    "id": coin["id"],
                    "symbol": coin["symbol"],
                    "name": coin["name"]
                }
                for coin in coins_list
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching supported coins: {str(e)}"
        )

if __name__ == "__main__" and not os.getenv("VERCEL"):
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
