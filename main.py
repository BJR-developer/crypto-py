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
        model="gpt-4-turbo-preview",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

def get_market_sentiment_prompt(market_data: Dict[str, Any], technical_indicators: Dict[str, float]) -> str:
    """Generate prompt for AI analysis based on market data"""
    return f"""You are a professional cryptocurrency analyst. Analyze the following market data and provide a trading signal.

Market Data:
- Current Price: ${market_data['current_price']['usd']}
- 24h Price Change: {market_data['price_change_percentage_24h']}%
- Market Cap: ${market_data['market_cap']['usd']}
- 24h Trading Volume: ${market_data['total_volume']['usd']}

Technical Indicators:
- RSI (14): {technical_indicators['rsi']:.2f}
- Trend: {technical_indicators['trend']}

Based on this data, provide your analysis in this EXACT JSON format:
{{
    "signal": "BULLISH",
    "confidence": 0.8,
    "target": {market_data['current_price']['usd'] * 1.05},
    "stop_loss": {market_data['current_price']['usd'] * 0.95},
    "analysis": "Brief analysis of the market conditions and trading recommendation."
}}

IMPORTANT: Return ONLY the JSON object, exactly as shown above. No other text."""

def calculate_technical_indicators(prices):
    """Calculate simple technical indicators from price data"""
    if len(prices) < 15:  # Need at least 15 data points
        return {'rsi': 50, 'trend': 'NEUTRAL'}
        
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    
    # Simple RSI calculation
    changes = df['price'].diff()
    gains = changes.where(changes > 0, 0)
    losses = -changes.where(changes < 0, 0)
    
    # Simple moving averages of gains and losses
    avg_gain = gains.rolling(window=14, min_periods=1).mean()
    avg_loss = losses.rolling(window=14, min_periods=1).mean()
    
    # Safe RSI calculation
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Handle potential NaN values
    final_rsi = float(rsi.iloc[-1])
    if pd.isna(final_rsi):
        final_rsi = 50  # Neutral RSI if calculation fails
        
    return {
        'rsi': min(100, max(0, final_rsi)),  # Ensure RSI is between 0-100
        'trend': 'BULLISH' if df['price'].iloc[-1] > df['price'].iloc[-5] else 'BEARISH'
    }

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

class TopCoin(BaseModel):
    coin_id: str = Field(..., description="The ID of the cryptocurrency")
    name: str = Field(..., description="The name of the cryptocurrency")
    current_price: float = Field(..., gt=0, description="Current price in USD")
    price_change_24h: float = Field(..., description="24h price change percentage")
    volume_24h: float = Field(..., gt=0, description="24h trading volume")
    market_cap: float = Field(..., gt=0, description="Market capitalization")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="AI confidence score")
    analysis: str = Field(..., min_length=10, description="Trading analysis and rationale")
    target_price: float = Field(..., gt=0, description="Target price prediction")
    stop_loss: float = Field(..., gt=0, description="Recommended stop loss")

class GridTradingCoin(BaseModel):
    coin_id: str = Field(..., description="The ID of the cryptocurrency")
    name: str = Field(..., description="The name of the cryptocurrency")
    current_price: float = Field(..., gt=0, description="Current price in USD")
    volatility_24h: float = Field(..., description="24h price volatility")
    volume_24h: float = Field(..., gt=0, description="24h trading volume")
    market_cap: float = Field(..., gt=0, description="Market capitalization")
    grid_score: float = Field(..., ge=0.0, le=1.0, description="Suitability score for grid trading")
    analysis: str = Field(..., min_length=10, description="Analysis of suitability for grid trading")
    upper_price: float = Field(..., gt=0, description="Recommended upper grid price")
    lower_price: float = Field(..., gt=0, description="Recommended lower grid price")
    grid_levels: int = Field(..., ge=3, le=100, description="Recommended number of grid levels")

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
            "xrp": "ripple",
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
                macd_signal="BULLISH" if technical_indicators['trend'] == 'BULLISH' else "BEARISH",
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

@app.get("/best_grid_trading_coins", response_model=list[GridTradingCoin])
async def get_best_grid_trading_coins():
    """Find the top 3 stable coins suitable for grid trading"""
    try:
        # Get top 100 coins by market cap
        markets = cg.get_coins_markets(
            vs_currency='usd',
            order='market_cap_desc',
            per_page=100,
            sparkline=False
        )
        
        # Filter for stable coins (low volatility, high volume)
        stable_coins = []
        for coin in markets:
            if coin['total_volume'] > 1000000:  # Minimum volume threshold
                try:
                    # Get volatility data
                    price_data = cg.get_coin_market_chart_by_id(
                        id=coin['id'],
                        vs_currency='usd',
                        days='7'
                    )
                    
                    prices = pd.DataFrame(price_data['prices'], columns=['timestamp', 'price'])
                    volatility = prices['price'].std() / prices['price'].mean()
                    
                    # Look for low volatility coins
                    if volatility < 0.05:  # 5% volatility threshold
                        stable_coins.append((coin, volatility))
                        
                except Exception:
                    continue
                    
                if len(stable_coins) >= 5:
                    break
        
        # Analyze potential grid trading coins
        results = []
        for coin, volatility in stable_coins:
            try:
                # Calculate grid parameters
                price = coin['current_price']
                upper_price = price * (1 + volatility * 2)
                lower_price = price * (1 - volatility * 2)
                grid_levels = min(max(int(1/volatility), 3), 100)
                
                # Calculate grid score based on volume and volatility
                volume_score = min(coin['total_volume'] / 1e9, 1)  # Normalize volume
                volatility_score = 1 - (volatility * 10)  # Lower volatility is better
                grid_score = (volume_score + volatility_score) / 2
                
                results.append(GridTradingCoin(
                    coin_id=coin['id'],
                    name=coin['name'],
                    current_price=coin['current_price'],
                    volatility_24h=volatility * 100,  # Convert to percentage
                    volume_24h=coin['total_volume'],
                    market_cap=coin['market_cap'],
                    grid_score=grid_score,
                    analysis=f"This coin shows stable price action with {volatility*100:.1f}% volatility and good trading volume. Suitable for grid trading with {grid_levels} levels.",
                    upper_price=upper_price,
                    lower_price=lower_price,
                    grid_levels=grid_levels
                ))
                
            except Exception:
                continue
                
            if len(results) >= 3:
                break
        
        # Sort by grid score
        results.sort(key=lambda x: x.grid_score, reverse=True)
        return results[:3]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error finding grid trading coins: {str(e)}"
        )

if __name__ == "__main__" and not os.getenv("VERCEL"):
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
