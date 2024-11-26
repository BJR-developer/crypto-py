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

@app.get("/best_bearish_coins", response_model=list[TopCoin])
async def get_best_bearish_coins():
    """Find the top 3 coins with strongest bearish signals"""
    try:
        # Get top 100 coins by market cap
        markets = cg.get_coins_markets(
            vs_currency='usd',
            order='market_cap_desc',
            per_page=100,
            sparkline=False
        )
        
        # Filter for coins with negative price change
        bearish_coins = [
            coin for coin in markets 
            if coin['price_change_24h'] is not None and coin['price_change_24h'] < 0
        ]
        
        # Sort by price change (most negative first)
        bearish_coins.sort(key=lambda x: x['price_change_24h'])
        
        # Analyze top 5 most bearish coins
        results = []
        for coin in bearish_coins[:5]:
            try:
                # Get technical indicators
                price_data = cg.get_coin_market_chart_by_id(
                    id=coin['id'],
                    vs_currency='usd',
                    days='1'
                )
                technical_indicators = calculate_technical_indicators(price_data['prices'])
                
                # Get AI analysis
                model = get_ai_model()
                prompt = PromptTemplate(
                    input_variables=["market_data"],
                    template="{market_data}"
                )
                
                sentiment_chain = LLMChain(llm=model, prompt=prompt)
                analysis_result = sentiment_chain.run(
                    market_data=get_market_sentiment_prompt(coin, technical_indicators)
                )
                
                analysis_json = json.loads(analysis_result)
                
                if analysis_json['signal'] == 'BEARISH':
                    results.append(TopCoin(
                        coin_id=coin['id'],
                        name=coin['name'],
                        current_price=coin['current_price'],
                        price_change_24h=coin['price_change_24h'],
                        volume_24h=coin['total_volume'],
                        market_cap=coin['market_cap'],
                        confidence_score=analysis_json['confidence'],
                        analysis=analysis_json['analysis'],
                        target_price=analysis_json['target'],
                        stop_loss=analysis_json['stop_loss']
                    ))
                
                if len(results) >= 3:
                    break
                    
            except Exception as e:
                continue
        
        # Sort by confidence score
        results.sort(key=lambda x: x.confidence_score, reverse=True)
        return results[:3]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error finding bearish coins: {str(e)}"
        )

@app.get("/best_bullish_coins", response_model=list[TopCoin])
async def get_best_bullish_coins():
    """Find the top 3 coins with potential bullish reversal in next 24 hours"""
    try:
        # Get top 100 coins by market cap
        markets = cg.get_coins_markets(
            vs_currency='usd',
            order='volume_desc',  # Sort by volume for potential momentum
            per_page=100,
            sparkline=False
        )
        
        # Filter for coins with high volume but price near support
        potential_coins = []
        for coin in markets:
            if coin['total_volume'] > 0:  # Ensure there's trading activity
                try:
                    # Get technical indicators
                    price_data = cg.get_coin_market_chart_by_id(
                        id=coin['id'],
                        vs_currency='usd',
                        days='1'
                    )
                    technical_indicators = calculate_technical_indicators(price_data['prices'])
                    
                    # Look for oversold conditions (RSI < 30)
                    if technical_indicators['rsi'] < 30:
                        potential_coins.append((coin, technical_indicators))
                        
                except Exception:
                    continue
                    
                if len(potential_coins) >= 5:
                    break
        
        # Analyze potential bullish coins
        results = []
        for coin, indicators in potential_coins:
            try:
                # Get AI analysis
                model = get_ai_model()
                prompt = PromptTemplate(
                    input_variables=["market_data"],
                    template="{market_data}"
                )
                
                sentiment_chain = LLMChain(llm=model, prompt=prompt)
                analysis_result = sentiment_chain.run(
                    market_data=get_market_sentiment_prompt(coin, indicators)
                )
                
                analysis_json = json.loads(analysis_result)
                
                if analysis_json['signal'] == 'BULLISH':
                    results.append(TopCoin(
                        coin_id=coin['id'],
                        name=coin['name'],
                        current_price=coin['current_price'],
                        price_change_24h=coin['price_change_24h'],
                        volume_24h=coin['total_volume'],
                        market_cap=coin['market_cap'],
                        confidence_score=analysis_json['confidence'],
                        analysis=analysis_json['analysis'],
                        target_price=analysis_json['target'],
                        stop_loss=analysis_json['stop_loss']
                    ))
                
            except Exception:
                continue
                
            if len(results) >= 3:
                break
        
        # Sort by confidence score
        results.sort(key=lambda x: x.confidence_score, reverse=True)
        return results[:3]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error finding bullish coins: {str(e)}"
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
