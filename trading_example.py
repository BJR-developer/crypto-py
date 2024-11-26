import requests
import json
from datetime import datetime

def print_trading_signal(signal):
    print("\n=== Cryptocurrency Trading Signal ===")
    print(f"Coin: {signal['coin_id'].upper()}")
    print(f"Signal: {signal['signal']}")
    print(f"Confidence: {signal['confidence']*100:.1f}%")
    print(f"\nPrice Information:")
    print(f"Current Price: ${signal['current_price']:,.2f}")
    print(f"24h Price Change: {signal['price_change_24h']:.2f}%")
    print(f"\nTechnical Indicators:")
    print(f"RSI (14): {signal['rsi_value']:.2f}")
    print(f"MACD Signal: {signal['macd_signal']}")
    print(f"\nTrading Levels:")
    print(f"Next Target: ${signal['next_target']:,.2f}")
    print(f"Stop Loss: ${signal['stop_loss']:,.2f}")
    print(f"\nAnalysis:")
    print(signal['analysis'])

def get_trading_signal(coin_id):
    """Get trading signal for a specific cryptocurrency"""
    url = f"http://localhost:8001/trading_signal/{coin_id}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"Error {response.status_code}"
            try:
                error_detail = response.json()
                if 'detail' in error_detail:
                    error_msg += f": {error_detail['detail']}"
            except:
                error_msg += f": {response.text}"
            print(f"\nAPI Error for {coin_id}: {error_msg}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"\nConnection Error for {coin_id}: {str(e)}")
        return None

def main():
    # Example coins to analyze
    coins = ["ethereum"]
    
    print(f"Analyzing trading signals at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    for coin in coins:
        print(f"\nGetting trading signal for {coin.upper()}...")
        signal = get_trading_signal(coin)
        if signal:
            print_trading_signal(signal)
            print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
