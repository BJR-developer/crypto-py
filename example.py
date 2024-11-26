import requests
import json

def print_formatted_analysis(analysis):
    print("\n=== Cryptocurrency Analysis ===")
    print(f"Coin: {analysis['coin_id']}")
    print(f"Current Price: ${analysis['current_price']:,.2f}")
    print(f"24h Price Change: {analysis['price_change_24h']:.2f}%")
    print(f"Market Cap: ${analysis['market_cap']:,.2f}")
    print("\nAI Analysis:")
    print(analysis['analysis'])
    print("\nPrice Prediction:")
    print(analysis['prediction'])

def get_supported_coins():
    """Get list of supported cryptocurrencies"""
    response = requests.get("http://localhost:8001/supported_coins")
    if response.status_code == 200:
        return response.json()["supported_coins"]
    else:
        print(f"Error: {response.status_code}")
        return None

def analyze_cryptocurrency(coin_id):
    """Analyze a specific cryptocurrency"""
    url = "http://localhost:8001/analyze"
    payload = {"coin_id": coin_id}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error analyzing {coin_id}: {str(e)}")
        return None

def main():
    # Example 1: Get list of supported coins
    print("Getting list of supported coins...")
    coins = get_supported_coins()
    if coins:
        print(f"Number of supported coins: {len(coins)}")
        print("First 5 coins:", json.dumps(coins[:5], indent=2))
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Analyze Bitcoin
    print("Analyzing Bitcoin...")
    btc_analysis = analyze_cryptocurrency("bitcoin")
    if btc_analysis:
        print_formatted_analysis(btc_analysis)
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Analyze Ethereum
    print("Analyzing Ethereum...")
    eth_analysis = analyze_cryptocurrency("ethereum")
    if eth_analysis:
        print_formatted_analysis(eth_analysis)

if __name__ == "__main__":
    main()
