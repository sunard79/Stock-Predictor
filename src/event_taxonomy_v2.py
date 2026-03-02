# Event Taxonomy for Stock Predictor
# This file defines the classification of financial news events and their impact on specific tickers.

EVENT_TAXONOMY = {
    "macroeconomics": {
        "inflation": ["CPI", "PPI", "inflation", "cost of living", "price index"],
        "employment": ["nonfarm payrolls", "unemployment rate", "jobless claims", "labor market"],
        "monetary_policy": ["Fed", "FOMC", "interest rates", "rate hike", "rate cut", "Jerome Powell"]
    },
    "regional_central_banks": {
        "rba_australia": ["RBA", "Reserve Bank Australia", "Philip Lowe", "Michele Bullock", "Australian interest rate", "cash rate"],
        "boj_japan": ["BOJ", "Bank of Japan", "yen", "Japanese monetary policy"],
        "ecb_europe": ["ECB", "European Central Bank", "euro", "Christine Lagarde"]
    },
    "regional_markets": {
        "australia": ["ASX", "Australian market", "Sydney exchange", "mining sector Australia", "iron ore", "coal exports", "China Australia trade"],
        "asia_pacific": ["Asia Pacific", "APAC markets", "emerging Asia", "regional trade"],
        "europe": ["European markets", "FTSE", "DAX", "Brexit", "EU policy"]
    },
    "sector_specific": {
        "technology": ["tech stocks", "NASDAQ", "semiconductor", "cloud computing", "AI stocks", "chip shortage"],
        "small_cap": ["small business", "SME", "Russell 2000", "small cap stocks"],
        "commodities": ["oil price", "crude oil", "gold price", "silver", "commodity markets"]
    }
}

# Mapping of subcategories to affected tickers
AFFECTED_ASSETS_MAPPING = {
    # Macro (General US)
    "inflation": ["SPY", "QQQ", "DIA", "IWM", "TLT"],
    "employment": ["SPY", "DIA", "IWM"],
    "monetary_policy": ["SPY", "QQQ", "DIA", "IWM", "TLT"],
    
    # Regional Central Banks
    "rba_australia": ["^AXJO", "EWA"],
    "boj_japan": ["EWJ"],
    "ecb_europe": ["EWU", "EWG"],
    
    # Regional Markets
    "australia": ["^AXJO", "EWA"],
    "asia_pacific": ["MCHI", "EWJ"],
    "europe": ["EWU", "EWG"],
    
    # Sector Specific
    "technology": ["QQQ"],
    "small_cap": ["IWM"],
    "commodities": ["USO", "GLD", "SLV"]
}

def get_affected_assets(event_category, subcategory):
    """
    Returns a list of tickers most affected by a specific event subcategory.
    
    Args:
        event_category (str): The broad category of the event.
        subcategory (str): The specific subcategory.
        
    Returns:
        list: Tickers (e.g., ['SPY', 'QQQ'])
    """
    # Using subcategory as it's more specific for mapping
    return AFFECTED_ASSETS_MAPPING.get(subcategory, ["SPY"])

def get_keywords(category, subcategory):
    """Returns keywords for a specific category/subcategory."""
    try:
        return EVENT_TAXONOMY[category][subcategory]
    except KeyError:
        return []

if __name__ == "__main__":
    # Example usage
    cat = "regional_central_banks"
    sub = "rba_australia"
    print(f"Category: {cat}, Subcategory: {sub}")
    print(f"Keywords: {get_keywords(cat, sub)}")
    print(f"Affected Assets: {get_affected_assets(cat, sub)}")
