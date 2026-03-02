# Event Taxonomy for Stock Predictor
# This file defines the classification of financial news events and their impact on specific tickers.

EVENT_TAXONOMY = {
    "macroeconomics": {
        "inflation": ["CPI", "PPI", "inflation", "cost of living", "price index"],
        "employment": ["nonfarm payrolls", "unemployment rate", "jobless claims", "labor market"],
        "monetary_policy": ["Fed", "FOMC", "interest rates", "rate hike", "rate cut", "Jerome Powell"]
    }
}

# Mapping of subcategories to affected tickers
AFFECTED_ASSETS_MAPPING = {
    # Macro (General US)
    "inflation": ["SPY", "QQQ", "DIA", "IWM", "TLT"],
    "employment": ["SPY", "DIA", "IWM"],
    "monetary_policy": ["SPY", "QQQ", "DIA", "IWM", "TLT"]
}

def get_affected_assets(event_category, subcategory):
    """
    Returns a list of tickers most affected by a specific event subcategory.
    """
    return AFFECTED_ASSETS_MAPPING.get(subcategory, ["SPY"])

def get_keywords(category, subcategory):
    """Returns keywords for a specific category/subcategory."""
    try:
        return EVENT_TAXONOMY[category][subcategory]
    except KeyError:
        return []

if __name__ == "__main__":
    # Example usage
    cat = "macroeconomics"
    sub = "inflation"
    print(f"Category: {cat}, Subcategory: {sub}")
    print(f"Keywords: {get_keywords(cat, sub)}")
    print(f"Affected Assets: {get_affected_assets(cat, sub)}")
