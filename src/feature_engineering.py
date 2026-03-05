import sqlite3
import pandas as pd
import numpy as np
import os
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "database/stocks.db"

def calculate_advanced_features():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # 1. Load raw price data
    prices_df = pd.read_sql_query("SELECT * FROM stock_prices", conn)
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    
    # 2. Load sentiment data
    sentiment_query = """
        SELECT date(na.published_date) as date, nt.ticker, AVG(res.weighted_sentiment) as sentiment_score
        FROM news_articles na
        JOIN news_analysis res ON na.id = res.article_id
        JOIN news_to_tickers nt ON na.id = nt.news_id
        GROUP BY date, nt.ticker
    """
    sent_df = pd.read_sql_query(sentiment_query, conn)
    sent_df['date'] = pd.to_datetime(sent_df['date'])

    # 3. Pre-process global assets for cross-asset features
    spy_data = prices_df[prices_df['ticker'] == 'SPY'].sort_values('date').copy()
    spy_data['return_1d'] = spy_data['close'].pct_change(1)
    
    vix_data = prices_df[prices_df['ticker'] == '^VIX'].sort_values('date').copy()
    
    gld_data = prices_df[prices_df['ticker'] == 'GLD'].sort_values('date').copy()

    # 4. Process each ticker
    all_features = []
    for ticker, group in prices_df.groupby('ticker'):
        group = group.sort_values('date').copy()
        
        # Standard indicators
        group['rsi'] = RSIIndicator(close=group['close'], window=14).rsi()
        macd = MACD(close=group['close'])
        group['macd_diff'] = macd.macd_diff()
        for d in [1, 3, 7, 14, 30]:
            group[f'return_{d}d'] = group['close'].pct_change(d)
        group['volatility_7d'] = group['return_1d'].rolling(window=7).std()

        # Bollinger Bands
        bb = BollingerBands(close=group['close'], window=20, window_dev=2)
        group['bb_high_dist'] = (group['close'] / bb.bollinger_hband()) - 1
        group['bb_low_dist'] = (group['close'] / bb.bollinger_lband()) - 1

        # ATR
        group['atr_14'] = AverageTrueRange(high=group['high'], low=group['low'], close=group['close'], window=14).average_true_range()

        # Volume ROC
        group['vol_roc_10'] = group['volume'].pct_change(10)
        group['obv'] = OnBalanceVolumeIndicator(close=group['close'], volume=group['volume']).on_balance_volume()
        group['obv_trend'] = group['obv'].diff(5)

        # Market Regime
        merged_spy_gld = pd.merge(spy_data[['date', 'close']], gld_data[['date', 'close']], on='date', suffixes=('_spy', '_gld'))
        merged_spy_gld['corr_30d'] = merged_spy_gld['close_spy'].rolling(30).corr(merged_spy_gld['close_gld'])
        group = pd.merge(group, merged_spy_gld[['date', 'corr_30d']], on='date', how='left')

        vix_copy = vix_data[['date', 'close']].copy()
        vix_copy['vix_rank'] = vix_copy['close'].rolling(window=252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        group = pd.merge(group, vix_copy[['date', 'vix_rank']], on='date', how='left')

        # Relative Strength vs SPY
        group = pd.merge(group, spy_data[['date', 'return_1d']], on='date', how='left', suffixes=('', '_spy'))
        group['rel_strength_spy'] = group['return_1d'] - group['return_1d_spy']
        
        all_features.append(group)

    final_df = pd.concat(all_features)
    
    # Sector Divergence
    market_map = {
        'SPY': 'US', 'QQQ': 'US', 'DIA': 'US', 'IWM': 'US',
        '^AXJO': 'AUS', 'EWA': 'AUS', 'EWJ': 'JPN', '^N225': 'JPN',
        'MCHI': 'CHN', 'FXI': 'CHN', 'KWEB': 'CHN', '^HSI': 'HK',
        'GLD': 'COMM', 'SLV': 'COMM', 'USO': 'COMM', 'TLT': 'BONDS', '^VIX': 'VOL',
        'EWY': 'KOR', 'EPI': 'IND', 'EWZ': 'BRA', 'EWU': 'UK', 'EWG': 'GER'
    }
    final_df['market'] = final_df['ticker'].map(market_map).fillna('INTL')
    
    market_avg_ret = final_df.groupby(['date', 'market'])['return_1d'].transform('mean')
    final_df['sector_divergence'] = final_df['return_1d'] - market_avg_ret

    # 5. Merge Sentiment & Sector Logic
    # Load Sector-Level Sentiment Features
    try:
        sector_sent_df = pd.read_sql_query("SELECT * FROM sector_sentiment_daily", conn)
        sector_sent_df['date'] = pd.to_datetime(sector_sent_df['date'])
        
        # We merge on date and ticker to get divergence, and date for global sector scores
        final_df = pd.merge(final_df, sector_sent_df, on=['date', 'ticker'], how='left', suffixes=('', '_drop'))
        # Cleanup redundant columns from the merge
        final_df = final_df.drop(columns=[col for col in final_df.columns if col.endswith('_drop')])
    except:
        print("Warning: sector_sentiment_daily table not found. Skipping sector sentiment features.")
        final_df = pd.merge(final_df, sent_df, on=['date', 'ticker'], how='left').fillna({'sentiment_score': 0})

    # 6. Save to Database
    print("Saving processed features to 'features_data' table...")
    final_df.to_sql("features_data", conn, if_exists='replace', index=False)
    
    conn.close()
    print()
    print(f"Feature engineering complete. Total records: {len(final_df)}")

if __name__ == "__main__":
    calculate_advanced_features()
