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

def calculate_robust_features():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # 1. Load raw price data
    prices_df = pd.read_sql_query("SELECT * FROM stock_prices", conn)
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    initial_rows = len(prices_df)
    print(f"Initial price rows loaded: {initial_rows:,}")
    
    # 2. Load sentiment data
    sentiment_query = """
        SELECT date(na.published_date) as date, nt.ticker, 
               AVG(COALESCE(res.sentiment_score, res.weighted_sentiment)) as sentiment_score
        FROM news_articles na
        JOIN news_analysis res ON na.id = res.article_id
        JOIN news_to_tickers nt ON na.id = nt.news_id
        GROUP BY date, nt.ticker
    """
    try:
        sent_df = pd.read_sql_query(sentiment_query, conn)
        sent_df['date'] = pd.to_datetime(sent_df['date'])
    except Exception as e:
        print(f"Warning: Could not load sentiment: {e}")
        sent_df = pd.DataFrame(columns=['date', 'ticker', 'sentiment_score'])

    # 3. Pre-process global assets
    spy_data = prices_df[prices_df['ticker'] == 'SPY'].sort_values('date').copy()
    spy_data['ret_spy'] = spy_data['close'].pct_change(1)
    
    vix_data = prices_df[prices_df['ticker'] == '^VIX'].sort_values('date').copy()
    gld_data = prices_df[prices_df['ticker'] == 'GLD'].sort_values('date').copy()
    slv_data = prices_df[prices_df['ticker'] == 'SLV'].sort_values('date').copy()
    vix3m_data = prices_df[prices_df['ticker'] == '^VIX3M'].sort_values('date').copy()
    hyg_data = prices_df[prices_df['ticker'] == 'HYG'].sort_values('date').copy()
    ief_data = prices_df[prices_df['ticker'] == 'IEF'].sort_values('date').copy()
    tnx_data = prices_df[prices_df['ticker'] == '^TNX'].sort_values('date').copy()
    irx_data = prices_df[prices_df['ticker'] == '^IRX'].sort_values('date').copy()
    eem_data = prices_df[prices_df['ticker'] == 'EEM'].sort_values('date').copy()
    tip_data = prices_df[prices_df['ticker'] == 'TIP'].sort_values('date').copy()
    uup_data = prices_df[prices_df['ticker'] == 'UUP'].sort_values('date').copy()

    # US cross-asset anchor data
    qqq_data = prices_df[prices_df['ticker'] == 'QQQ'].sort_values('date').copy()
    iwm_data = prices_df[prices_df['ticker'] == 'IWM'].sort_values('date').copy()
    tlt_data = prices_df[prices_df['ticker'] == 'TLT'].sort_values('date').copy()

    macro_features = pd.DataFrame()
    
    # Core Macro Signals
    if not vix_data.empty and not vix3m_data.empty:
        vix_ts = pd.merge(vix_data[['date', 'close']].rename(columns={'close': 'v1'}),
                          vix3m_data[['date', 'close']].rename(columns={'close': 'v2'}), on='date')
        vix_ts['vix_term_structure'] = vix_ts['v1'] / vix_ts['v2']
        macro_features = vix_ts[['date', 'vix_term_structure']]

    if not hyg_data.empty and not ief_data.empty:
        credit = pd.merge(hyg_data[['date', 'close']].rename(columns={'close': 'h'}),
                          ief_data[['date', 'close']].rename(columns={'close': 'i'}), on='date')
        credit['credit_spread_5d'] = (credit['h'].pct_change() - credit['i'].pct_change()).rolling(5).mean()
        macro_features = pd.merge(macro_features, credit[['date', 'credit_spread_5d']], on='date', how='outer')

    if not tnx_data.empty and not irx_data.empty:
        yc = pd.merge(tnx_data[['date', 'close']].rename(columns={'close': 't'}),
                      irx_data[['date', 'close']].rename(columns={'close': 'r'}), on='date')
        yc['yield_curve_slope'] = yc['t'] - yc['r']
        macro_features = pd.merge(macro_features, yc[['date', 'yield_curve_slope']], on='date', how='outer')

    if not tnx_data.empty and not tip_data.empty and not ief_data.empty:
        ry = pd.merge(tnx_data[['date', 'close']].rename(columns={'close': 't'}),
                      tip_data[['date', 'close']].rename(columns={'close': 'tp'}), on='date')
        ry = pd.merge(ry, ief_data[['date', 'close']].rename(columns={'close': 'i'}), on='date')
        ry['real_yield_proxy'] = ry['t'] - (ry['tp'].pct_change() - ry['i'].pct_change()).rolling(20).mean() * 100
        macro_features = pd.merge(macro_features, ry[['date', 'real_yield_proxy']], on='date', how='outer')

    if not uup_data.empty:
        uup_data['dollar_momentum_10d'] = uup_data['close'].pct_change(10)
        macro_features = pd.merge(macro_features, uup_data[['date', 'dollar_momentum_10d']], on='date', how='outer')

    if not gld_data.empty and not slv_data.empty:
        gs = pd.merge(gld_data[['date', 'close']].rename(columns={'close': 'g'}),
                      slv_data[['date', 'close']].rename(columns={'close': 's'}), on='date')
        gs['gs_ratio_mom_5d'] = (gs['g'] / gs['s']).pct_change(5)
        macro_features = pd.merge(macro_features, gs[['date', 'gs_ratio_mom_5d']], on='date', how='outer')

    if not macro_features.empty:
        macro_features = macro_features.sort_values('date').ffill()

    # 4. Process each ticker
    all_features = []
    for ticker, group in prices_df.groupby('ticker'):
        group = group.sort_values('date').copy()
        
        # --- TECHNICAL INDICATORS ---
        group['rsi'] = RSIIndicator(close=group['close'], window=14).rsi()
        macd = MACD(close=group['close'])
        group['macd_diff'] = macd.macd_diff()
        for d in [1, 3, 5, 7, 10, 14, 30]:
            group[f'return_{d}d'] = group['close'].pct_change(d)
        group['volatility_7d'] = group['return_1d'].rolling(window=7).std()
        group['volatility_20d'] = group['return_1d'].rolling(window=20).std()

        bb = BollingerBands(close=group['close'], window=20, window_dev=2)
        group['bb_high_dist'] = (group['close'] / bb.bollinger_hband()) - 1
        group['bb_low_dist'] = (group['close'] / bb.bollinger_lband()) - 1
        group['atr_14'] = AverageTrueRange(high=group['high'], low=group['low'], close=group['close'], window=14).average_true_range()

        group['vol_roc_10'] = group['volume'].pct_change(10)
        group['obv'] = OnBalanceVolumeIndicator(close=group['close'], volume=group['volume']).on_balance_volume()
        group['obv_trend'] = group['obv'].diff(5)

        # --- 1. MOMENTUM OVERRIDES (OHLCV) ---
        group['sma_50'] = group['close'].rolling(window=50).mean()
        group['trend_ext_idx'] = (group['close'] - group['sma_50']) / group['atr_14'].replace(0, np.nan)
        group['momentum_accel'] = group['return_3d'] - group['return_10d']
        group['vpt_ratio'] = (group['return_1d'] * group['volume'].pct_change()).rolling(5).mean()

        # --- MARKET REGIME ---
        if not vix_data.empty:
            vix_c = vix_data[['date', 'close']].copy()
            vix_c['vix_rank'] = vix_c['close'].rolling(window=252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x)==252 else np.nan)
            group = pd.merge(group, vix_c[['date', 'vix_rank']], on='date', how='left')

        group = pd.merge(group, spy_data[['date', 'ret_spy']], on='date', how='left')
        group['rel_strength_spy'] = group['return_1d'] - group['ret_spy']
        
        merged_spy_gld = pd.merge(spy_data[['date', 'close']], gld_data[['date', 'close']], on='date', suffixes=('_s', '_g'))
        merged_spy_gld['corr_30d'] = merged_spy_gld['close_s'].rolling(30).corr(merged_spy_gld['close_g'])
        group = pd.merge(group, merged_spy_gld[['date', 'corr_30d']], on='date', how='left')

        # Merge macro features
        if not macro_features.empty:
            group = pd.merge(group, macro_features, on='date', how='left')

        # --- 2. REGIME FILTER (Macro-Price Divergence) ---
        # Detects when price ignores "scary" macro
        group['macro_price_div'] = (
            (group['return_10d'] > 0) & 
            ((group['vix_rank'] > 0.7) | (group.get('yield_curve_slope', 0) < 0))
        ).astype(int)
        group['price_vix_corr'] = group['return_5d'].rolling(20).corr(group['vix_rank'])

        # --- 3. AUSTRALIA CROSS-ASSET (Overnight Alpha) ---
        if ticker in ["^AXJO", "EWA"]:
            group['asx_spy_alpha'] = group['return_1d'] - group['ret_spy']
            group['asx_global_beta'] = group['return_1d'].rolling(30).corr(group['ret_spy'])
        else:
            group['asx_spy_alpha'] = 0.0
            group['asx_global_beta'] = 0.0

        # --- 4. US CROSS-ASSET FEATURES ---
        if ticker in ["SPY", "QQQ", "DIA", "IWM"]:
            # Tech-broad spread: QQQ vs SPY
            us_cross = pd.merge(
                qqq_data[['date', 'close']].rename(columns={'close': 'qqq_c'}),
                spy_data[['date', 'close']].rename(columns={'close': 'spy_c'}),
                on='date', how='inner'
            )
            us_cross = pd.merge(us_cross,
                iwm_data[['date', 'close']].rename(columns={'close': 'iwm_c'}),
                on='date', how='inner'
            )
            us_cross = pd.merge(us_cross,
                tlt_data[['date', 'close']].rename(columns={'close': 'tlt_c'}),
                on='date', how='inner'
            )
            us_cross['tech_broad_spread_5d'] = us_cross['qqq_c'].pct_change(5) - us_cross['spy_c'].pct_change(5)
            us_cross['size_spread_10d'] = us_cross['iwm_c'].pct_change(10) - us_cross['spy_c'].pct_change(10)
            us_cross['equity_bond_rotation'] = (us_cross['spy_c'] / us_cross['tlt_c']).pct_change(10)
            us_cross['us_breadth_5d'] = (us_cross['iwm_c'] / us_cross['spy_c']).pct_change(5)
            group = pd.merge(group, us_cross[['date', 'tech_broad_spread_5d', 'size_spread_10d',
                                               'equity_bond_rotation', 'us_breadth_5d']], on='date', how='left')
            # HYG momentum
            hyg_mom = hyg_data[['date', 'close']].copy()
            hyg_mom['hyg_momentum_5d'] = hyg_mom['close'].pct_change(5)
            group = pd.merge(group, hyg_mom[['date', 'hyg_momentum_5d']], on='date', how='left')
        else:
            group['tech_broad_spread_5d'] = 0.0
            group['size_spread_10d'] = 0.0
            group['hyg_momentum_5d'] = 0.0
            group['equity_bond_rotation'] = 0.0
            group['us_breadth_5d'] = 0.0

        # Temporal
        dow = group['date'].dt.dayofweek
        moy = group['date'].dt.month
        group['dow_sin'] = np.sin(2 * np.pi * dow / 5)
        group['dow_cos'] = np.cos(2 * np.pi * dow / 5)
        group['moy_sin'] = np.sin(2 * np.pi * (moy - 1) / 12)
        group['moy_cos'] = np.cos(2 * np.pi * (moy - 1) / 12)
        
        group['_ym'] = group['date'].dt.to_period('M')
        group['_rank_desc'] = group.groupby('_ym').cumcount(ascending=False)
        group['is_month_end'] = (group['_rank_desc'] <= 2).astype(int)
        group.drop(columns=['_ym', '_rank_desc'], inplace=True)

        # Merge Sentiment
        ticker_sent = sent_df[sent_df['ticker'] == ticker][['date', 'sentiment_score']]
        group = pd.merge(group, ticker_sent, on='date', how='left')
        group['sentiment_score'] = group['sentiment_score'].fillna(0.0)
        
        all_features.append(group)

    final_df = pd.concat(all_features)
    
    # Sector Stats
    market_map = {'SPY': 'US', 'QQQ': 'US', 'DIA': 'US', 'IWM': 'US', '^AXJO': 'AUS', 'EWA': 'AUS', 'GLD': 'COMM', 'SLV': 'COMM', 'USO': 'COMM'}
    final_df['market'] = final_df['ticker'].map(market_map).fillna('INTL')
    market_avg_ret = final_df.groupby(['date', 'market'])['return_1d'].transform('mean')
    final_df['sector_divergence'] = final_df['return_1d'] - market_avg_ret

    # Sector Sentiment
    try:
        sec_sent = pd.read_sql_query("SELECT * FROM sector_sentiment_daily", conn)
        sec_sent['date'] = pd.to_datetime(sec_sent['date'])
        if 'sentiment_score' in sec_sent.columns: sec_sent = sec_sent.drop(columns=['sentiment_score'])
        final_df = pd.merge(final_df, sec_sent, on=['date', 'ticker'], how='left')
        for col in ['sector_avg_sentiment', 'sentiment_divergence', 'sector_sentiment_momentum']:
            final_df[col] = final_df[col].fillna(0.0)
    except:
        final_df['sector_avg_sentiment'] = 0.0
        final_df['sentiment_divergence'] = 0.0
        final_df['sector_sentiment_momentum'] = 0.0

    final_df = final_df.dropna(subset=['close', 'rsi']) 
    print(f"\nFeature engineering complete. Records: {len(final_df)}")
    final_df.to_sql("features_data_robust", conn, if_exists='replace', index=False)
    conn.close()

if __name__ == "__main__":
    calculate_robust_features()
