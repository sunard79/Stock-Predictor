"""
Extended feature engineering for backtesting validation.
Features are computed on-the-fly and NOT saved to DB until validated.
"""
import pandas as pd
import numpy as np


def add_extended_features(df):
    """
    Add extended features to a DataFrame that already has the base features.
    Returns (df_with_new_features, list_of_new_column_names).

    Expects df to have columns: ticker, date, close, return_1d, sentiment_score,
    news_count, volatility_7d, return_7d, and cross-asset columns.
    """
    df = df.copy()
    new_cols = []

    # --- Lagged cross-asset returns ---
    # SPY lag-1 return (captures lead-lag between markets)
    spy_returns = (
        df[df['ticker'] == 'SPY'][['date', 'return_1d']]
        .rename(columns={'return_1d': 'spy_return_lag1'})
        .sort_values('date')
    )
    spy_returns['spy_return_lag1'] = spy_returns['spy_return_lag1'].shift(1)
    df = df.merge(spy_returns, on='date', how='left')
    new_cols.append('spy_return_lag1')

    # VIX lag-1 return
    vix_data = df[df['ticker'] == '^VIX'][['date', 'return_1d']].rename(
        columns={'return_1d': 'vix_return_lag1'}
    ).sort_values('date')
    vix_data['vix_return_lag1'] = vix_data['vix_return_lag1'].shift(1)
    df = df.merge(vix_data, on='date', how='left')
    new_cols.append('vix_return_lag1')

    # GLD lag-1 return
    gld_data = df[df['ticker'] == 'GLD'][['date', 'return_1d']].rename(
        columns={'return_1d': 'gld_return_lag1'}
    ).sort_values('date')
    gld_data['gld_return_lag1'] = gld_data['gld_return_lag1'].shift(1)
    df = df.merge(gld_data, on='date', how='left')
    new_cols.append('gld_return_lag1')

    # --- Momentum acceleration (change in 7d return over 7 days) ---
    df = df.sort_values(['ticker', 'date'])
    df['momentum_accel_7d'] = df.groupby('ticker')['return_7d'].diff(7)
    new_cols.append('momentum_accel_7d')

    # --- Sentiment momentum (3-day trend in sentiment) ---
    df['sentiment_momentum_3d'] = df.groupby('ticker')['sentiment_score'].transform(
        lambda x: x.rolling(3, min_periods=1).mean() - x.rolling(7, min_periods=3).mean()
    )
    new_cols.append('sentiment_momentum_3d')

    # --- Day of week (0=Monday, 4=Friday) ---
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    new_cols.append('day_of_week')

    # --- News volume spike (binary: > 2x 30-day average) ---
    df['news_avg_30d'] = df.groupby('ticker')['news_count'].transform(
        lambda x: x.rolling(30, min_periods=5).mean()
    )
    df['news_volume_spike'] = (df['news_count'] > 2 * df['news_avg_30d']).astype(float)
    df.drop(columns=['news_avg_30d'], inplace=True)
    new_cols.append('news_volume_spike')

    # --- Return/volatility ratio (breakout detector) ---
    df['return_vol_ratio'] = df['return_1d'] / df['volatility_7d'].replace(0, np.nan)
    df['return_vol_ratio'] = df['return_vol_ratio'].clip(-5, 5).fillna(0)
    new_cols.append('return_vol_ratio')

    # Fill NaN in new columns
    for col in new_cols:
        df[col] = df[col].fillna(0.0)

    return df, new_cols
