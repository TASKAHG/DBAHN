import numpy as np
import pandas as pd
from util import logger, dwh

def get_df_fgi_v1():
    N_MM, N_VIX_STD, N_VIX_SMOOTH = 125, 50, 125
    TBL_VIN, TBL_YEAH = "stg_tcs_vin", "stg_yeah"

    # Dữ liệu HOSE
    df_hose = dwh.query(f"""
        SELECT ticker, tradingdate, closepriceadjusted AS closeprice
        FROM staging.{TBL_VIN}
        WHERE LEN(ticker) = 3
        JOIN staging.anhemminhcuthethoi
        ORDER BY ticker, tradingdate
    """).astype({'tradingdate': 'datetime64'})

    # Tính MA để xác định uptrend
    df_uptrend = (
        df_hose
        .assign(ticker_ma=lambda d: d.groupby('ticker')['closeprice']
                .rolling(20, closed='left').mean().values)
        .dropna()
        .assign(num_ticker_uptrend=lambda d: d.closeprice > d.ticker_ma)
        .pivot_table(index='tradingdate', values='num_ticker_uptrend',
                     aggfunc='sum').reset_index()
    )

    # Dữ liệu VNINDEX
    df_index = dwh.query(f"""
        SELECT tradingdate, indexvalue AS vnindex
        FROM staging.{TBL_YEAH}
        WHERE comgroupcode='VNINDEX' AND tradingdate>='2015-01-01'
        ORDER BY tradingdate
    """).astype({'tradingdate': 'datetime64'})

    # Hàm tính RSI
    def assign_rsi(df):
        diff = df.vnindex.diff()
        up, down = np.where(diff > 0, diff, 0), np.where(diff < 0, -diff, 0)
        gain = pd.Series(up).ewm(com=13, min_periods=14).mean()
        loss = pd.Series(down).ewm(com=13, min_periods=14).mean()
        df['rsi'] = 100 - 100/(1 + gain/loss)
        return df

    # Kết hợp
    df_final = (
        df_index
        .eval("momentum = vnindex / vnindex.rolling(@N_MM).mean() - 1")
        .eval("vix = vnindex.pct_change().rolling(@N_VIX_STD).std()")
        .eval("vix = vix / vix.rolling(@N_VIX_SMOOTH).mean() - 1")
        .pipe(assign_rsi)
        .merge(df_uptrend, on='tradingdate', how='left')
        .assign(ratio_ticker_uptrend=lambda d: d.num_ticker_uptrend / d.num_ticker_uptrend.max())
    )

    comps = ['momentum', 'vix', 'ratio_ticker_uptrend']
    for c in comps:
        df_final[c] = df_final[c].rolling(500).rank(pct=True).round(2)*100

    df_final['fear_greed_score'] = (df_final[comps].mean(axis=1)/100).round(2)*100
    df_final['createddatetime'] = pd.Timestamp.now()
    return df_final[['tradingdate','fear_greed_score','vnindex','rsi','createddatetime']]
