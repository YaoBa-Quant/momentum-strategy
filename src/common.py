import streamlit as st
import pandas as pd
import os

ASSETS = {
    '黄金': '518880.SH',
    '纳指': '513100.SH',
    '创业板': '159915.SZ',
    '沪深300': '510300.SH'
}

@st.cache_data
def _load_data_internal(mtime):
    """
    Internal cached function to load data. 
    The 'mtime' argument ensures cache invalidation when file changes.
    """
    path = os.path.join('data', 'processed', 'etf_close_wide.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date')
    return df

def load_data():
    """
    Load data with automatic cache invalidation based on file modification time.
    """
    path = os.path.join('data', 'processed', 'etf_close_wide.csv')
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    return _load_data_internal(mtime)
