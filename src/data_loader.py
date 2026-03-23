import tushare as ts
import pandas as pd
import os
import toml
from datetime import datetime, timedelta
import time

try:
    from common import ASSETS
except ImportError:
    # Fallback if run directly or path issues
    try:
        from src.common import ASSETS
    except ImportError:
        # Define locally if all else fails (or move to config file)
        ASSETS = {
            '黄金': '518880.SH',
            '纳指': '513100.SH',
            '创业板': '159915.SZ',
            '沪深300': '510300.SH'
        }

# Configuration

DATA_DIR_RAW = os.path.join('data', 'raw')
DATA_DIR_PROCESSED = os.path.join('data', 'processed')
SECRETS_PATH = os.path.join('.streamlit', 'secrets.toml')

def get_token():
    if os.path.exists(SECRETS_PATH):
        try:
            secrets = toml.load(SECRETS_PATH)
            return secrets.get('tushare', {}).get('token')
        except Exception as e:
            print(f"Error reading secrets: {e}")
            return None
    return os.environ.get('TUSHARE_TOKEN')

def init_tushare():
    token = get_token()
    if not token:
        raise ValueError("Tushare token not found in .streamlit/secrets.toml or env var TUSHARE_TOKEN")
    ts.set_token(token)
    return ts.pro_api()

def fetch_in_chunks(api_func, ts_code, start_date, end_date, chunk_years=3, **kwargs):
    """Fetch data in chunks to avoid API limits."""
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    
    all_df = pd.DataFrame()
    
    curr_start = start_dt
    while curr_start <= end_dt:
        curr_end = curr_start + timedelta(days=365*chunk_years)
        # Don't exceed end_date
        if curr_end > end_dt:
            curr_end = end_dt
            
        s_str = curr_start.strftime('%Y%m%d')
        e_str = curr_end.strftime('%Y%m%d')
        
        try:
            chunk = api_func(ts_code=ts_code, start_date=s_str, end_date=e_str, **kwargs)
            if not chunk.empty:
                all_df = pd.concat([all_df, chunk], ignore_index=True)
        except Exception as e:
            print(f"    Error fetching chunk {s_str}-{e_str}: {e}")
            
        # Move to next day after current chunk
        curr_start = curr_end + timedelta(days=1)
        time.sleep(0.3)
        
    if not all_df.empty:
        all_df = all_df.drop_duplicates(subset=['trade_date'])
        
    return all_df

def fetch_data(pro, ts_code, start_date, end_date):
    print(f"Fetching {ts_code} from {start_date} to {end_date}...")
    
    # 1. Fetch raw daily data (using chunks)
    df = fetch_in_chunks(pro.fund_daily, ts_code, start_date, end_date)
    
    if df.empty:
        print(f"Warning: No data found for {ts_code}")
        return df
        
    print(f"  Got {len(df)} records. Date range: {df['trade_date'].min()} - {df['trade_date'].max()}")
    
    # 2. Fetch adjustment factors (using chunks)
    adj_df = fetch_in_chunks(pro.fund_adj, ts_code, start_date, end_date)
    
    latest_factor = 1.0
    try:
        # Fetch the absolute latest factor available (global latest)
        latest_adj_rec = pro.fund_adj(ts_code=ts_code, limit=1)
        if not latest_adj_rec.empty:
            latest_factor = float(latest_adj_rec.iloc[0]['adj_factor'])
    except Exception as e:
        print(f"Error fetching latest adj_factor for {ts_code}: {e}")
        
    # 3. Merge and adjust
    if not adj_df.empty:
        # Merge
        df = pd.merge(df, adj_df[['trade_date', 'adj_factor']], on='trade_date', how='left')
        
        # Sort by date ascending for correct forward filling
        df = df.sort_values('trade_date', ascending=True)
        
        # Fill missing factors (forward fill: use previous day's factor for today)
        df['adj_factor'] = df['adj_factor'].ffill().fillna(1.0)
        
        # Calculate QFQ Price
        ratio = df['adj_factor'] / latest_factor
        for col in ['close', 'open', 'high', 'low']:
            if col in df.columns:
                df[col] = df[col] * ratio
    else:
        # If no adj factors found, just sort
        df = df.sort_values('trade_date', ascending=True)

    # Reset index
    df = df.reset_index(drop=True)
    
    return df

def process_data():
    pro = init_tushare()
    
    # Define date range (start from very beginning)
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = '20050101' # Sufficiently early for these ETFs
    
    all_closes = pd.DataFrame()
    
    for name, ts_code in ASSETS.items():
        # Fetch
        df = fetch_data(pro, ts_code, start_date, end_date)
        
        if df.empty:
            continue
            
        # Save raw (now actually processed/adjusted raw)
        raw_path = os.path.join(DATA_DIR_RAW, f"{ts_code}.csv")
        df.to_csv(raw_path, index=False)
        print(f"Saved adjusted data to {raw_path}")
        
        # Prepare for wide format
        # Keep trade_date and close
        df_close = df[['trade_date', 'close']].copy()
        df_close.columns = ['trade_date', ts_code]
        df_close['trade_date'] = pd.to_datetime(df_close['trade_date'])
        
        if all_closes.empty:
            all_closes = df_close
        else:
            all_closes = pd.merge(all_closes, df_close, on='trade_date', how='outer')
            
    # Sort and fill
    if not all_closes.empty:
        all_closes = all_closes.sort_values('trade_date').reset_index(drop=True)
        # Forward fill for small gaps (optional, but good for alignment)
        # all_closes = all_closes.fillna(method='ffill')
        
        wide_path = os.path.join(DATA_DIR_PROCESSED, 'etf_close_wide.csv')
        all_closes.to_csv(wide_path, index=False)
        print(f"Saved processed wide data to {wide_path}")
        
        # Calculate returns (optional, can be done on fly)
        # returns = all_closes.set_index('trade_date').pct_change()
        # returns.to_csv(os.path.join(DATA_DIR_PROCESSED, 'etf_returns.csv'))

if __name__ == "__main__":
    process_data()
