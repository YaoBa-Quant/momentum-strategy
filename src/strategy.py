import pandas as pd
import numpy as np
from scipy import stats

def calculate_returns(prices_df):
    """
    Calculate daily logarithmic returns.
    """
    return np.log(prices_df / prices_df.shift(1))

def calculate_volatility(returns_df, window=20):
    """
    Calculate rolling volatility (annualized).
    """
    return returns_df.rolling(window=window).std() * np.sqrt(252)

def calculate_momentum_cumulative(prices_df, lookback=252, skip=21):
    """
    Classic Momentum: Cumulative return over lookback, skipping recent 'skip' days.
    M = P(t-skip) / P(t-skip-lookback) - 1
    """
    # Shifted prices to skip recent days
    recent_prices = prices_df.shift(skip)
    past_prices = prices_df.shift(lookback + skip)
    return recent_prices / past_prices - 1

def calculate_slope_r2(prices_df, lookback=252):
    """
    Calculate annualized slope and R-squared of log prices over a lookback window.
    Returns two DataFrames: slope_ann, r2
    """
    log_prices = np.log(prices_df)
    
    # We use a rolling apply. It's slow for large DataFrames but fine for <100 assets.
    # For speed, we can use numpy strides or specialized libraries, 
    # but for clarity and "reproducibility", we'll use a clean loop or apply.
    
    slope_df = pd.DataFrame(index=prices_df.index, columns=prices_df.columns)
    r2_df = pd.DataFrame(index=prices_df.index, columns=prices_df.columns)
    
    # Pre-calculate X (time index)
    x = np.arange(lookback)
    
    # Helper for rolling regression
    def rolling_reg(y):
        # y is a series of length lookback
        if np.isnan(y).any():
            return np.nan, np.nan
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return slope, r_value**2

    # Iterate over columns (assets) to vectorise over time somewhat efficiently
    for col in prices_df.columns:
        series = log_prices[col]
        
        # Use rolling window
        # Scipy linregress is not easily vectorised for rolling. 
        # Using numpy polyfit inside rolling apply is faster.
        
        def get_stats(y):
            if np.isnan(y).any(): return np.nan, np.nan
            # Polyfit degree 1
            # b = slope, a = intercept
            b, a = np.polyfit(x, y, 1)
            
            # Calculate R2
            y_pred = a + b * x
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            return b, r2

        # To speed up, we can use stride_tricks or just a loop if data is small. 
        # Since we have ~2500 rows and 4 cols, a loop over rolling windows is acceptable.
        # But rolling().apply only returns scalar.
        
        # Optimization:
        # Calculate vectorized rolling covariance and variance
        # slope = cov(x, y) / var(x)
        # r2 = corr(x, y)^2
        
        y = series
        
        # Rolling covariance of y with x (x is constant [0, 1, ..., L-1])
        # We can treat x as a moving window feature? No, x is always 0..L-1 relative to the window.
        # So we construct a rolling correlation against a fixed ramp.
        # But rolling corr in pandas expects another series.
        # We can't easily do rolling corr against "index" because index changes.
        
        # Actually, if we assume daily data is contiguous, we can use rolling correlation 
        # between price series and an integer sequence of same length?
        # No, that's correlation with "absolute time". We need correlation with "relative time in window".
        # But for R2, Corr(y, t) is the same as Corr(y, t_relative) since shift doesn't affect correlation.
        # So we can just make a series T = 0, 1, 2... and do rolling corr(Price, T).
        
        T_series = pd.Series(np.arange(len(series)), index=series.index)
        
        # Rolling Correlation (which is sqrt(R2) * sign(slope))
        # R2 = Correlation^2
        rolling_corr = series.rolling(lookback).corr(T_series)
        r2 = rolling_corr ** 2
        
        # Slope = Corr * (StdY / StdX)
        # StdX is constant for a fixed window length L
        std_x = np.std(np.arange(lookback), ddof=1) # Sample std dev
        rolling_std_y = series.rolling(lookback).std()
        
        slope = rolling_corr * (rolling_std_y / std_x)
        
        slope_df[col] = slope
        r2_df[col] = r2

    # Annualize slope: exp(252 * b) - 1
    slope_ann = np.exp(252 * slope_df) - 1
    
    return slope_ann, r2_df

def calculate_score_slope_r2(slope_ann, r2):
    """
    Score = SlopeAnn * R2
    """
    return slope_ann * r2

def backtest_strategy(prices, signals, top_k=1, buffer=0, cash_proxy_return=0.0, timing_filter=False):
    """
    Vectorised backtest for Top K strategy.
    
    signals: DataFrame of scores (higher is better)
    top_k: Number of assets to hold
    buffer: Rank buffer to reduce turnover
    cash_proxy_return: Daily return for uninvested cash (not implemented yet, assuming full investment in TopK)
    timing_filter: If True, do not hold assets with negative signal (absolute momentum)
    
    Returns:
        portfolio_returns, positions_mask
    """
    # 1. Rank assets daily: 1 is best, N is worst
    # We use ascending=False, so Rank 1 is highest score
    ranks = signals.rank(axis=1, ascending=False, method='min')
    
    # 2. Apply Buffer Logic (Iterative)
    # If we have buffer, we need to know yesterday's holdings.
    # This makes it path-dependent, so we can't fully vectorise easily.
    # We will loop through days for the position logic.
    
    positions = pd.DataFrame(0, index=signals.index, columns=signals.columns)
    
    # Initial holdings (first day with valid signals)
    # Find first valid index
    valid_idx = signals.dropna(how='all').index
    if len(valid_idx) == 0:
        return pd.Series(0, index=signals.index), positions
        
    start_idx = signals.index.get_loc(valid_idx[0])
    
    # Array for speed
    ranks_arr = ranks.values
    pos_arr = np.zeros_like(ranks_arr)
    n_assets = ranks_arr.shape[1]
    
    current_holdings = set()
    
    for i in range(start_idx, len(signals)):
        # Today's ranks
        today_ranks = ranks_arr[i]
        
        # Candidates for entry: Rank <= K
        # Candidates for exit: Rank > K + Buffer
        
        # If we have no holdings, just take Top K
        if not current_holdings:
            # Indices where rank <= K
            # Handle NaNs (rank is NaN if signal is NaN)
            candidates = np.where(today_ranks <= top_k)[0]
            # Take top K (in case of ties or just to be safe)
            # Actually rank method='min' handles ties, but let's stick to the set logic
            new_holdings = set(candidates)
        else:
            new_holdings = set()
            # Check existing holdings
            for asset_idx in current_holdings:
                # Keep if rank <= K + buffer
                if today_ranks[asset_idx] <= (top_k + buffer):
                    new_holdings.add(asset_idx)
                else:
                    # Sell
                    pass
            
            # Fill remaining spots with best available non-held assets
            needed = top_k - len(new_holdings)
            if needed > 0:
                # Find best assets not in new_holdings
                # Sort indices by rank
                # We need to handle NaNs in ranks (np.argsort puts NaNs at end usually)
                sorted_indices = np.argsort(today_ranks)
                
                for idx in sorted_indices:
                    if needed == 0: break
                    if np.isnan(today_ranks[idx]): continue # Skip invalid
                    
                    if idx not in new_holdings:
                        # Only buy if it qualifies as a "good" asset? 
                        # Strictly speaking, we just buy the best available to fill Top K.
                        # But typically we only buy if Rank <= K (strict entry).
                        # Let's assume strict entry: Must be in Top K to enter.
                        if today_ranks[idx] <= top_k:
                            new_holdings.add(idx)
                            needed -= 1
        
        current_holdings = new_holdings
        
        # Mark positions
        for idx in current_holdings:
            pos_arr[i, idx] = 1
            
    positions = pd.DataFrame(pos_arr, index=signals.index, columns=signals.columns)
    
    # Shift signals by 1 day to respect "Signal t-1, Trade t Close"
    # eff_signals.loc[t] contains signal computed at t-1
    eff_signals = signals.shift(1)
    
    # We loop to determine positions based on eff_signals
    # Loop starts from index 1
    
    # Re-initialize for safety
    positions = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    pos_arr = np.zeros((len(signals), signals.shape[1]))
    
    # Pre-calculate ranks for eff_signals
    # eff_signals has NaN at row 0
    eff_ranks = eff_signals.rank(axis=1, ascending=False, method='min')
    eff_ranks_arr = eff_ranks.values
    eff_signals_arr = eff_signals.values
    
    current_holdings = set()
    
    for i in range(len(signals)):
        # If signal is all NaN (e.g. first day), skip
        # checking row i of eff_signals
        if np.isnan(eff_ranks_arr[i]).all():
            current_holdings = set() # No holdings
            continue
            
        today_ranks = eff_ranks_arr[i]
        today_signals = eff_signals_arr[i]
        
        # Buffer Logic
        if not current_holdings:
            # Entry: Top K
            # Indices where rank <= top_k
            candidates = np.where(today_ranks <= top_k)[0]
            
            # Apply Timing Filter
            if timing_filter:
                # Filter candidates where signal <= 0
                valid_candidates = []
                for c in candidates:
                    if today_signals[c] > 0:
                        valid_candidates.append(c)
                new_holdings = set(valid_candidates)
            else:
                new_holdings = set(candidates)
        else:
            new_holdings = set()
            # 1. Keep existing if rank <= top_k + buffer
            for idx in current_holdings:
                if not np.isnan(today_ranks[idx]) and today_ranks[idx] <= (top_k + buffer):
                    # Timing Check
                    if timing_filter and today_signals[idx] <= 0:
                        continue
                    new_holdings.add(idx)
            
            # 2. Fill if needed
            needed = top_k - len(new_holdings)
            if needed > 0:
                # Get all valid candidates sorted by rank
                # argsort handles NaNs by putting them at end, but we must check
                sorted_indices = np.argsort(today_ranks)
                
                for idx in sorted_indices:
                    if needed == 0: break
                    if np.isnan(today_ranks[idx]): continue
                    
                    if idx not in new_holdings:
                        # Timing Check
                        if timing_filter and today_signals[idx] <= 0:
                            continue

                        # Strict entry: Only buy if Rank <= top_k
                        if today_ranks[idx] <= top_k:
                            new_holdings.add(idx)
                            needed -= 1
        
        current_holdings = new_holdings
        
        # Record positions
        # Equal weight for now (1/K)
        if len(current_holdings) > 0:
            weight = 1.0 / len(current_holdings)
            for idx in current_holdings:
                pos_arr[i, idx] = weight
                
    positions = pd.DataFrame(pos_arr, index=signals.index, columns=signals.columns)
    
    # Calculate Portfolio Returns
    # Strategy Return at t = Sum(Position[t-1] * AssetReturn[t])
    # Position[t-1] is the portfolio held from Close t-1 to Close t.
    # We calculated `positions` such that `positions.iloc[i]` is the target portfolio at Close i.
    # So we need to shift positions by 1 to align with returns.
    
    daily_rets = calculate_returns(prices)
    # Fill NaN returns with 0 for dot product safety
    safe_rets = daily_rets.fillna(0)
    
    # Portfolio Return
    # shift(1) moves pos at t to t+1
    pos_shifted = positions.shift(1)
    
    port_rets = (pos_shifted * safe_rets).sum(axis=1)
    
    return port_rets, positions

def backtest_single_asset_momentum(price_series, signal_series, buy_threshold, sell_threshold, target_vol_ann=None):
    """
    Single Asset Momentum Strategy with Thresholds and Volatility Targeting.
    
    price_series: pd.Series of asset prices
    signal_series: pd.Series of momentum scores (e.g. Slope*R2)
    buy_threshold: float (e.g. 0.05 for 5%)
    sell_threshold: float (e.g. -0.02 for -2%)
    target_vol_ann: float (e.g. 0.15) or None. If None, no vol targeting.
    
    Returns:
        strategy_returns: pd.Series
        positions: pd.Series (weights)
    """
    # 1. Generate Signal State (0 or 1) based on thresholds
    # We iterate because of hysteresis (state dependence)
    
    # Ensure alignment
    common_idx = price_series.index.intersection(signal_series.index)
    price_s = price_series.loc[common_idx]
    sig_s = signal_series.loc[common_idx]
    
    states = []
    current_state = 0
    
    for i in range(len(sig_s)):
        sig = sig_s.iloc[i]
        
        if np.isnan(sig):
            states.append(0)
            continue
            
        if sig > buy_threshold:
            current_state = 1
        elif sig < sell_threshold:
            current_state = 0
        # Else keep current_state
        
        states.append(current_state)
        
    state_series = pd.Series(states, index=sig_s.index)
    
    # 2. Volatility Targeting
    if target_vol_ann is not None:
        # Calculate Realized Vol (20-day rolling?)
        rets = np.log(price_s / price_s.shift(1))
        # Use 20-day rolling window for realized volatility
        realized_vol = rets.rolling(20).std() * np.sqrt(252)
        
        # Vol Scalar
        # Avoid division by zero
        vol_scalar = target_vol_ann / realized_vol.replace(0, np.inf)
        
        # Cap scalar at 1.0 (no leverage)
        vol_scalar = vol_scalar.fillna(0)
        vol_scalar = np.minimum(vol_scalar, 1.0)
        
        positions = state_series * vol_scalar
    else:
        positions = state_series
        
    # 3. Calculate Returns
    # Strategy Return = Position(t-1) * Return(t)
    rets = np.log(price_s / price_s.shift(1)).fillna(0)
    strategy_returns = positions.shift(1) * rets
    
    return strategy_returns, positions

def get_metrics(r):
    """
    Calculate performance metrics for a return series.
    Returns: ann_ret, ann_vol, sharpe, mdd, win_rate, odds, sortino
    """
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    
    # Max DD
    cr = (1+r).cumprod()
    pk = cr.cummax()
    d = (cr - pk) / pk
    mdd = d.min()
    
    # Win Rate
    valid_days = r[r != 0]
    if len(valid_days) > 0:
        win_rate = len(valid_days[valid_days > 0]) / len(valid_days)
    else:
        win_rate = 0.0

    # Profit/Loss Ratio (Odds)
    avg_win = valid_days[valid_days > 0].mean() if len(valid_days[valid_days > 0]) > 0 else 0
    avg_loss = valid_days[valid_days < 0].mean() if len(valid_days[valid_days < 0]) > 0 else 0
    odds = avg_win / abs(avg_loss) if avg_loss != 0 else np.inf

    # Sortino Ratio
    downside_returns = r[r < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = ann_ret / downside_std if downside_std != 0 else np.inf
    
    return ann_ret, ann_vol, sharpe, mdd, win_rate, odds, sortino


