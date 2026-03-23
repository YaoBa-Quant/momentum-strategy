import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from strategy import (
    calculate_momentum_cumulative,
    backtest_single_asset_momentum,
    get_metrics,
    calculate_returns
)
    from common import load_data, ASSETS
except ImportError as e:
    st.error(f"导入模块错误: {e}")
    st.stop()

# --- Sidebar ---
st.sidebar.title("单标的策略配置")

# Load Data
prices = load_data()

if prices is None:
    st.warning("未找到数据。请在侧边栏刷新数据。")
    st.stop()

# Asset Selection
# Invert ASSETS to get Name -> Code
asset_name = st.sidebar.selectbox(
    "选择标的", 
    options=list(ASSETS.keys()),
    format_func=lambda x: f"{x} ({ASSETS[x]})"
)
asset_code = ASSETS[asset_name]

# Check if asset exists in prices
if asset_code not in prices.columns:
    st.error(f"数据中未找到标的: {asset_name} ({asset_code})")
    st.stop()

selected_price = prices[asset_code]

# --- Date Selection ---
st.sidebar.subheader("日期范围")

# Determine valid date range for selected asset
valid_first_idx = selected_price.first_valid_index()
valid_last_idx = selected_price.last_valid_index()

if valid_first_idx is not None:
    min_date = valid_first_idx.date()
else:
    min_date = prices.index[0].date()

if valid_last_idx is not None:
    max_date = valid_last_idx.date()
else:
    max_date = prices.index[-1].date()

# Use asset-specific keys to reset date pickers when asset changes
col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    start_d = st.date_input(
        "开始日期",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        key=f'single_start_d_{asset_code}'
    )
with col_d2:
    end_d = st.date_input(
        "结束日期",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key=f'single_end_d_{asset_code}'
    )

if start_d > end_d:
    st.sidebar.error("开始日期不能晚于结束日期")
    start_d, end_d = end_d, start_d

# Slice Data
# Ensure we have enough data for lookback calculation before the start date if possible?
# But here we just slice and then calculate.
# If we slice first, the first 'lookback' days will be NaN.
# Ideally we calculate on full data then slice results.
# But for consistency with dashboard, let's just use the sliced data for calculation or handle it.
# Dashboard loads full data, calculates scores, THEN slices for backtest?
# No, dashboard slices `prices` THEN calculates? 
# Let's check dashboard.
# Dashboard: `prices = prices.loc[str(start_d):str(end_d)]` is done early.
# So first lookback days are lost in dashboard too. That's fine.
selected_price = selected_price.loc[str(start_d):str(end_d)]

if selected_price.empty:
    st.warning("所选日期范围内无数据")
    st.stop()

# --- Strategy Parameters ---
st.sidebar.subheader("策略参数")
lookback = st.sidebar.number_input("回看窗口 (天)", min_value=10, max_value=500, value=20)
# target_vol = st.sidebar.slider("目标波动率 (年化 %)", 5.0, 50.0, 15.0, step=1.0) / 100.0

st.sidebar.subheader("动量阈值")
buy_threshold_pct = st.sidebar.number_input("买入阈值 (分数 > %)", value=5.0, step=0.5, help="当动量得分（近似年化收益率）大于此值时买入") / 100.0
sell_threshold_pct = st.sidebar.number_input("卖出阈值 (分数 < %)", value=-5.0, step=0.5, help="当动量得分小于此值时卖出/空仓") / 100.0

# --- Calculation ---
st.title(f"{asset_name} 动量择时策略")

# 1. Calculate Signal (Cumulative Return / ROC)
# Signal = P(t) / P(t-lookback) - 1
price_df = selected_price.to_frame()
# skip=0 ensures we compare current day vs lookback days ago
momentum_scores = calculate_momentum_cumulative(price_df, lookback=lookback, skip=0)
signal_series = momentum_scores[asset_code]

# 2. Backtest
strategy_rets, positions = backtest_single_asset_momentum(
    selected_price, 
    signal_series, 
    buy_threshold_pct, 
    sell_threshold_pct, 
    target_vol_ann=None
)

# 3. Metrics
bench_rets = calculate_returns(selected_price).fillna(0)
# Align dates
common_idx = strategy_rets.index.intersection(bench_rets.index)
strategy_rets = strategy_rets.loc[common_idx]
bench_rets = bench_rets.loc[common_idx]
positions = positions.loc[common_idx]
signal_series = signal_series.loc[common_idx]

s_ret, s_vol, s_sharpe, s_mdd, s_win, s_odds, s_sortino = get_metrics(strategy_rets)
b_ret, b_vol, b_sharpe, b_mdd, b_win, b_odds, b_sortino = get_metrics(bench_rets)

# --- Display ---

# Metrics Table
metrics_df = pd.DataFrame({
    "指标": ["年化收益", "年化波动率", "夏普比率", "最大回撤", "胜率", "赔率 (盈亏比)", "索提诺比率"],
    "策略": [
        f"{s_ret:.2%}", f"{s_vol:.2%}", f"{s_sharpe:.2f}", f"{s_mdd:.2%}",
        f"{s_win:.2%}", f"{s_odds:.2f}", f"{s_sortino:.2f}"
    ],
    "基准 (持有)": [
        f"{b_ret:.2%}", f"{b_vol:.2%}", f"{b_sharpe:.2f}", f"{b_mdd:.2%}",
        f"{b_win:.2%}", f"{b_odds:.2f}", f"{b_sortino:.2f}"
    ]
})

st.subheader("核心指标对比")
st.dataframe(metrics_df, hide_index=True)

# Charts
st.subheader("累计收益曲线")
perf_df = pd.DataFrame({
    "策略": (1 + strategy_rets).cumprod(),
    "基准": (1 + bench_rets).cumprod()
})
fig = go.Figure()
fig.add_trace(go.Scatter(x=perf_df.index, y=perf_df["策略"], name="策略"))
fig.add_trace(go.Scatter(x=perf_df.index, y=perf_df["基准"], name="基准 (持有)", line=dict(dash='dash')))
st.plotly_chart(fig, width='stretch')
