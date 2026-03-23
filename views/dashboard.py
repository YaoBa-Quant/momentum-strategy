import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from datetime import datetime
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from strategy import (
        calculate_momentum_cumulative, 
        calculate_volatility,
        calculate_returns,
        backtest_strategy,
        get_metrics
    )
    from data_loader import process_data
    from common import load_data, ASSETS
except ImportError as e:
    st.error(f"导入模块错误: {e}")
    st.stop()

# --- Sidebar ---

# Communication (Sidebar) - Top
st.sidebar.subheader("🤝 交流与反馈")
qr_code_path = os.path.join("assets", "qrcode.jpg")
if os.path.exists(qr_code_path):
    st.sidebar.image(qr_code_path, caption="扫码加好友 (备注: 动量)")
else:
    st.sidebar.info("请将二维码图片命名为 qrcode.jpg 并放入 assets 文件夹")

st.sidebar.markdown("**🧘 作者微信:** `Code_Mvp`")
st.sidebar.markdown("获取最新策略代码 · 探讨量化思路")
st.sidebar.markdown("---")
st.sidebar.title("策略配置")

# Load Data
prices = load_data()

if prices is None:
    st.warning("未找到数据。请在侧边栏刷新数据。")
    st.stop()

# --- Date Selection ---
st.sidebar.subheader("日期范围")

min_date = prices.index[0].date()
max_date = prices.index[-1].date()

if 'date_range' not in st.session_state:
    st.session_state.date_range = (min_date, max_date)
    # Initialize widget keys if not present
    if 'start_date_input' not in st.session_state:
        st.session_state.start_date_input = min_date
    if 'end_date_input' not in st.session_state:
        st.session_state.end_date_input = max_date
else:
    # Validate and adjust stored date range to be within current min/max
    stored_range = st.session_state.date_range
    
    if isinstance(stored_range, (list, tuple)) and len(stored_range) == 2:
        start_date, end_date = stored_range
        
        # Adjust start date
        if start_date < min_date:
            start_date = min_date
        if start_date > max_date:
            start_date = min_date
            
        # Adjust end date
        if end_date > max_date:
            end_date = max_date
        if end_date < min_date:
            end_date = max_date
            
        # Ensure start <= end
        if start_date > end_date:
            start_date = min_date
            end_date = max_date
            
        st.session_state.date_range = (start_date, end_date)
        
        # Sync widget keys to avoid min_value/max_value errors
        # If the valid range changed, we MUST update the widget state manually
        st.session_state.start_date_input = start_date
        st.session_state.end_date_input = end_date
    else:
        # Reset if format is unexpected
        st.session_state.date_range = (min_date, max_date)
        st.session_state.start_date_input = min_date
        st.session_state.end_date_input = max_date

# Replace single range input with two separate inputs for better UI in sidebar
col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    start_d = st.date_input(
        "开始日期",
        value=st.session_state.date_range[0],
        min_value=min_date,
        max_value=max_date,
        key='start_date_input'
    )
with col_d2:
    end_d = st.date_input(
        "结束日期",
        value=st.session_state.date_range[1],
        min_value=min_date,
        max_value=max_date,
        key='end_date_input'
    )

# Update session state and ensure valid range
if start_d > end_d:
    st.sidebar.error("开始日期不能晚于结束日期")
    # Don't update prices if range is invalid, or swap them? 
    # Let's swap them for user convenience or just stick to old valid range
    # Here we just keep using old valid range from session state if invalid,
    # OR we can just swap them for the calculation.
    # Let's swap for calculation but warn user.
    start_d, end_d = end_d, start_d

# Update session state
st.session_state.date_range = (start_d, end_d)

if start_d and end_d:
    prices = prices.loc[str(start_d):str(end_d)]

# --- Layout ---
st.title("大鸡腿动量策略看板")
st.markdown("> 股市最大悖论之一：看似过高的通常会更高，看似过低的通常会更低 ---威廉・欧奈尔")

# Asset Selection (Main Area)
code_to_name = {v: k for k, v in ASSETS.items()}

def format_option(code):
    return f"{code_to_name.get(code, code)} ({code})"

with st.expander("标的选择", expanded=True):
    assets = st.multiselect(
        "选择参与回测的标的", 
        options=list(prices.columns), 
        default=list(prices.columns),
        format_func=format_option
    )

if not assets:
    st.error("请至少选择一个标的。")
    st.stop()

selected_prices = prices[assets]

# Signal Parameters
st.sidebar.subheader("信号参数")
st.sidebar.info("当前策略：短期动量")

lookback = st.sidebar.slider("回看窗口 (天)", 10, 40, 21)
skip = 0

# Portfolio Parameters
st.sidebar.subheader("组合构建")
top_k = st.sidebar.number_input("持仓数量 (Top K)", min_value=1, max_value=len(assets), value=1, help="决定每一期持有动量得分最高的 K 个标的。例如，设为 1 则只持有得分最高的那个。")
buffer = st.sidebar.slider("排名缓冲区 (降低换手)", 0, 3, 1, help="防止因为排名的微小变动而频繁换手。例如设置为 1，则只有当持仓标的排名跌出 Top K + 1 时才会被卖出。")

# Risk Control
st.sidebar.subheader("风险控制")
enable_vol_target = st.sidebar.checkbox("启用目标波动率控制", value=True)

if enable_vol_target:
    target_vol = st.sidebar.slider("目标波动率 (年化 %)", 5.0, 30.0, 15.0, help="基于目标波动率动态调整仓位。如果近期市场波动较大，会降低仓位以控制风险；反之则提高仓位。") / 100.0
else:
    target_vol = None

# Data Management (Bottom)
st.sidebar.markdown("---")
st.sidebar.subheader("数据管理")

if prices is not None and not prices.empty:
    last_date = prices.index[-1].strftime('%Y-%m-%d')
    st.sidebar.info(f"当前数据日期: {last_date}")

if st.sidebar.button("刷新数据 (Tushare)"):
    try:
        with st.spinner("正在从 Tushare 获取数据..."):
            process_data()
        st.success("数据已更新！")
        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(f"获取数据失败: {e}")

# --- Calculations ---

# 1. Calculate Signal
with st.spinner("正在计算信号..."):
    scores = calculate_momentum_cumulative(selected_prices, lookback, skip)
    details = {}

# 2. Backtest
# timing_filter=True: Enforce absolute momentum (cash if negative)
port_returns, positions = backtest_strategy(selected_prices, scores, top_k, buffer, timing_filter=True)

# Apply Risk Control (Target Vol)
if target_vol is not None:
    # Rolling 20d vol of the portfolio strategy
    port_rolling_std = port_returns.rolling(20).std() * np.sqrt(252)
    # Vol Scaler: Target / Realized (capped at 1.0 usually, or 1.5/2.0 for leverage)
    vol_scaler = (target_vol / port_rolling_std).clip(upper=1.0)
    # Shift scaler to avoid lookahead? 
    # Usually we scale today's position based on yesterday's vol.
    # port_returns[t] is return from t-1 to t.
    # We want to scale position at t-1 based on vol up to t-1.
    # port_rolling_std[t-1] is vol up to t-1.
    # So we multiply port_returns[t] by scaler[t-1].
    vol_scaler_shifted = vol_scaler.shift(1).fillna(0) # or 1?
    # Initial period use 1
    vol_scaler_shifted = vol_scaler_shifted.replace(0, 1.0)
else:
    # No vol targeting, scaler is 1.0
    vol_scaler_shifted = pd.Series(1.0, index=port_returns.index)

final_returns = port_returns * vol_scaler_shifted

# Drawdown Control (Hard Stop)
# If DD > Limit, cut exposure.
# This is path dependent, hard to vectorise fully if we want "cooling off".
# For simple visualization, we can just mask returns if previous DD > Limit.
cum_ret = (1 + final_returns).cumprod()
peak = cum_ret.cummax()
dd = (cum_ret - peak) / peak

# --- Tabs ---
tab1, tab2 = st.tabs(["策略看板 (截面)", "历史持仓"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("累计表现")
        
        # Benchmark: Equal Weight of selected assets
        eq_weights = pd.Series(1/len(assets), index=assets)
        bench_rets = calculate_returns(selected_prices).mean(axis=1)
        
        perf_df = pd.DataFrame({
            "策略": (1 + final_returns).cumprod(),
            "基准 (等权)": (1 + bench_rets).cumprod()
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=perf_df.index, y=perf_df["策略"], name="策略"))
        fig.add_trace(go.Scatter(x=perf_df.index, y=perf_df["基准 (等权)"], name="基准", line=dict(dash='dash')))
        st.plotly_chart(fig, width='stretch')
        
        # Drawdown Chart
        st.subheader("回撤")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd.index, y=dd, fill='tozeroy', name="回撤", line=dict(color='red')))
        st.plotly_chart(fig_dd, width='stretch')

    with col2:
        st.subheader("核心指标")
        
        s_ret, s_vol, s_sharpe, s_mdd, s_win, s_odds, s_sortino = get_metrics(final_returns)
        b_ret, b_vol, b_sharpe, b_mdd, b_win, b_odds, b_sortino = get_metrics(bench_rets)
        
        metrics_df = pd.DataFrame({
            "指标": ["年化收益", "年化波动率", "夏普比率", "最大回撤", "胜率", "赔率 (盈亏比)", "索提诺比率"],
            "策略": [
                f"{s_ret:.2%}", f"{s_vol:.2%}", f"{s_sharpe:.2f}", f"{s_mdd:.2%}",
                f"{s_win:.2%}", f"{s_odds:.2f}", f"{s_sortino:.2f}"
            ],
            "基准 (等权)": [
                f"{b_ret:.2%}", f"{b_vol:.2%}", f"{b_sharpe:.2f}", f"{b_mdd:.2%}",
                f"{b_win:.2%}", f"{b_odds:.2f}", f"{b_sortino:.2f}"
            ]
        })
        st.dataframe(metrics_df, hide_index=True)
        
        st.subheader("当前排名")
        # Get latest valid scores
        last_date = scores.dropna().index[-1]
        last_scores = scores.loc[last_date].sort_values(ascending=False)
        
        rank_df = pd.DataFrame({"得分": last_scores})
        if details:
            for k, v in details.items():
                rank_df[k] = v.loc[last_date]
        
        st.write(f"日期: {last_date.date()}")
        st.dataframe(rank_df.style.background_gradient(subset=["得分"], cmap="RdYlGn"))
        
        st.subheader("当前持仓")
        # Holdings at the END of last_date
        # Apply Volatility Scaler to match "Historical Position" and "Total Exposure"
        current_scaler = vol_scaler_shifted.loc[last_date]
        last_pos = positions.loc[last_date] * current_scaler
        
        held_assets = last_pos[last_pos > 0.0001].index.tolist()
        
        if held_assets:
            holdings_data = []
            for code in held_assets:
                holdings_data.append({
                    "代码": code,
                    "名称": code_to_name.get(code, code),
                    "权重": f"{last_pos[code]:.1%}"
                })
            st.dataframe(pd.DataFrame(holdings_data), hide_index=True)
        else:
            st.info("当前空仓 (持有现金)")

with tab2:
    st.subheader("仓位变化趋势 (总仓位)")
    
    # Calculate Total Exposure (Actual Invested %)
    # positions: Raw weights from ranking (0 or 1/K)
    # vol_scaler_shifted: Volatility targeting scalar (0.0 to 1.0)
    total_exposure = positions.sum(axis=1) * vol_scaler_shifted
    
    # Create Area Chart for Total Exposure
    fig_pos = go.Figure()
    
    fig_pos.add_trace(go.Scatter(
        x=total_exposure.index, 
        y=total_exposure, 
        mode='lines',
        name='总仓位',
        fill='tozeroy', # Fill area under line
        line_shape='hv' # Step chart logic
    ))
        
    fig_pos.update_layout(
        yaxis_title="总仓位比例 (0.0 - 1.0)",
        yaxis_tickformat='.0%',
        hovermode="x unified",
        height=400
    )
    st.plotly_chart(fig_pos, use_container_width=True)

    st.subheader("历史持仓记录 (最近100天)")
    
    # Show positions (Holdings)
    # positions df contains theoretical weights (1/K).
    # We need to multiply by vol_scaler to get actual invested weights.
    actual_positions = positions.multiply(vol_scaler_shifted, axis=0)
    
    pos_display = actual_positions.tail(100).copy()
    
    # Format: 0.50 -> "50.0%", 0.0 -> ""
    def format_pos(x):
        # Use a small threshold for float comparison
        return f"{x:.1%}" if x > 0.0001 else ""
        
    pos_display = pos_display.applymap(format_pos)
    
    # Rename columns using code_to_name
    new_cols = {c: f"{code_to_name.get(c, c)}" for c in pos_display.columns}
    pos_display = pos_display.rename(columns=new_cols)
    
    # Sort by date descending
    st.dataframe(pos_display.sort_index(ascending=False))
