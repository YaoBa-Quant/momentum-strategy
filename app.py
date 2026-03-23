import streamlit as st

st.set_page_config(page_title="大鸡腿动量策略看板 -- 股市最大悖论之一：看似过高的通常会更高，看似过低的通常会更低 ---威廉・欧奈尔", layout="wide")

# Page setup
dashboard = st.Page("views/dashboard.py", title="策略看板 (多标的)", icon="📈", default=True)
single_asset = st.Page("views/single_asset.py", title="单标的择时", icon="⏱️")
system_info = st.Page("views/system_info.py", title="系统说明", icon="📘")
version_log = st.Page("views/version_log.py", title="版本更新", icon="📝")

# Navigation
pg = st.navigation([dashboard, single_asset, system_info, version_log])
pg.run()
