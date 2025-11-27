import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="DFS Command Center", page_icon="⚡")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    div[data-testid="stMetricValue"] { color: #4ade80; font-size: 24px; }
    div[data-testid="stExpander"] { background-color: #1e293b; border: 1px solid #334155; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- MOCK DATA (Replace with your real data later) ---
nba_data = pd.DataFrame([
    {"name": "Tyus Jones", "team": "WAS", "salary": 4300, "proj_mins": 32.0, "avg_mins": 18.5, "fppm": 0.95, "proj_pts": 30.4},
    {"name": "Naz Reid", "team": "MIN", "salary": 3800, "proj_mins": 28.0, "avg_mins": 14.0, "fppm": 1.15, "proj_pts": 32.2},
    {"name": "LeBron James", "team": "LAL", "salary": 10500, "proj_mins": 36.0, "avg_mins": 35.0, "fppm": 1.45, "proj_pts": 52.2},
    {"name": "Christian Braun", "team": "DEN", "salary": 3500, "proj_mins": 18.0, "avg_mins": 15.0, "fppm": 0.75, "proj_pts": 13.5},
])

nfl_data = pd.DataFrame([
    {"team": "PHI", "qb": "Jalen Hurts", "wr1": "AJ Brown", "wr2": "DeVonta Smith", "total_proj": 59.0, "cost": 23400},
    {"team": "MIA", "qb": "Tua Tagovailoa", "wr1": "Tyreek Hill", "wr2": "Jaylen Waddle", "total_proj": 61.5, "cost": 24500},
])

# --- SIDEBAR ---
st.sidebar.title("⚡ Strategy Lab")
strategy = st.sidebar.radio(
    "Select Winning Pattern:",
    ["⏰ Minutes > Talent (NBA)", "🏈 The Onslaught (NFL)", "🏆 Stars & Scrubs (NBA)"]
)
st.sidebar.info("v9.5 | Cloud System Active")

# --- MAIN PAGE ---
st.title("DFS Command Center")

if strategy == "⏰ Minutes > Talent (NBA)":
    st.subheader("High Upside Value Plays (82% Win Rate)")
    st.write("Targeting bench players with starter minutes (24+) and low salaries.")
    
    # Logic Engine
    nba_data['minute_diff'] = nba_data['proj_mins'] - nba_data['avg_mins']
    nba_data['value_score'] = (nba_data['proj_pts'] / nba_data['salary']) * 1000
    
    opportunities = nba_data[
        (nba_data['minute_diff'] >= 8.0) & 
        (nba_data['proj_mins'] >= 24.0) & 
        (nba_data['salary'] <= 5000)
    ]
    
    if not opportunities.empty:
        cols = st.columns(len(opportunities))
        for idx, (_, row) in enumerate(opportunities.iterrows()):
            with cols[idx]:
                st.metric(
                    label=f"{row['name']} ({row['team']})", 
                    value=f"{row['value_score']:.1f}x Value",
                    delta=f"+{row['minute_diff']} Mins Surge"
                )
                st.write(f"💰 ${row['salary']} | 📈 {row['proj_pts']} Pts")
                st.caption(f"FPPM Efficiency: {row['fppm']}")
    else:
        st.warning("No plays found fitting this criteria today.")

elif strategy == "🏈 The Onslaught (NFL)":
    st.subheader("Max Correlation Stacks (71% Win Rate)")
    for _, row in nfl_data.iterrows():
        with st.expander(f"🔥 {row['team']} Stack ({row['total_proj']} Pts)", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("QB", row['qb'])
            c2.metric("WR1", row['wr1'])
            c3.metric("WR2", row['wr2'])
            st.caption(f"Total Cost: ${row['cost']}")

elif strategy == "🏆 Stars & Scrubs (NBA)":
    st.subheader("Stars & Scrubs Construction")
    c1, c2 = st.columns(2)
    with c1:
        st.success("⭐ ELITE PLAYS (> $10k)")
        st.dataframe(nba_data[nba_data['salary'] >= 10000][['name', 'salary', 'proj_pts']], use_container_width=True)
    with c2:
        st.warning("🗑️ PUNT PLAYS (< $4k)")
        st.dataframe(nba_data[nba_data['salary'] <= 4000][['name', 'salary', 'proj_pts']], use_container_width=True)
