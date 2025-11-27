import streamlit as st
import pandas as pd
import numpy as np
import pulp
import base64
from duckduckgo_search import DDGS
from sklearn.preprocessing import MinMaxScaler
import requests
import feedparser
import math
import sys

# Increase recursion limit for the pulp solver
sys.setrecursionlimit(2000)

# Try importing NFL data (optional, but robust)
try:
    import nfl_data_py as nfl
    NFL_DATA_AVAILABLE = True
except ImportError:
    NFL_DATA_AVAILABLE = False

# --- CONFIGURATION & STYLING (Standard) ---
st.set_page_config(layout="wide", page_title="TITAN COMMAND: FINAL FUSION", page_icon="💥")

st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #e2e8f0; font-family: 'Roboto', sans-serif; }
    div[data-testid="stMetricValue"] { color: #38bdf8; font-size: 24px; font-weight: 800; }
    .reasoning-text { font-size: 12px; color: #94a3b8; font-style: italic; border-left: 2px solid #3b82f6; padding-left: 8px; }
    .stButton>button { width: 100%; border-radius: 6px; font-weight: bold; background-color: #3b82f6; border: none; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 2. THE TITAN BRAIN (LOGIC CLASS)
# ==========================================

class TitanBrain:
    def __init__(self, sport, spread=0, total=0):
        self.sport = sport
        self.spread = spread
        self.total = total
        
    def evaluate_player(self, row):
        reasons = []
        score = 50.0 
        
        # 1. LEVERAGE LOGIC
        if 'rank_proj' in row and 'rank_own' in row and row['rank_proj'] > 0 and row['rank_own'] > 0:
            leverage_diff = row['rank_own'] - row['rank_proj']
            if leverage_diff > 15:
                score += 15
                reasons.append(f"💎 High Leverage (Contrarian Value)")
            elif leverage_diff < -10:
                score -= 10
                reasons.append("⚠️ Chalk Trap (Overowned)")
                
        # 2. GAME SCRIPT LOGIC (NFL Focus)
        if self.sport == "NFL" and 'position' in row:
            if abs(self.spread) > 7 and 'WR' in str(row['position']):
                score += 7
                reasons.append("📜 Blowout Script: Garbage Time potential")

        # 3. PROP VALUE LOGIC
        if 'prop_line' in row and 'proj_pts' in row and row['prop_line'] > 0:
            edge = ((row['proj_pts'] - row['prop_line']) / row['proj_pts']) * 100
            if edge > 15:
                score += 15
                reasons.append(f"💰 Prop Smash ({edge:.1f}% Edge)")

        # 4. UPSIDE/USAGE METRICS
        if 'ceiling' in row and row['ceiling'] > row['proj_pts'] * 1.5:
            score += 10
            reasons.append("🔥 Massive Ceiling Differential")
            
        final_score = max(0, min(100, score))
        verdict_text = "✅ PLAY: " + " | ".join(reasons) if reasons else "ℹ️ NEUTRAL: Balanced profile."
        return final_score, verdict_text

# ==========================================
# 🛠️ 3. DATA PROCESSING & UTILITIES
# ==========================================

def standardize_columns(df):
    """Universal Translator & Cleaner (Safest Version)"""
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('-', '_').str.replace('$', '').str.replace(',', '').str.replace('%', '')
    column_map = {
        'name': ['player', 'athlete', 'full_name'], 'proj_pts': ['projection', 'proj', 'fpts', 'median'],
        'ownership': ['own', 'projected_ownership'], 'salary': ['cost', 'sal', 'price'],
        'prop_line': ['line', 'prop', 'ou', 'total', 'strike'], 'position': ['pos', 'roster_position'],
        'team': ['squad', 'tm'], 'ceiling': ['ceil', 'max_pts'], 'opp_rank': ['dvp', 'opp_rank']
    }
    renamed = {}
    for std, alts in column_map.items():
        for col in df.columns:
            if col not in renamed.values():
                if any(a in col for a in al
