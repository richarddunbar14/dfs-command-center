import streamlit as st
import pandas as pd
import numpy as np
import pulp
import re
import time
import sqlite3
import base64
import requests
import plotly.express as px
from datetime import datetime

# 🛡️ SAFE IMPORT
try:
    from duckduckgo_search import DDGS
    HAS_AI = True
except ImportError:
    HAS_AI = False

# ==========================================
# ⚙️ 1. CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="TITAN OMNI: V48.0", page_icon="🌌")

st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    .titan-card { background: #111; border: 1px solid #333; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    div[data-testid="stDataFrame"] { border: 1px solid #333; border-radius: 5px; }
    .stButton>button { width: 100%; border-radius: 4px; font-weight: 800; text-transform: uppercase; background: linear-gradient(90deg, #111 0%, #222 100%); color: #29b6f6; border: 1px solid #29b6f6; transition: 0.3s; }
    .stButton>button:hover { background: #29b6f6; color: #000; box-shadow: 0 0 15px #29b6f6; }
</style>
""", unsafe_allow_html=True)

if 'dfs_pool' not in st.session_state: st.session_state['dfs_pool'] = pd.DataFrame()
if 'prop_pool' not in st.session_state: st.session_state['prop_pool'] = pd.DataFrame()

# ==========================================
# 📡 2. API GATEWAY
# ==========================================
class MultiVerseGateway:
    def __init__(self, api_key):
        self.headers = {"X-RapidAPI-Key": api_key, "Content-Type": "application/json"}
    def fetch_data(self, sport):
        # Robust Placeholder
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def cached_api_fetch(sport, key): return MultiVerseGateway(key).fetch_data(sport)

# ==========================================
# 💾 3. DATABASE
# ==========================================
def init_db():
    conn = sqlite3.connect('titan_multiverse.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS bankroll (id INTEGER PRIMARY KEY, date TEXT, amount REAL, notes TEXT)''')
    c.execute('SELECT count(*) FROM bankroll')
    if c.fetchone()[0] == 0: c.execute("INSERT INTO bankroll (date, amount, notes) VALUES (?, ?, ?)", (datetime.now().strftime("%Y-%m-%d"), 1000.0, 'Genesis')); conn.commit()
    return conn

def get_bankroll(conn):
    try: return pd.read_sql("SELECT * FROM bankroll ORDER BY id DESC LIMIT 1", conn).iloc[0]['amount']
    except: return 1000.0

# ==========================================
# 🧠 4. BRAIN & LOGIC
# ==========================================
class TitanBrain:
    def apply_boosts(self, row, sport):
        proj = row.get('projection', 0.0); notes = []
        # Simple Boosts to avoid crashes
        try:
            if row.get('l3_fpts', 0) > row.get('avg_fpts', 0) * 1.2: 
                proj *= 1.05; notes.append("🔥 Hot")
            if sport in ['NFL', 'MLB'] and row.get('wind', 0) > 15:
                if 'QB' in str(row.get('position')): proj *= 0.85; notes.append("🌪️ Wind")
        except: pass
        return proj, " | ".join(notes)

    def calc_ev(self, slip, book, stake):
        probs = slip['Win Prob'].values / 100.0
        legs = len(probs)
        payout = 3.0 if legs==2 else 5.0 if legs==3 else 10.0
        win = np.prod(probs)
        return win * 100, payout, (win * payout * stake) - stake

# ==========================================
# 📂 5. DATA REFINERY
# ==========================================
class DataRefinery:
    @staticmethod
    def clean(val):
        try: return float(re.sub(r'[^\d.-]', '', str(val).strip()))
        except: return 0.0

    @staticmethod
    def norm_pos(pos):
        p = str(pos).upper().strip()
        if 'QUARTER' in p: return 'QB'
        if 'RUNNING' in p: return 'RB'
        if 'RECEIVER' in p: return 'WR'
        if 'TIGHT' in p: return 'TE'
        if 'DEF' in p: return 'DST'
        if 'GUARD' in p or 'G/UTIL' in p: return 'G'
        if 'FORWARD' in p or 'F/UTIL' in p: return 'F'
        if 'CENTER' in p: return 'C'
        if 'GOALIE' in p: return 'G'
        return p

    @staticmethod
    def ingest(df, sport, source):
        df.columns = df.columns.astype(str).str.upper().str.strip()
        std = pd.DataFrame()
        
        # KEYWORD MAPPING
        maps = {
            'name': ['PLAYER', 'NAME'],
            'projection': ['FPTS', 'PROJ', 'ROTOWIRE PROJECTION', 'PTS'],
            'salary': ['SAL', 'COST'],
            'position': ['POS', 'POSITION'],
            'team': ['TEAM', 'TM'],
            'opp': ['OPP', 'VS'],
            'prop_line': ['LINE', 'PROP', 'TOTAL'],
            'market': ['MARKET'],
            'game_info': ['GAME INFO', 'GAME'],
            'status': ['STATUS', 'INJURY'],
            'factor_hit_rate': ['HIT RATE FACTOR']
        }
        
        # Site Specifics
        site_maps = {
            'prizepicks_line': ['PRIZEPICKS LINE', 'PRIZEPICKS'],
            'underdog_line': ['UNDERDOG LINE', 'UNDERDOG']
        }
        
        for target, sources in maps.items():
            for s in sources:
                if s in df.columns:
                    if target in ['projection','salary','prop_line','factor_hit_rate']: std[target] = df[s].apply(DataRefinery.clean)
                    elif target == 'position': std[target] = df[s].apply(DataRefinery.norm_pos)
                    else: std[target] = df[s].astype(str).str.strip()
                    break
        
        for target, sources in site_maps.items():
            for s in sources:
                if s in df.columns:
                    std[target] = df[s].apply(DataRefinery.clean)
                    break
            if target not in std.columns: std[target] = 0.0

        if 'name' not in std.columns: return pd.DataFrame()
        
        # FORCE GAME INFO
        if 'game_info' not in std.columns:
            if 'team' in std.columns and 'opp' in std.columns: std['game_info'] = std['team'] + ' vs ' + std.get('opp', 'Opp')
            else: std['game_info'] = 'All Games'
        
        std['sport'] = sport.upper()
        std['status'] = std.get('status', pd.Series(['Active']*len(std))).fillna('Active')
        std['salary'] = std.get('salary', 0.0)
        std['projection'] = std.get('projection', 0.0)
        std['prop_line'] = std.get('prop_line', 0.0)
        std['spike_score'] = std.get('factor_hit_rate', 0.0)
        
        if source == 'PrizeP
