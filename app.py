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

# 🛡️ SAFE IMPORT: Prevents crash if AI module is missing
try:
    from duckduckgo_search import DDGS
    HAS_AI = True
except ImportError:
    HAS_AI = False

# ==========================================
# ⚙️ 1. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="TITAN OMNI: V42.0", page_icon="⚡")

st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    .titan-card { background: #111; border: 1px solid #333; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    div[data-testid="stDataFrame"] { border: 1px solid #333; border-radius: 5px; }
    .stButton>button { width: 100%; background: linear-gradient(90deg, #111 0%, #222 100%); color: #00e676; border: 1px solid #333; }
    .stButton>button:hover { border-color: #00e676; color: #fff; }
</style>""", unsafe_allow_html=True)

# Session State
if 'dfs_pool' not in st.session_state: st.session_state['dfs_pool'] = pd.DataFrame()
if 'prop_pool' not in st.session_state: st.session_state['prop_pool'] = pd.DataFrame()

# ==========================================
# 📡 2. API GATEWAY
# ==========================================
class MultiVerseGateway:
    def __init__(self, api_key):
        self.headers = {"X-RapidAPI-Key": api_key, "Content-Type": "application/json"}
    
    def fetch_data(self, sport):
        data = []
        try:
            if sport == 'NBA':
                res = requests.get("https://api-nba-v1.p.rapidapi.com/games", headers=self.headers, params={"date": datetime.now().strftime("%Y-%m-%d")})
                if res.status_code == 200:
                    for g in res.json().get('response', []):
                        data.append({'game_info': f"{g['teams']['visitors']['code']} @ {g['teams']['home']['code']}"})
            elif sport == 'NFL':
                res = requests.get("https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getDailyGameList", headers=self.headers, params={"gameDate": datetime.now().strftime("%Y%m%d")})
                if res.status_code == 200:
                    for g in res.json().get('body', []):
                        data.append({'game_info': g.get('gameID')})
        except: pass
        return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def cached_api_fetch(sport, key): return MultiVerseGateway(key).fetch_data(sport)

# ==========================================
# 🧠 3. INTELLIGENCE ENGINE
# ==========================================
class TitanBrain:
    def __init__(self, bankroll): self.bankroll = bankroll

    def apply_boosts(self, row, sport):
        proj = row.get('projection', 0.0); notes = []
        try:
            # Hot Hand Logic
            if row.get('l3_fpts', 0) > 0 and row.get('avg_fpts', 0) > 0:
                if (row['l3_fpts'] - row['avg_fpts']) / row['avg_fpts'] > 0.2:
                    proj *= 1.05; notes.append("🔥 Hot")
            # Weather Logic
            if sport in ['NFL', 'MLB'] and row.get('wind', 0) > 15:
                if 'QB' in str(row.get('position')) or 'WR' in str(row.get('position')):
                    proj *= 0.85; notes.append("🌪️ Wind Fade")
        except: pass
        return proj, " | ".join(notes)

    def calc_ev(self, slip, book, stake):
        probs = slip['Win Prob %'].values / 100.0
        legs = len(probs)
        payout = 0
        if book == 'PrizePicks': payout = {2:3, 3:5, 4:10, 5:10, 6:25}.get(legs, 0)
        elif book == 'Underdog': payout = {2:3, 3:6, 4:10, 5:20}.get(legs, 0)
        elif book == 'Sleeper': payout = {2:3, 3:5, 4:10, 5:17}.get(legs, 0) # Approx
        
        win_chance = np.prod(probs)
        ev = (win_chance * payout * stake) - stake
        return win_chance * 100, payout, ev

# ==========================================
# 📂 4. DATA REFINERY
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
        if 'GOALIE' in p: return 'G'
        return p

    @staticmethod
    def ingest(df, source):
        # UNIVERSAL MAPPER
        col_map = {}
        df.columns = df.columns.astype(str).str.upper().str.strip()
        
        # Keywords to look for
        keywords = {
            'name': ['PLAYER', 'NAME', 'WHO'],
            'projection': ['FPTS', 'PROJ', 'ROTOWIRE PROJECTION', 'PTS'],
            'salary': ['SAL', 'COST'],
            'position': ['POS', 'SLOT'],
            'team': ['TEAM', 'TM'],
            'opp': ['OPP', 'VS'],
            'prop_line': ['LINE', 'PROP', 'TOTAL'],
            'game_info': ['GAME', 'MATCHUP'],
            'status': ['STATUS', 'INJURY'],
            'hit_rate': ['HIT RATE', 'L5', 'L10'],
            'wind': ['WIND']
        }
        
        # Auto-Detect Columns
        for col in df.columns:
            for key, patterns in keywords.items():
                if any(p in col for p in patterns):
                    if key not in col_map: col_map[key] = col
        
        # Create Standard DF
        std = pd.DataFrame()
        for key, col in col_map.items():
            if key in ['projection', 'salary', 'prop_line', 'hit_rate', 'wind']:
                std[key] = df[col].apply(DataRefinery.clean)
            elif key == 'position':
                std[key] = df[col].apply(DataRefinery.norm_pos)
            else:
                std[key] = df[col].astype(str)

        # Fill Missing
        req = ['name', 'projection', 'salary', 'prop_line', 'position', 'team', 'game_info']
        for r in req:
            if r not in std.columns: 
                std[r] = 0.0 if r in ['projection', 'salary', 'prop_line'] else 'N/A'
        
        # Site Specific
        if source == 'PrizePicks': std['prizepicks_line'] = std['prop_line']
        elif source == 'Underdog': std['underdog_line'] = std['prop_line']
        
        return std

    @staticmethod
    def merge(base, new):
        if base.empty: return new
        combined = pd.concat([base, new])
        # Smart Merge: Max for numbers, Last for text
        agg = {c: 'max' if pd.api.types.is_numeric_dtype(combined[c]) else 'last' for c in combined.columns}
        if 'name' in agg: del agg['name'] # Group key
        
        try: return combined.groupby(['name', 'team'], as_index=False).agg(agg)
        except: return combined

# ==========================================
# 🏭 5. OPTIMIZER ENGINE
# ==========================================
def get_rules(sport, site, mode):
    # Default
    rules = {'size': 6, 'cap': 50000, 'constraints': []}
    
    if site in ['PrizePicks', 'Underdog', 'Sleeper']: rules['cap'] = 999999
    
    if sport == 'NFL':
        rules['size'] = 9
        if site == 'DK': rules['constraints'] = [('QB',1,1), ('RB',2,3), ('WR',3,4), ('TE',1,2), ('DST',1,1)]
    elif sport == 'NBA':
        rules['size'] = 8
        if site == 'DK': rules['constraints'] = [('PG',1,3), ('SG',1,3), ('SF',1,3), ('PF',1,3), ('C',1,2)]
    elif sport == 'CBB':
        rules['size'] = 8
        if site == 'DK': rules['constraints'] = [('G',3,5), ('F',3,5)]
        
    if mode == 'Showdown':
        rules['size'] = 6
        rules['constraints'] = [('CPT',1,1)]
        
    return rules

def run_optimizer(pool, config):
    # Filtering
    df = pool.copy()
    if not config['ignore_salary']: df = df[df['salary'] > 0]
    if config['slate']: df = df[df['game_info'].isin(config['slate'])]
    
    if df.empty: return None

    # Setup Problem
    prob = pulp.LpProblem("Titan", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("player", df.index, cat='Binary')
    
    # Objective: Maximize Projection + Randomness (Sim)
    noise = np.random.normal(0, 0.5, len(df))
    prob += pulp.lpSum([(df.loc[i, 'projection'] + noise[i]) * x[i] for i in df.index])
    
    # Constraints
    prob += pulp.lpSum([df.loc[i, 'salary'] * x[i] for i in df.index]) <= config['cap']
    prob += pulp.lpSum([x[i] for i in df.index]) == config['size']
    
    # Positional
    for pos, min_q, max_q in config['pos_limits']:
        # Flexible match (e.g. "G" matches "G/UTIL")
        eligible = [i for i in df.index if pos in str(df.loc[i, 'position'])]
        prob += pulp.lpSum([x[i] for i in eligible]) >= min_q
        prob += pulp.lpSum([x[i] for i in eligible]) <= max_q
        
    # Milly Maker Logic (NFL Stacking)
    if config['stack'] and config['sport'] == 'NFL':
        qbs = df[df['position'].str.contains('QB')].index
        for qb in qbs:
            team = df.loc[qb, 'team']
            receivers = df[(df['team'] == team) & (df['position'].str.contains('WR|TE'))].index
            if len(receivers) > 0:
                prob += pulp.lpSum([x[r] for r in receivers]) >= x[qb] # If QB, then >= 1 Receiver

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if prob.status == 1:
        indices = [i for i in df.index if x[i].varValue == 1]
        return df.loc[indices]
    return None

# ==========================================
# 🧩 6. PROP OPTIMIZER
# ==========================================
def run_prop_opt(pool, config):
    df = pool.copy()
    brain = TitanBrain(0)
    
    # Logic Boosts
    res = df.apply(lambda r: brain.apply_strategic_boosts(r, config['sport']), axis=1, result_type='expand')
    df['smart_proj'] = res[0]
    df['notes'] = res[1]
    
    # Score Calculation
    line_col = 'prizepicks_line' if config['book'] == 'PrizePicks' else 'prop_line'
    if line_col not in df.columns: line_col = 'prop_line'
    
    def get_score(row):
        line = row[line_col] if row[line_col] > 0 else row['prop_line']
        if line <= 0: return 0
        edge = abs(row['smart_proj'] - line) / line
        return edge * 100
        
    df['score'] = df.apply(get_score, axis=1)
    df = df.sort_values('score', ascending=False).head(50)
    
    # Generate Slips
    slips = []
    for i in range(config['count']):
        slip = []
        used_teams = set()
        for idx, row in df.iterrows():
            if len(slip) >= config['legs']: break
            
            # Correlation Logic (SGP)
            is_correlated = row['team'] in used_teams
            score = row['score'] * (1.2 if is_correlated and config['corr'] else 1.0)
            
            line = row[line_col] if row[line_col] > 0 else row['prop_line']
            pick = "OVER" if row['smart_proj'] > line else "UNDER"
            
            slip.append({
                'Player': row['name'],
                'Line': line,
                'Pick': pick,
                'Win Prob %': min(70, 50 + score/2)
            })
            used_teams.add(row['team'])
            
        if len(slip) == config['legs']:
            slips.append(pd.DataFrame(slip))
            df = df.iloc[1:] # Rotate players
            
    return slips

# ==========================================
# 🖥️ 7. UI & DASHBOARD
# ==========================================
conn = init_db()
st.sidebar.title("TITAN OMNI V42")
st.sidebar.metric("Bankroll", f"${get_bankroll(conn):,.2f}")

sport = st.sidebar.selectbox("Sport", ["NBA", "NFL", "MLB", "CBB", "NHL", "PGA"])
site = st.sidebar.selectbox("Platform", ["DK", "FD", "PrizePicks", "Underdog"])

t1, t2, t3, t4 = st.tabs(["1. Data", "2. DFS Optimizer", "3. Prop Optimizer", "4. Analysis"])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        st.success("🏰 DFS Upload (Salaries)")
        f1 = st.file_uploader("Drop DFS CSVs", accept_multiple_files=True, key="dfs")
        if st.button("Process DFS"):
            ref = DataRefinery(); merged = pd.DataFrame()
            for f in f1: merged = ref.merge(merged, ref.ingest(pd.read_csv(f), sport, "Generic"))
            st.session_state['dfs_pool'] = merged; st.success(f"Loaded {len(merged)} Players")
            
    with c2:
        st.info("🚀 Prop Upload (Lines)")
        f2 = st.file_uploader("Drop Prop CSVs", accept_multiple_files=True, key="prop")
        if st.button("Process Props"):
            ref = DataRefinery(); merged = pd.DataFrame()
            for f in f2: merged = ref.merge(merged, ref.ingest(pd.read_csv(f), sport, site))
            st.session_state['prop_pool'] = merged; st.success(f"Loaded {len(merged)} Props")

with t2:
    df = st.session_state['dfs_pool']
    if not df.empty:
        mode = st.radio("Mode", ["Classic", "Showdown"], horizontal=True)
        num = st.slider("Lineups", 1, 50, 5)
        stack = st.checkbox("Milly Maker Stacking", True)
        ignore = st.checkbox("Ignore Salary", False)
        
        if st.button("Run Optimizer"):
            rules = DataEngine.get_roster_rules(sport, site) if 'DataEngine' in globals() else get_rules(sport, site, mode)
            if ignore: rules['cap'] = 999999
            
            results = []
            for i in range(num):
                cfg = {'cap': rules['cap'], 'size': rules['size'], 'pos_limits': rules['constraints'], 
                       'stack': stack, 'sport': sport, 'max_exposure': 100, 'slate': None, 'ignore_salary': ignore}
                lu = run_optimizer(df, cfg)
                if lu is not None:
                    lu['Lineup'] = i+1
                    results.append(lu)
            
            if results:
                final = pd.concat(results)
                st.dataframe(final)
                st.download_button("Download CSV", final.to_csv(), "lineups.csv")
            else:
                st.error("Optimization failed. Check constraints.")

with t3:
    df = st.session_state['prop_pool']
    if not df.empty:
        legs = st.slider("Legs", 2, 6, 5)
        wager = st.number_input("Wager", 20)
        corr = st.checkbox("Boost Correlation (SGP)", True)
        
        if st.button("Find Slips"):
            cfg = {'sport': sport, 'book': site, 'legs': legs, 'count': 3, 'corr': corr}
            slips = run_prop_opt(df, cfg)
            
            if slips:
                for i, s in enumerate(slips):
                    brain = TitanBrain(0)
                    win, pay, ev = brain.calculate_slip_ev(s, site, wager)
                    st.markdown(f"**Slip #{i+1}** | EV: ${ev:.2f}")
                    st.table(s)
            else: st.warning("No valid slips found.")
