import streamlit as st
import pandas as pd
import numpy as np
import pulp
import re
import time
import sqlite3
import requests
import plotly.express as px
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="TITAN OMNI V40", page_icon="⚡")
st.markdown("""<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; font-family: monospace; }
    .titan-card { background: #1f2937; border: 1px solid #374151; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    div[data-testid="stDataFrame"] { border: 1px solid #374151; }
    .stButton>button { width: 100%; border-radius: 4px; font-weight: bold; background: #3b82f6; color: white; border: none; }
    .stButton>button:hover { background: #2563eb; }
</style>""", unsafe_allow_html=True)

if 'dfs_pool' not in st.session_state: st.session_state['dfs_pool'] = pd.DataFrame()
if 'prop_pool' not in st.session_state: st.session_state['prop_pool'] = pd.DataFrame()

# --- 1. API ROUTER ---
class MultiVerseGateway:
    def __init__(self, api_key): self.headers = {"X-RapidAPI-Key": api_key, "Content-Type": "application/json"}
    def fetch_data(self, sport):
        data = []
        try:
            if sport == 'NBA':
                res = requests.get("https://api-nba-v1.p.rapidapi.com/games", headers=self.headers, params={"date": datetime.now().strftime("%Y-%m-%d")})
                if res.status_code == 200:
                    for g in res.json().get('response', []): data.append({'game_info': f"{g['teams']['visitors']['code']} @ {g['teams']['home']['code']}"})
            elif sport == 'NFL':
                res = requests.get("https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getDailyGameList", headers=self.headers, params={"gameDate": datetime.now().strftime("%Y%m%d")})
                if res.status_code == 200:
                    for g in res.json().get('body', []): data.append({'game_info': g.get('gameID')})
        except: pass
        return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def cached_api_fetch(sport, key): return MultiVerseGateway(key).fetch_data(sport)

# --- 2. DATA REFINERY ---
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
    def ingest(df, sport_tag, source_tag="Generic"):
        df.columns = df.columns.astype(str).str.upper().str.strip()
        std = pd.DataFrame()
        
        # Universal Mapping
        maps = {
            'name': ['PLAYER', 'NAME', 'WHO'],
            'projection': ['FPTS', 'PROJ', 'ROTOWIRE PROJECTION', 'PTS'],
            'salary': ['SAL', 'SALARY', 'COST'],
            'position': ['POS', 'POSITION', 'ROSTER POSITION'],
            'team': ['TEAM', 'TM'],
            'opp': ['OPP', 'VS'],
            'prop_line': ['LINE', 'PROP', 'TOTAL', 'STRIKE'],
            'market': ['MARKET', 'STAT'],
            'game_info': ['GAME INFO', 'GAME', 'MATCHUP'],
            'status': ['STATUS', 'INJURY'],
            'l3': ['L3', 'LAST 3'], 'avg': ['AVG', 'SEASON AVG'], 'wind': ['WIND']
        }
        
        for target, sources in maps.items():
            for s in sources:
                if s in df.columns:
                    if target in ['projection','salary','prop_line','l3','avg','wind']: std[target] = df[s].apply(DataRefinery.clean)
                    elif target == 'position': std[target] = df[s].apply(DataRefinery.norm_pos)
                    else: std[target] = df[s].astype(str).str.strip()
                    break
        
        if 'name' not in std.columns: return pd.DataFrame()
        
        # Defaults
        defaults = {'projection':0.0, 'salary':0.0, 'prop_line':0.0, 'position':'FLEX', 'team':'N/A', 'game_info':'All Games', 'status':'Active'}
        for k,v in defaults.items(): 
            if k not in std.columns: std[k] = v
            
        # Source Specifics
        if source_tag == 'PrizePicks': std['prizepicks_line'] = std['prop_line']
        elif source_tag == 'Underdog': std['underdog_line'] = std['prop_line']
        
        std['sport'] = sport_tag.upper()
        return std

    @staticmethod
    def merge(base, new):
        if base.empty: return new
        combined = pd.concat([base, new])
        agg = {c: 'max' if pd.api.types.is_numeric_dtype(combined[c]) else 'last' for c in combined.columns}
        if 'name' in agg: del agg['name']
        try: return combined.groupby(['name', 'sport'], as_index=False).agg(agg)
        except: return combined

# --- 3. TITAN BRAIN ---
class TitanBrain:
    def apply_boosts(self, row, sport):
        proj = row.get('projection', 0.0); notes = []
        try:
            # Hot Hand
            if row.get('l3', 0) > 0 and row.get('avg', 0) > 0:
                if (row['l3'] - row['avg']) / row['avg'] > 0.2: 
                    proj *= 1.05; notes.append("🔥 Hot")
            # Weather
            if sport in ['NFL','MLB'] and row.get('wind', 0) > 15:
                if 'QB' in str(row.get('position')): 
                    proj *= 0.85; notes.append("🌪️ Wind Fade")
        except: pass
        return proj, " | ".join(notes)

    def calc_ev(self, slip, book, stake):
        probs = slip['Win Prob'].values / 100.0
        legs = len(probs)
        payout = 3.0 if legs==2 else 5.0 if legs==3 else 10.0
        win_chance = np.prod(probs)
        return win_chance * 100, payout, (win_chance * payout * stake) - stake

# --- 4. OPTIMIZER ENGINES ---
def get_rules(sport, site):
    rules = {'size': 6, 'cap': 50000, 'constraints': []}
    if site in ['PrizePicks', 'Underdog']: rules['cap'] = 999999
    
    if sport == 'NFL':
        rules['size'] = 9
        if site == 'DK': rules['constraints'] = [('QB',1,1), ('RB',2,3), ('WR',3,4), ('TE',1,2), ('DST',1,1)]
    elif sport == 'NBA':
        rules['size'] = 8
        if site == 'DK': rules['constraints'] = [('PG',1,3), ('SG',1,3), ('SF',1,3), ('PF',1,3), ('C',1,2)]
    elif sport == 'CBB':
        rules['size'] = 8
        if site == 'DK': rules['constraints'] = [('G',3,5), ('F',3,5)]
        
    return rules

def run_dfs_opt(pool, config):
    # Filter
    df = pool[pool['salary'] > 0].copy() if not config['ignore_sal'] else pool.copy()
    df = df[~df['status'].str.contains('Out|IR', case=False, na=False)]
    if df.empty: return None
    
    brain = TitanBrain()
    res = df.apply(lambda r: brain.apply_boosts(r, config['sport']), axis=1, result_type='expand')
    df['projection'] = res[0]; df['notes'] = res[1]
    
    rules = get_roster_rules(config['sport'], config['site'])
    lineups = []
    
    for i in range(config['count']):
        prob = pulp.LpProblem("Titan", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", df.index, cat='Binary')
        
        # Sim Noise
        noise = np.random.normal(0, 1.0, len(df))
        prob += pulp.lpSum([(df.loc[i, 'projection'] + noise[i]) * x[i] for i in df.index])
        
        # Constraints
        prob += pulp.lpSum([df.loc[i, 'salary'] * x[i] for i in df.index]) <= rules['cap']
        prob += pulp.lpSum([x[i] for i in df.index]) == rules['size']
        
        for pos, min_q, max_q in rules['constraints']:
            idx = df[df['position'].str.contains(pos, na=False)].index
            prob += pulp.lpSum([x[i] for i in idx]) >= min_q
            
        # Stacking
        if config['stack'] and config['sport'] == 'NFL':
            qbs = df[df['position'] == 'QB'].index
            for qb in qbs:
                team = df.loc[qb, 'team']
                mates = df[(df['team'] == team) & (df['position'].isin(['WR','TE']))].index
                prob += pulp.lpSum([x[m] for m in mates]) >= x[qb]
                
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == 1:
            sel = [i for i in df.index if x[i].varValue == 1]
            lu = df.loc[sel].copy(); lu['Lineup'] = i+1
            lineups.append(lu)
            
    return pd.concat(lineups) if lineups else None

def run_prop_opt(pool, config):
    df = pool.copy()
    line_col = 'prizepicks_line' if config['book'] == 'PrizePicks' else 'prop_line'
    
    def score(row):
        line = row.get(line_col, 0) or row['prop_line']
        if line <= 0: return 0
        return abs(row['projection'] - line) / line
        
    df['score'] = df.apply(score, axis=1)
    df = df.sort_values('score', ascending=False)
    
    slips = []
    for i in range(config['count']):
        slip = []; teams = set()
        for idx, row in df.iterrows():
            if len(slip) >= config['legs']: break
            
            # Correlation Boost
            val = row['score'] * (1.2 if row['team'] in teams and config['corr'] else 1.0)
            
            line = row.get(line_col, 0) or row['prop_line']
            pick = "OVER" if row['projection'] > line else "UNDER"
            
            slip.append({'Player':row['name'], 'Line':line, 'Pick':pick, 'Win Prob': 50 + val*50})
            teams.add(row['team'])
            
        if len(slip) == config['legs']:
            slips.append(pd.DataFrame(slip))
            df = df.iloc[1:] 
            
    return slips

# ==========================================
# 🖥️ DASHBOARD
# ==========================================
# Helper to get rules if not defined above
def get_roster_rules(sport, site):
    return get_rules(sport, site, "Classic")

conn = init_db()
st.sidebar.title("TITAN OMNI V40")
st.sidebar.metric("Bankroll", f"${get_bankroll(conn):,.2f}")

sport = st.sidebar.selectbox("Sport", ["NBA", "NFL", "MLB", "NHL", "CBB"])
site = st.sidebar.selectbox("Site", ["DK", "FD", "PrizePicks", "Underdog"])

t1, t2, t3 = st.tabs(["1. Data Fusion", "2. Optimizer", "3. Prop Engine"])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        st.success("🏰 DFS Upload (Salary)")
        f1 = st.file_uploader("DFS CSVs", accept_multiple_files=True, key="dfs")
        if st.button("Load DFS"):
            ref = DataRefinery(); df = pd.DataFrame()
            for f in f1: df = ref.merge(df, ref.ingest(pd.read_csv(f), sport, "Generic"))
            st.session_state['dfs_pool'] = df; st.success(f"Loaded {len(df)} Players")
            
    with c2:
        st.info("🚀 Prop Upload (Lines)")
        f2 = st.file_uploader("Prop CSVs", accept_multiple_files=True, key="prop")
        if st.button("Load Props"):
            ref = DataRefinery(); df = pd.DataFrame()
            for f in f2: df = ref.merge(df, ref.ingest(pd.read_csv(f), sport, site))
            st.session_state['prop_pool'] = df; st.success(f"Loaded {len(df)} Props")

with t2:
    df = st.session_state['dfs_pool']
    if not df.empty:
        num = st.slider("Lineups", 1, 100, 10)
        stack = st.checkbox("Milly Maker Stacking", True)
        ign = st.checkbox("Ignore Salary", False)
        
        if st.button("Run Optimizer"):
            cfg = {'sport':sport, 'site':site, 'count':num, 'stack':stack, 'ignore_sal':ign, 'slate_games':[], 'locks':[], 'bans':[], 'positions':[], 'max_exposure':100, 'mode':"Classic"}
            res = run_dfs_opt(df, cfg)
            if res is not None: 
                st.dataframe(res)
                st.download_button("Download CSV", res.to_csv(), "lineups.csv")

with t3:
    df = st.session_state['prop_pool']
    if not df.empty:
        legs = st.slider("Legs", 2, 6, 5)
        wager = st.number_input("Wager", 20)
        corr = st.checkbox("Boost Correlation", True)
        
        if st.button("Generate Slips"):
            slips = run_prop_opt(df, {'sport':sport, 'book':site, 'legs':legs, 'count':3, 'corr':corr})
            if slips:
                for i, s in enumerate(slips):
                    brain = TitanBrain(0)
                    win, pay, ev = brain.calculate_slip_ev(s, site, wager)
                    st.write(f"Slip {i+1} | EV: ${ev:.2f}"); st.table(s)
