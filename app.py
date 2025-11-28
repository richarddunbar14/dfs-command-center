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

# 🛡️ SAFE IMPORT FOR AI
try:
    from duckduckgo_search import DDGS
    HAS_AI = True
except ImportError:
    HAS_AI = False

# ==========================================
# ⚙️ 1. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="TITAN OMNI: V44.0", page_icon="🌌")

st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    .titan-card { background: #111; border: 1px solid #333; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #29b6f6; }
    div[data-testid="stDataFrame"] { border: 1px solid #333; border-radius: 5px; }
    .stButton>button { width: 100%; border-radius: 4px; font-weight: 800; text-transform: uppercase; background: linear-gradient(90deg, #111 0%, #222 100%); color: #29b6f6; border: 1px solid #29b6f6; transition: 0.3s; }
    .stButton>button:hover { background: #29b6f6; color: #000; box-shadow: 0 0 15px #29b6f6; }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if 'dfs_raw' not in st.session_state: st.session_state['dfs_raw'] = [] # Store RAW files to re-process dynamically
if 'prop_raw' not in st.session_state: st.session_state['prop_raw'] = []
if 'dfs_pool' not in st.session_state: st.session_state['dfs_pool'] = pd.DataFrame()
if 'prop_pool' not in st.session_state: st.session_state['prop_pool'] = pd.DataFrame()

# ==========================================
# 📡 2. API & DB
# ==========================================
class MultiVerseGateway:
    def __init__(self, api_key): 
        self.headers = {"X-RapidAPI-Key": api_key, "Content-Type": "application/json"}
    
    def fetch_data(self, sport): 
        # Placeholder for real API logic
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def cached_api_fetch(sport, key): return MultiVerseGateway(key).fetch_data(sport)

def init_db():
    conn = sqlite3.connect('titan_multiverse.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS bankroll (id INTEGER PRIMARY KEY, date TEXT, amount REAL, notes TEXT)''')
    c.execute('SELECT count(*) FROM bankroll')
    if c.fetchone()[0] == 0: 
        c.execute("INSERT INTO bankroll (date, amount, notes) VALUES (?, ?, ?)", (datetime.now().strftime("%Y-%m-%d"), 1000.0, 'Genesis'))
        conn.commit()
    return conn

def get_bankroll(conn):
    try: return pd.read_sql("SELECT * FROM bankroll ORDER BY id DESC LIMIT 1", conn).iloc[0]['amount']
    except: return 1000.0

# ==========================================
# 🧠 3. BRAIN INTELLIGENCE
# ==========================================
class TitanBrain:
    def __init__(self, bankroll): self.bankroll = bankroll
    
    def apply_boosts(self, row, sport):
        """Applies situational boosts to projections"""
        proj = float(row.get('projection', 0.0))
        notes = []
        try:
            # Hot Hand
            l3 = float(row.get('l3_fpts', 0))
            avg = float(row.get('avg_fpts', 0))
            if l3 > 0 and avg > 0:
                if (l3 - avg) / avg > 0.2: 
                    proj *= 1.05
                    notes.append("🔥 Hot")
            
            # Weather
            if sport in ['NFL','MLB'] and float(row.get('wind', 0)) > 15:
                pos = str(row.get('position', ''))
                if 'QB' in pos or 'WR' in pos: 
                    proj *= 0.85
                    notes.append("🌪️ Wind")
        except: pass
        return proj, " | ".join(notes)

    def calc_ev(self, slip, book, stake):
        """Calculates expected value based on win probability"""
        if 'Win Prob' not in slip.columns: return 0, 0, 0
        
        probs = slip['Win Prob'].values / 100.0
        legs = len(probs)
        
        # Payout Tables
        if book == 'PrizePicks':
            payouts = {2: 3.0, 3: 5.0, 4: 10.0, 5: 10.0, 6: 25.0}
            payout = payouts.get(legs, 0)
        else:
            payouts = {2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0}
            payout = payouts.get(legs, 0)
            
        win_chance = np.prod(probs)
        ev = (win_chance * payout * stake) - stake
        return win_chance * 100, payout, ev

# ==========================================
# 📂 4. DATA REFINERY (DYNAMIC)
# ==========================================
class DataRefinery:
    @staticmethod
    def clean(val):
        try:
            s = str(val).strip()
            if s.lower() in ['-', '', 'nan', 'none', 'null']: return 0.0
            return float(re.sub(r'[^\d.-]', '', s))
        except: return 0.0

    @staticmethod
    def norm_pos(pos):
        p = str(pos).upper().strip()
        if 'QUARTER' in p: return 'QB'
        if 'RUNNING' in p: return 'RB'
        if 'RECEIVER' in p: return 'WR'
        if 'TIGHT' in p: return 'TE'
        if 'DEF' in p or 'DST' in p: return 'DST'
        if 'GUARD' in p or 'G/UTIL' in p: return 'G'
        if 'FORWARD' in p or 'F/UTIL' in p: return 'F'
        if 'CENTER' in p: return 'C'
        if 'GOALIE' in p: return 'G'
        return p

    @staticmethod
    def ingest(df, sport, site, type="DFS"):
        # Normalize headers
        df.columns = df.columns.astype(str).str.upper().str.strip()
        std = pd.DataFrame()
        
        # Dynamic Salary Mapping based on Site Selection
        sal_col = 'SALARY' # Default fallback
        if site == 'DK': 
            for c in df.columns: 
                if 'DK' in c and 'SAL' in c: sal_col = c; break
        elif site == 'FD':
            for c in df.columns: 
                if 'FD' in c and 'SAL' in c: sal_col = c; break

        # Column Aliases
        maps = {
            'name': ['PLAYER', 'NAME', 'WHO', 'ATHLETE'],
            'projection': ['FPTS', 'PROJ', 'ROTOWIRE PROJECTION', 'AVG FPTS'],
            'salary': [sal_col, 'SAL', 'COST', 'PRICE'], 
            'position': ['POS', 'POSITION', 'ROSTER POSITION'],
            'team': ['TEAM', 'TM'],
            'prop_line': ['LINE', 'PROP', 'TOTAL', 'STRIKE'],
            'market': ['MARKET', 'STAT'],
            'game_info': ['GAME INFO', 'GAME', 'MATCHUP'],
            'status': ['STATUS', 'INJURY', 'AVAILABILITY'],
            'l3_fpts': ['L3', 'LAST 3'],
            'avg_fpts': ['AVG', 'SEASON AVG'],
            'wind': ['WIND']
        }

        for target, sources in maps.items():
            for s in sources:
                if s in df.columns:
                    if target in ['projection','salary','prop_line','l3_fpts','avg_fpts','wind']:
                        std[target] = df[s].apply(DataRefinery.clean)
                    elif target == 'position':
                        std[target] = df[s].apply(DataRefinery.norm_pos)
                    else:
                        std[target] = df[s].astype(str).str.strip()
                    break
        
        # Mandatory Field Check
        if 'name' not in std.columns: return pd.DataFrame()
        
        # Defaults
        if 'projection' not in std.columns: std['projection'] = 0.0
        if 'salary' not in std.columns: std['salary'] = 0.0
        if 'status' not in std.columns: std['status'] = 'Active'
        if 'game_info' not in std.columns: 
            if 'team' in std.columns and 'opp' in std.columns: 
                std['game_info'] = std['team'] + ' vs ' + std.get('opp', 'OPP')
            else: 
                std['game_info'] = 'All Games'
            
        std['sport'] = sport.upper()
        
        return std

    @staticmethod
    def merge(base, new):
        if base.empty: return new
        if new.empty: return base
        combined = pd.concat([base, new])
        
        # Smart Merge: Max for numbers, Last for strings
        agg = {}
        for c in combined.columns:
            if pd.api.types.is_numeric_dtype(combined[c]): agg[c] = 'max'
            else: agg[c] = 'last'
            
        # Group by Name+Sport
        if 'name' in agg: del agg['name']
        if 'sport' in agg: del agg['sport']
        
        try: 
            return combined.groupby(['name', 'sport'], as_index=False).agg(agg)
        except: 
            return combined.drop_duplicates(subset=['name','sport'], keep='last')

# ==========================================
# 🏭 5. OPTIMIZER ENGINE
# ==========================================
def get_rules(sport, site):
    rules = {'size': 6, 'cap': 50000, 'constraints': []}
    
    if site == 'FD': rules['cap'] = 60000
    if site in ['PrizePicks', 'Underdog']: rules['cap'] = 9999999
    
    if sport == 'NFL':
        rules['size'] = 9
        if site == 'DK': 
            rules['constraints'] = [('QB',1,1), ('RB',2,3), ('WR',3,4), ('TE',1,2), ('DST',1,1)]
    elif sport == 'NBA':
        rules['size'] = 8
        if site == 'DK': 
            rules['constraints'] = [('PG',1,3), ('SG',1,3), ('SF',1,3), ('PF',1,3), ('C',1,2)]
    elif sport == 'CBB':
        rules['size'] = 8
        if site == 'DK': 
            rules['constraints'] = [('G',3,5), ('F',3,5)] 
        
    return rules

def run_dfs_opt(pool, config):
    df = pool.copy()
    
    # Brain Boosts
    brain = TitanBrain(0)
    res = df.apply(lambda r: brain.apply_boosts(r, config['sport']), axis=1, result_type='expand')
    df['projection'] = res[0]
    
    # Strict Filter: Salary > 0 (unless PP/UD) & Active Status
    if config['site'] not in ['PrizePicks','Underdog'] and not config.get('ignore_sal'):
        df = df[df['salary'] > 0]
        
    df = df[~df['status'].str.contains('Out|IR|NA|Doubtful', case=False, na=False)]
    
    if df.empty: return None
    
    rules = get_rules(config['sport'], config['site'])
    if config['ignore_sal']: rules['cap'] = 99999999
    
    lineups = []
    exposure = {i:0 for i in df.index}
    
    for i in range(config['count']):
        prob = pulp.LpProblem("Titan", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", df.index, cat='Binary')
        
        # Objective: Proj + Noise (Simulation)
        noise = np.random.normal(0, 0.5, len(df))
        prob += pulp.lpSum([(df.loc[i, 'projection'] + noise[i]) * x[i] for i in df.index])
        
        # Constraints
        prob += pulp.lpSum([df.loc[i, 'salary'] * x[i] for i in df.index]) <= rules['cap']
        prob += pulp.lpSum([x[i] for i in df.index]) == rules['size']
        
        for pos, min_q, max_q in rules['constraints']:
            idx = df[df['position'].str.contains(pos, na=False)].index
            if not idx.empty:
                prob += pulp.lpSum([x[i] for i in idx]) >= min_q
        
        # Stacking (NFL QB+WR)
        if config['stack'] and config['sport'] == 'NFL':
            qbs = df[df['position'] == 'QB'].index
            for qb in qbs:
                team = df.loc[qb, 'team']
                mates = df[(df['team'] == team) & (df['position'].isin(['WR','TE']))].index
                if not mates.empty:
                    prob += pulp.lpSum([x[m] for m in mates]) >= x[qb]

        # Max Exposure Limit (60%)
        limit = max(1, int(config['count'] * 0.6))
        for p in df.index:
            if exposure[p] >= limit: prob += x[p] == 0

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == 1:
            sel = [i for i in df.index if x[i].varValue == 1]
            lu = df.loc[sel].copy()
            lu['Lineup_ID'] = i+1
            lineups.append(lu)
            for p in sel: exposure[p] += 1
            
    return pd.concat(lineups) if lineups else None

def run_prop_opt(pool, config):
    df = pool.copy()
    brain = TitanBrain(0)
    
    try:
        res = df.apply(lambda r: brain.apply_boosts(r, config['sport']), axis=1, result_type='expand')
        df['smart_proj'] = res[0]
    except: 
        df['smart_proj'] = df['projection']
    
    # Calculate Prop Edge
    def score(row):
        line = float(row.get('prop_line', 0))
        if line <= 0: return 0
        return abs(row['smart_proj'] - line) / line
        
    df['score'] = df.apply(score, axis=1)
    df = df[df['score'] > 0.05].sort_values('score', ascending=False) # Only >5% edge
    
    slips = []
    # Greedy Slip Builder
    for i in range(config['count']):
        slip = []
        teams = set()
        used_names = set()
        
        for idx, row in df.iterrows():
            if len(slip) >= config['legs']: break
            if row['name'] in used_names: continue
            
            # Correlation Boost
            val = row['score']
            if config['corr'] and row['team'] in teams: val *= 1.2
            
            line = row['prop_line']
            pick = "OVER" if row['smart_proj'] > line else "UNDER"
            
            # Heuristic Win Prob: 52% baseline + (edge * 50)
            win_prob = min(75, 52 + (val * 50))
            
            slip.append({
                'Player': row['name'], 
                'Market': row.get('market', 'Std'),
                'Line': line, 
                'Pick': pick, 
                'Win Prob': win_prob,
                'Titan Proj': row['smart_proj']
            })
            teams.add(row['team'])
            used_names.add(row['name'])
            
        if len(slip) == config['legs']:
            slips.append(pd.DataFrame(slip))
            # Rotate pool to force variety
            if not df.empty: df = df.iloc[1:] 
            
    return slips

# ==========================================
# 🖥️ DASHBOARD INTERFACE
# ==========================================
conn = init_db()
st.sidebar.title("TITAN OMNI V44")
st.sidebar.caption("Hydra Engine")

current_bank = get_bankroll(conn)
st.sidebar.metric("Bankroll", f"${current_bank:,.2f}")

sport = st.sidebar.selectbox("Sport", ["NBA", "NFL", "MLB", "NHL", "CBB", "PGA"])
site = st.sidebar.selectbox("Platform", ["DK", "FD", "PrizePicks", "Underdog"])

# AUTO-RELOAD LOGIC: 
# When user changes Sport/Site, re-process raw files with new rules (e.g. searching for 'FD Salary')
if 'dfs_raw' in st.session_state and st.session_state['dfs_raw']:
    ref = DataRefinery()
    new_data = pd.DataFrame()
    for raw_df in st.session_state['dfs_raw']:
        new_data = ref.merge(new_data, ref.ingest(raw_df, sport, site, "DFS"))
    st.session_state['dfs_pool'] = new_data

t1, t2, t3 = st.tabs(["1. Data Fusion", "2. Optimizer", "3. Prop Engine"])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        st.success("🏰 DFS Upload (Salaries)")
        f1 = st.file_uploader("DFS CSVs", accept_multiple_files=True, key="dfs")
        if f1:
            raw_frames = [pd.read_csv(f) for f in f1]
            st.session_state['dfs_raw'] = raw_frames # Store RAW
            st.experimental_rerun() # Force reload to process immediately
            
    with c2:
        st.info("🚀 Prop Upload (Lines)")
        f2 = st.file_uploader("Prop CSVs", accept_multiple_files=True, key="prop")
        if f2:
            ref = DataRefinery()
            df = pd.DataFrame()
            for f in f2: 
                df = ref.merge(df, ref.ingest(pd.read_csv(f), sport, site, "Props"))
            st.session_state['prop_pool'] = df
            st.success(f"Loaded {len(df)} Props")

    if st.button("Reset All Data"):
        st.session_state['dfs_pool'] = pd.DataFrame()
        st.session_state['prop_pool'] = pd.DataFrame()
        st.session_state['dfs_raw'] = []
        st.experimental_rerun()

    # Show Data Preview
    if not st.session_state['dfs_pool'].empty:
        st.markdown("### 📊 Active Pool Preview")
        st.dataframe(st.session_state['dfs_pool'].head())

with t2:
    df = st.session_state['dfs_pool']
    if not df.empty:
        # Value Calculation
        df['value_score'] = np.where(df['salary']>0, (df['projection']/df['salary'])*1000, 0)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 💎 Core Value Plays")
            st.dataframe(df.sort_values('value_score', ascending=False).head(5)[['name','salary','projection','value_score']])
        
        st.divider()
        
        c_set1, c_set2 = st.columns(2)
        num = c_set1.slider("Lineups", 1, 100, 10)
        stack = c_set2.checkbox("Smart Stacking (QB+WR)", True)
        ign = c_set2.checkbox("Ignore Salary Cap", False)
        
        if st.button("⚡ GENERATE OPTIMAL LINEUPS"):
            with st.spinner("Solving Matrix..."):
                cfg = {'sport':sport, 'site':site, 'count':num, 'stack':stack, 'ignore_sal':ign}
                res = run_dfs_opt(df, cfg)
                
                if res is not None: 
                    st.success(f"Generated {num} Lineups!")
                    st.dataframe(res)
                    csv = res.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download CSV", csv, "titan_lineups.csv", "text/csv")
                    
                    # Exposure Chart
                    st.bar_chart(res['name'].value_counts())
                else:
                    st.error("Optimization Failed. Try relaxing constraints (Ignore Salary).")

with t3:
    df = st.session_state['prop_pool']
    if not df.empty:
        c1, c2 = st.columns(2)
        legs = c1.slider("Legs", 2, 6, 5)
        wager = c2.number_input("Wager ($)", 20)
        corr = st.checkbox("Boost Correlation", True)
        
        if st.button("🚀 BUILD SLIPS"):
            slips = run_prop_opt(df, {'sport':sport, 'book':site, 'legs':legs, 'count':3, 'corr':corr})
            if slips:
                for i, s in enumerate(slips):
                    brain = TitanBrain(0)
                    win, pay, ev = brain.calc_ev(s, site, wager)
                    
                    with st.expander(f"🎫 Slip #{i+1} | EV: ${ev:.2f} | Win%: {win:.1f}%", expanded=True):
                        st.table(s)
                        if ev > 0: st.success("✅ Positive EV")
                        else: st.error("❌ Negative EV")
            else:
                st.error("No valid slips found. Check data or reduce legs.")
