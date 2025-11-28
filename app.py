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

# 🛡️ SAFE IMPORT FOR AI
try:
    from duckduckgo_search import DDGS
    HAS_AI = True
except ImportError:
    HAS_AI = False

# ==========================================
# ⚙️ 1. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="TITAN OMNI: V38.0", page_icon="🌌")

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    .titan-card { background: #111; border: 1px solid #333; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #29b6f6; }
    div[data-testid="stDataFrame"] { border: 1px solid #333; border-radius: 5px; }
    .stButton>button { width: 100%; border-radius: 4px; font-weight: 800; text-transform: uppercase; background: linear-gradient(90deg, #111 0%, #222 100%); color: #29b6f6; border: 1px solid #29b6f6; transition: 0.3s; }
    .stButton>button:hover { background: #29b6f6; color: #000; box-shadow: 0 0 15px #29b6f6; }
    .metric-value { color: #00e676; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Session State
if 'dfs_pool' not in st.session_state: st.session_state['dfs_pool'] = pd.DataFrame()
if 'prop_pool' not in st.session_state: st.session_state['prop_pool'] = pd.DataFrame()
if 'api_log' not in st.session_state: st.session_state['api_log'] = []

# ==========================================
# 📡 2. API ROUTER (With Mock Fallback)
# ==========================================
class MultiVerseGateway:
    def __init__(self, api_key):
        self.key = api_key
        self.headers = {"X-RapidAPI-Key": self.key, "Content-Type": "application/json"}

    def fetch_data(self, sport):
        data = []
        # If no key provided, generate mock data immediately to prevent crash
        if not self.key:
            return pd.DataFrame(self.generate_mock_data(sport))

        try:
            if sport == 'NBA':
                url = "https://api-nba-v1.p.rapidapi.com/games"
                self.headers["X-RapidAPI-Host"] = "api-nba-v1.p.rapidapi.com"
                res = requests.get(url, headers=self.headers, params={"date": datetime.now().strftime("%Y-%m-%d")})
                if res.status_code == 200:
                    for g in res.json().get('response', []):
                        data.append({'game_info': f"{g['teams']['visitors']['code']} @ {g['teams']['home']['code']}", 'sport': 'NBA'})
            elif sport == 'NFL':
                # Similar logic for NFL...
                pass
        except Exception as e:
            st.error(f"API Error: {e}")
        
        # If API returns empty (no games today or error), return mock data for demo
        if not data:
            return pd.DataFrame(self.generate_mock_data(sport))
        return pd.DataFrame(data)

    def generate_mock_data(self, sport):
        """Generates sample data so the app works out-of-the-box"""
        names = ['Luka Doncic', 'Nikola Jokic', 'Shai Gilgeous-Alexander', 'Giannis Antetokounmpo', 'Jayson Tatum']
        if sport == 'NFL': names = ['Josh Allen', 'Christian McCaffrey', 'Tyreek Hill', 'CeeDee Lamb', 'Lamar Jackson']
        
        mock = []
        for i, n in enumerate(names):
            mock.append({
                'name': n, 'team': 'FA', 'position': 'PG' if sport=='NBA' else 'QB',
                'salary': 11000-(i*500), 'projection': 60-(i*5), 'prop_line': 45-(i*2),
                'sport': sport, 'status': 'Active', 'avg_fpts': 50, 'l3_fpts': 55
            })
        return mock

@st.cache_data(ttl=3600)
def cached_api_fetch(sport, key): return MultiVerseGateway(key).fetch_data(sport)

# ==========================================
# 💾 3. DATABASE
# ==========================================
def init_db():
    # check_same_thread=False fixed for Streamlit Cloud
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

def update_bankroll(conn, amount, note):
    c = conn.cursor()
    c.execute("INSERT INTO bankroll (date, amount, notes) VALUES (?, ?, ?)", (datetime.now().strftime("%Y-%m-%d"), float(amount), note))
    conn.commit()

# ==========================================
# 🧠 4. TITAN BRAIN
# ==========================================
class TitanBrain:
    def __init__(self, bankroll): self.bankroll = bankroll
    
    def apply_strategic_boosts(self, row, sport):
        proj = float(row.get('projection', 0.0))
        notes = []
        try:
            # Hot Hand Logic
            l3 = float(row.get('l3_fpts', 0))
            avg = float(row.get('avg_fpts', 0))
            if l3 > 0 and avg > 0:
                diff = (l3 - avg) / avg
                if diff > 0.2: 
                    proj *= 1.05
                    notes.append("🔥 Hot")
                elif diff < -0.2: 
                    proj *= 0.95
                    notes.append("❄️ Cold")
            
            # Weather
            if sport in ['NFL', 'MLB'] and float(row.get('wind', 0)) > 15:
                pos = str(row.get('position', ''))
                if 'QB' in pos or 'WR' in pos: 
                    proj *= 0.85
                    notes.append("🌪️ Wind Fade")
            
            # Matchup
            rank = float(row.get('opp_rank', 16))
            if rank >= 25: 
                proj *= 1.08
                notes.append("🟢 Elite Matchup")
            elif rank <= 5: 
                proj *= 0.92
                notes.append("🔴 Tough Matchup")
        except: pass
        
        return proj, " | ".join(notes)

    def calculate_slip_ev(self, slip_df, book, stake):
        probs = slip_df['Win Prob %'].values / 100.0
        legs = len(probs)
        payout = 0
        
        # Exact Payout Tables
        if book == 'PrizePicks': 
            payouts = {2:3, 3:5, 4:10, 5:10, 6:25}
            payout = payouts.get(legs, 0)
        elif book == 'Underdog': 
            payouts = {2:3, 3:6, 4:10, 5:20}
            payout = payouts.get(legs, 0)
            
        win_chance = np.prod(probs)
        ev = (win_chance * payout * stake) - stake
        return win_chance * 100, payout, ev

# ==========================================
# 📂 5. DATA REFINERY
# ==========================================
class DataRefinery:
    @staticmethod
    def clean_curr(val):
        try:
            s = str(val).strip()
            if s.lower() in ['-', '', 'nan', 'none', 'null']: return 0.0
            # Remove non-numeric except . and -
            return float(re.sub(r'[^\d.-]', '', s))
        except: return 0.0

    @staticmethod
    def normalize_pos(pos):
        p = str(pos).upper().strip()
        if 'QUARTER' in p: return 'QB'
        if 'RUNNING' in p: return 'RB'
        if 'RECEIVER' in p: return 'WR'
        if 'TIGHT' in p: return 'TE'
        if 'DEF' in p or 'DST' in p: return 'DST'
        if 'GUARD' in p or 'G/UTIL' in p: return 'G'
        if 'FORWARD' in p or 'F/UTIL' in p: return 'F'
        if 'CENTER' in p: return 'C'
        return p

    @staticmethod
    def ingest(df, sport_tag, source_tag="Generic"):
        df.columns = df.columns.astype(str).str.upper().str.strip()
        std = pd.DataFrame()
        
        maps = {
            'name': ['PLAYER','NAME','WHO'], 
            'projection': ['FPTS','PROJ','ROTOWIRE PROJECTION','AVG FPTS'], 
            'salary': ['SAL','SALARY','COST'], 
            'position': ['POS','POSITION'], 
            'team': ['TEAM','TM'], 
            'prop_line': ['LINE','PROP','STRIKE'], 
            'market': ['MARKET','STAT'], 
            'status': ['STATUS','INJURY'],
            'l3_fpts': ['L3','LAST 3'], 
            'avg_fpts': ['AVG','SEASON AVG'], 
            'opp_rank': ['OPP RANK','DVP'],
            'wind': ['WIND']
        }
        
        for target, sources in maps.items():
            for s in sources:
                if s in df.columns:
                    if target in ['projection','salary','prop_line','l3_fpts','avg_fpts','opp_rank','wind']:
                        std[target] = df[s].apply(DataRefinery.clean_curr)
                    elif target == 'position': 
                        std[target] = df[s].apply(DataRefinery.normalize_pos)
                    else: 
                        std[target] = df[s].astype(str).str.strip()
                    break
        
        # Fill missing basics to prevent crashes
        if 'name' not in std.columns: return pd.DataFrame()
        
        defaults = {'projection':0.0, 'salary':0.0, 'prop_line':0.0, 'position':'FLEX', 
                    'team':'FA', 'status':'Active', 'sport': sport_tag.upper()}
        for k,v in defaults.items():
            if k not in std.columns: std[k] = v
            
        # Specific source handling
        if source_tag == 'PrizePicks': std['prizepicks_line'] = std['prop_line']
        elif source_tag == 'Underdog': std['underdog_line'] = std['prop_line']
        
        return std

    @staticmethod
    def merge(base, new_df):
        if base.empty: return new_df
        if new_df.empty: return base
        combined = pd.concat([base, new_df])
        
        # Max for numbers, Last for text
        agg_dict = {}
        for c in combined.columns:
            if pd.api.types.is_numeric_dtype(combined[c]): agg_dict[c] = 'max'
            else: agg_dict[c] = 'last'
            
        try: 
            return combined.groupby(['name','sport'], as_index=False).agg(agg_dict)
        except: 
            return combined.drop_duplicates(subset=['name','sport'], keep='last')

# ==========================================
# 🏭 6. OPTIMIZER
# ==========================================
def get_roster_rules(sport, site, mode):
    rules = {'size': 6, 'cap': 50000, 'constraints': []}
    
    if site == 'FD': rules['cap'] = 60000
    if site in ['PrizePicks', 'Underdog']: rules['cap'] = 9999999

    if mode == 'Showdown':
        rules['size'] = 6 if site == 'DK' else 5
        rules['constraints'] = [('CPT',1,1)] if site == 'DK' else [('MVP',1,1)]
        return rules

    if sport == 'NFL':
        if site == 'DK': rules.update({'size':9, 'constraints':[('QB',1,1),('DST',1,1),('RB',2,3),('WR',3,4),('TE',1,2)]})
        elif site == 'FD': rules.update({'size':9, 'constraints':[('QB',1,1),('DST',1,1),('RB',2,3),('WR',3,4),('TE',1,2)]})
    elif sport == 'NBA':
        if site == 'DK': rules.update({'size':8, 'constraints':[('PG',1,3),('SG',1,3),('SF',1,3),('PF',1,3),('C',1,2)]})
    elif sport == 'CBB':
        rules.update({'size':8, 'constraints':[('G',3,5),('F',3,5)]})
        
    return rules

def optimize_lineup(df, config):
    target_sport = config['sport'].strip().upper()
    pool = df[df['sport'] == target_sport].copy()
    
    if pool.empty: return None
    
    # Brain Boosts
    brain = TitanBrain(0)
    res = pool.apply(lambda r: brain.apply_strategic_boosts(r, target_sport), axis=1, result_type='expand')
    pool['projection'] = res[0]
    pool['notes'] = res[1]
    
    # Pre-Filter
    if not config.get('ignore_salary') and config['site'] not in ['PrizePicks', 'Underdog']:
        pool = pool[pool['salary'] > 0]
    
    # Status Check
    pool = pool[~pool['status'].str.contains('Out|IR|NA|Doubtful', case=False, na=False)]
    
    # Showdown Mode Setup
    if config['mode'] == 'Showdown':
        flex = pool.copy(); flex['pos_id'] = 'FLEX'
        cpt = pool.copy(); cpt['pos_id'] = 'CPT'
        cpt['name'] = cpt['name'] + " (CPT)"
        cpt['projection'] *= 1.5
        cpt['salary'] *= 1.5
        pool = pd.concat([cpt, flex]).reset_index(drop=True)
    else:
        pool['pos_id'] = pool['position']
        
    rules = get_roster_rules(target_sport, config['site'], config['mode'])
    if config.get('ignore_salary'): rules['cap'] = 99999999
    
    lineups = []
    
    for i in range(config['count']):
        prob = pulp.LpProblem("Titan", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("p", pool.index, cat='Binary')
        
        # Noise for variance
        noise = np.random.normal(0, 0.5, len(pool))
        
        # Objective
        prob += pulp.lpSum([(pool.loc[p, 'projection'] + noise[p]) * x[p] for p in pool.index])
        
        # Constraints
        prob += pulp.lpSum([pool.loc[p, 'salary'] * x[p] for p in pool.index]) <= rules['cap']
        prob += pulp.lpSum([x[p] for p in pool.index]) == rules['size']
        
        for role, min_req, max_req in rules['constraints']:
            idx = pool[pool['pos_id'].str.contains(role, na=False)].index
            if not idx.empty:
                prob += pulp.lpSum([x[p] for p in idx]) >= min_req
                prob += pulp.lpSum([x[p] for p in idx]) <= max_req
        
        # Prevent Duplicates
        for l in lineups:
            # Constraint: New lineup must differ from previous by at least 1 player
            prob += pulp.lpSum([x[p] for p in l['index_id']]) <= rules['size'] - 1
            
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == 1:
            sel = [p for p in pool.index if x[p].varValue == 1]
            lu = pool.loc[sel].copy()
            lu['Lineup_ID'] = i+1
            lu['index_id'] = sel
            lineups.append(lu)
            
    return pd.concat(lineups) if lineups else None

def optimize_slips(df, config):
    pool = df[df['sport'] == config['sport'].upper()].copy()
    if pool.empty: return None
    
    line_col = 'prizepicks_line' if config['book'] == 'PrizePicks' else 'underdog_line'
    if line_col not in pool.columns: line_col = 'prop_line'
    
    brain = TitanBrain(0)
    res = pool.apply(lambda r: brain.apply_strategic_boosts(r, config['sport']), axis=1, result_type='expand')
    pool['smart_proj'] = res[0]
    pool['notes'] = res[1]
    
    # Calculate Score
    def score_row(row):
        line = row.get(line_col, 0)
        if line <= 0: return 0
        edge = abs(row['smart_proj'] - line) / line
        return edge
    
    pool['score'] = pool.apply(score_row, axis=1)
    pool = pool[pool['score'] > 0.05].sort_values('score', ascending=False)
    
    slips = []
    for i in range(config['count']):
        slip = []
        players = set()
        teams = set()
        
        for idx, row in pool.iterrows():
            if len(slip) >= config['legs']: break
            if row['name'] in players: continue
            
            # Correlation Boost
            score = row['score']
            if config['corr'] and row['team'] in teams: score *= 1.2
            
            line = row.get(line_col, 0) or row['prop_line']
            pick = "OVER" if row['smart_proj'] > line else "UNDER"
            
            # Heuristic Win Prob
            win_prob = min(75, 52 + (score * 50))
            
            slip.append({
                'Player': row['name'], 
                'Market': row.get('market','Standard'), 
                'Line': line, 
                'Pick': pick, 
                'Win Prob %': win_prob,
                'Titan Proj': row['smart_proj']
            })
            players.add(row['name'])
            teams.add(row['team'])
            
        if len(slip) == config['legs']: 
            slips.append(pd.DataFrame(slip))
            # Rotate pool to diversify
            if not pool.empty: pool = pool.iloc[1:]
            
    return slips

# ==========================================
# 🖥️ 7. UI DASHBOARD
# ==========================================
conn = init_db()
st.sidebar.title("TITAN OMNI V38")
st.sidebar.caption("The Hydra Engine")

# Sidebar
api_key = st.sidebar.text_input("RapidAPI Key (Optional)", type="password")
current_bank = get_bankroll(conn)
st.sidebar.metric("Bankroll", f"${current_bank:,.2f}")
if st.sidebar.button("Reset Bankroll"):
    update_bankroll(conn, 1000, "Reset")

sport = st.sidebar.selectbox("Sport", ["NBA", "NFL", "MLB", "NHL", "CBB"])
site = st.sidebar.selectbox("Platform", ["DK", "FD", "PrizePicks", "Underdog"])

t1, t2, t3, t4 = st.tabs(["1. Data Fusion", "2. Optimizer", "3. Props", "4. Parlay"])

# --- TAB 1: DATA ---
with t1:
    c1, c2 = st.columns(2)
    with c1:
        st.info("DFS Ingestion (Salaries)")
        dfs_files = st.file_uploader("Upload CSVs", accept_multiple_files=True, key="dfs")
        if st.button("Load DFS Data"):
            ref = DataRefinery()
            new_data = pd.DataFrame()
            if dfs_files:
                for f in dfs_files: 
                    new_data = ref.merge(new_data, ref.ingest(pd.read_csv(f), sport, "Generic"))
            else:
                # Mock Data Fallback
                st.warning("No files? Loading Simulation Data...")
                gw = MultiVerseGateway(api_key)
                new_data = gw.fetch_data(sport)
                
            st.session_state['dfs_pool'] = new_data
            st.success(f"Pool Size: {len(new_data)} Players")
            
    with c2:
        st.warning("Prop Ingestion (Lines)")
        prop_files = st.file_uploader("Upload CSVs", accept_multiple_files=True, key="prop")
        if st.button("Load Prop Data"):
            ref = DataRefinery()
            new_data = pd.DataFrame()
            if prop_files:
                for f in prop_files: 
                    new_data = ref.merge(new_data, ref.ingest(pd.read_csv(f), sport, site))
            else:
                st.warning("No files? Loading Simulation Data...")
                gw = MultiVerseGateway(api_key)
                new_data = gw.fetch_data(sport)

            st.session_state['prop_pool'] = new_data
            st.success(f"Pool Size: {len(new_data)} Players")
            
    # AI Scout
    if HAS_AI and st.button("Run AI Web Scout"):
        with st.spinner("Scouting..."):
            with DDGS() as ddgs:
                results = list(ddgs.text(f"{sport} dfs sleepers analysis", max_results=3))
                for r in results:
                    st.caption(f"**{r['title']}**: {r['body']}")

# --- TAB 2: OPTIMIZER ---
with t2:
    pool = st.session_state['dfs_pool']
    if not pool.empty:
        c1, c2, c3 = st.columns(3)
        mode = c1.radio("Mode", ["Classic", "Showdown"], horizontal=True)
        count = c2.slider("Lineups", 1, 50, 10)
        smart = c3.checkbox("Smart Stack", True)
        ignore_sal = st.checkbox("Ignore Salary Cap", False)
        
        if st.button("RUN OPTIMIZER"):
            with st.spinner("Solving..."):
                cfg = {'sport':sport, 'site':site, 'mode':mode, 'count':count, 
                       'smart_stack':smart, 'locks':[], 'bans':[], 'slate_games':[], 
                       'ignore_salary':ignore_sal}
                res = optimize_lineup(pool, cfg)
                
                if res is not None:
                    st.dataframe(res)
                    csv = res.to_csv(index=False).encode('utf-8')
                    st.download_button("Download CSV", csv, "lineups.csv", "text/csv")
                    
                    # Visuals
                    st.bar_chart(res['name'].value_counts())
                else:
                    st.error("Infeasible. Try ignoring salary or reducing constraints.")
    else:
        st.info("Load DFS Data in Tab 1 first.")

# --- TAB 3: PROP SLIPS ---
with t3:
    pool = st.session_state['prop_pool']
    if not pool.empty:
        c1, c2 = st.columns(2)
        legs = c1.slider("Legs", 2, 6, 5)
        wager = c2.number_input("Wager ($)", 20)
        corr = st.checkbox("Correlation Boost", True)
        
        if st.button("BUILD OPTIMAL SLIPS"):
            cfg = {'sport':sport, 'book':site, 'legs':legs, 'count':3, 'corr':corr}
            slips = optimize_slips(pool, cfg)
            
            if slips:
                for i, s in enumerate(slips):
                    brain = TitanBrain(0)
                    prob, payout, profit = brain.calculate_slip_ev(s, site, wager)
                    
                    with st.expander(f"🎫 Slip #{i+1} | EV: ${profit:.2f}", expanded=True):
                        st.table(s)
                        if profit > 0: st.success(f"✅ Positive EV (+${profit:.2f})")
                        else: st.error("❌ Negative EV")
            else:
                st.warning("No valid slips found. Check data.")
    else:
        st.info("Load Prop Data in Tab 1 first.")
