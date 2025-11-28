import streamlit as st
import pandas as pd
import numpy as np
import pulp
import re
import time
import sqlite3
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from itertools import combinations

# 🛡️ SAFE IMPORT FOR AI SCOUT
try:
    from duckduckgo_search import DDGS
    HAS_AI = True
except ImportError:
    HAS_AI = False

# ==========================================
# ⚙️ 1. SYSTEM CONFIGURATION & STYLING
# ==========================================
st.set_page_config(layout="wide", page_title="TITAN OMNI: V38.0", page_icon="🌌")

# Custom CSS for Dark Mode/Neon Theme
st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    
    /* Titan Card Style */
    div.css-1r6slb0.e1tzin5v2 {
        background-color: #111111;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #29b6f6;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #29b6f6;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 4px;
        font-weight: 800;
        text-transform: uppercase;
        background: linear-gradient(90deg, #111 0%, #222 100%);
        color: #29b6f6;
        border: 1px solid #29b6f6;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: #29b6f6;
        color: #000;
        box-shadow: 0 0 15px #29b6f6;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if 'dfs_pool' not in st.session_state: st.session_state['dfs_pool'] = pd.DataFrame()
if 'prop_pool' not in st.session_state: st.session_state['prop_pool'] = pd.DataFrame()
if 'api_log' not in st.session_state: st.session_state['api_log'] = []
if 'ai_intel' not in st.session_state: st.session_state['ai_intel'] = {}

# ==========================================
# 📡 2. MULTI-VERSE API ROUTER
# ==========================================
class MultiVerseGateway:
    def __init__(self, api_key):
        self.key = api_key
        self.headers = {"X-RapidAPI-Key": self.key, "Content-Type": "application/json"}

    def fetch_data(self, sport):
        data = []
        try:
            # --- NBA (API-NBA) ---
            if sport == 'NBA':
                url = "https://api-nba-v1.p.rapidapi.com/games"
                self.headers["X-RapidAPI-Host"] = "api-nba-v1.p.rapidapi.com"
                params = {"date": datetime.now().strftime("%Y-%m-%d")}
                # NOTE: Real API call commented out to prevent crashing without valid paid key
                # res = requests.get(url, headers=self.headers, params=params)
                # if res.status_code == 200: ... 
                # MOCK DATA FOR DEMO
                data = self.generate_mock_data('NBA')
            
            # --- NFL (Tank01) ---
            elif sport == 'NFL':
                # MOCK DATA FOR DEMO
                data = self.generate_mock_data('NFL')

        except Exception as e:
            st.error(f"API Error: {e}")
            
        return pd.DataFrame(data)

    def generate_mock_data(self, sport):
        """Generates fake data if API fails or no key provided"""
        mock = []
        names = ['Luka Doncic', 'Nikola Jokic', 'Shai Gilgeous-Alexander', 'Giannis Antetokounmpo', 'Jayson Tatum'] if sport == 'NBA' else ['Josh Allen', 'Christian McCaffrey', 'Tyreek Hill', 'CeeDee Lamb', 'Lamar Jackson']
        teams = ['DAL', 'DEN', 'OKC', 'MIL', 'BOS'] if sport == 'NBA' else ['BUF', 'SF', 'MIA', 'DAL', 'BAL']
        
        for i, name in enumerate(names):
            mock.append({
                'name': name, 'team': teams[i], 'position': 'PG' if sport=='NBA' else 'QB',
                'salary': 10000 - (i*500), 'projection': 55 - (i*3),
                'prop_line': 45.5 - i, 'sport': sport, 'status': 'Active',
                'avg_fpts': 50, 'l3_fpts': 60 if i % 2 == 0 else 40, # Simulate Hot/Cold
                'opp_rank': (i * 5) + 1, 'wind': 5, 'precip': 0
            })
        return mock

@st.cache_data(ttl=3600)
def cached_api_fetch(sport, key):
    gw = MultiVerseGateway(key)
    return gw.fetch_data(sport)

# ==========================================
# 💾 3. DATABASE (Bankroll Management)
# ==========================================
def init_db():
    conn = sqlite3.connect('titan_multiverse.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS bankroll (id INTEGER PRIMARY KEY, date TEXT, amount REAL, notes TEXT)''')
    c.execute('SELECT count(*) FROM bankroll')
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO bankroll (date, amount, notes) VALUES (?, ?, ?)", (datetime.now().strftime("%Y-%m-%d"), 1000.0, 'Genesis'))
        conn.commit()
    return conn

def get_bankroll(conn):
    try:
        df = pd.read_sql("SELECT * FROM bankroll ORDER BY id DESC LIMIT 1", conn)
        return df.iloc[0]['amount'] if not df.empty else 1000.0
    except: return 1000.0

def update_bankroll(conn, amount, note):
    c = conn.cursor()
    c.execute("INSERT INTO bankroll (date, amount, notes) VALUES (?, ?, ?)", (datetime.now().strftime("%Y-%m-%d"), float(amount), note))
    conn.commit()

# ==========================================
# 🧠 4. TITAN BRAIN (Logic Layer)
# ==========================================
class TitanBrain:
    def __init__(self):
        pass

    def apply_strategic_boosts(self, row, sport):
        """Applies Hot Hand, Weather, and Matchup logic"""
        proj = row.get('projection', 0.0)
        notes = []
        
        try:
            # 1. HOT HAND (Last 3 > Avg by 20%)
            if row.get('l3_fpts', 0) > 0 and row.get('avg_fpts', 0) > 0:
                diff_pct = (row['l3_fpts'] - row['avg_fpts']) / row['avg_fpts']
                if diff_pct > 0.20:
                    proj *= 1.05
                    notes.append("🔥 Hot Form")
                elif diff_pct < -0.20:
                    proj *= 0.95
                    notes.append("❄️ Cold Streak")

            # 2. WEATHER (NFL/MLB)
            if sport in ['NFL', 'MLB'] and 'wind' in row and 'precip' in row:
                if row['wind'] > 15 or row['precip'] > 50:
                    pos = str(row.get('position', ''))
                    if 'QB' in pos or 'WR' in pos or 'TE' in pos:
                        proj *= 0.85 
                        notes.append("🌪️ Weather Fade")
                    elif 'RB' in pos or 'DST' in pos:
                        proj *= 1.05 
                        notes.append("🛡️ Weather Boost")

            # 3. MATCHUP (DvP)
            if 'opp_rank' in row and row['opp_rank'] > 0:
                if row['opp_rank'] >= 25: 
                    proj *= 1.08
                    notes.append("🟢 Elite Matchup")
                elif row['opp_rank'] <= 5: 
                    proj *= 0.92
                    notes.append("🔴 Tough Matchup")
                
        except Exception: pass

        return proj, " | ".join(notes)

    def calculate_prop_edge(self, row, line_col='prop_line'):
        """Calculates Edge and Kelly Criterion Units"""
        line = row.get(line_col, 0)
        if line == 0 or pd.isna(line): line = row.get('prop_line', 0)
        proj = row.get('smart_projection', row.get('projection', 0))
        
        if line <= 0 or proj <= 0: return 0, "No Data", 0.0, "Insufficient Data"
        
        edge_raw = abs(proj - line) / line
        # Heuristic Win Prob: 0% edge = 52%, 15% edge = 60%
        win_prob = min(0.78, 0.52 + (edge_raw * 0.55))
        
        # Kelly Criterion Staking
        odds = 0.909 # -110 implied
        kelly = ((odds * win_prob) - (1 - win_prob)) / odds
        units = max(0, kelly * 0.3) * 100 # Fractional Kelly
        
        rating = "PASS"
        if units > 3.0: rating = "💎 ATOMIC"
        elif units > 1.5: rating = "🟢 NUCLEAR"
        elif units > 0.5: rating = "🟡 KINETIC"
        
        reasoning = []
        if edge_raw > 0.10: reasoning.append(f"🟢 {edge_raw*100:.0f}% Edge")
        if 'notes' in row and row['notes']: reasoning.append(row['notes'])
        
        logic_text = " + ".join(reasoning) if reasoning else "Neutral"
        
        return units, rating, win_prob * 100, logic_text

    def calculate_slip_ev(self, slip_df, book, stake):
        """Calculates Expected Value of a parlay"""
        probs = slip_df['Win Prob %'].values / 100.0
        legs = len(probs)
        payout = 0
        
        # Hardcoded Payout Tables
        if book == 'PrizePicks':
            if legs == 2: payout = 3.0
            elif legs == 3: payout = 5.0
            elif legs == 4: payout = 10.0
            elif legs == 5: payout = 10.0 
            elif legs == 6: payout = 25.0
        else: # Underdog/Standard
            if legs == 2: payout = 3.0
            elif legs == 3: payout = 6.0
            elif legs == 4: payout = 10.0
            elif legs == 5: payout = 20.0
        
        win_chance = np.prod(probs)
        expected_return = (win_chance * payout * stake) - stake
        roi = (expected_return / stake) * 100
        return win_chance * 100, payout, expected_return, roi

# ==========================================
# 📂 5. DATA REFINERY (Ingestion)
# ==========================================
class DataRefinery:
    @staticmethod
    def clean_curr(val):
        try:
            s = str(val).strip()
            if s.lower() in ['-', '', 'nan', 'none']: return 0.0
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
        if 'GUARD' in p: return 'G'
        if 'FORWARD' in p: return 'F'
        if 'CENTER' in p: return 'C'
        return p

    @staticmethod
    def ingest(df, sport_tag, source_tag="Generic"):
        """Maps CSV columns to internal schema"""
        df.columns = df.columns.astype(str).str.upper().str.strip()
        std = pd.DataFrame()
        
        # Column Aliases
        maps = {
            'name': ['PLAYER', 'NAME', 'WHO', 'ATHLETE'],
            'projection': ['ROTOWIRE PROJECTION', 'PROJECTION', 'FPTS', 'PROJ', 'AVG FPTS', 'PTS'], 
            'salary': ['SAL', 'SALARY', 'CAP', 'COST'],
            'position': ['POS', 'POSITION', 'ROSTER POSITION'],
            'team': ['TEAM', 'TM'],
            'prop_line': ['LINE', 'PROP', 'STRIKE', 'TOTAL'], 
            'market': ['MARKET', 'STAT'],
            'status': ['STATUS', 'INJURY', 'AVAILABILITY'], 
            'l3_fpts': ['L3 FPTS', 'LAST 3'],
            'avg_fpts': ['AVG FPTS', 'SEASON AVG'],
            'opp_rank': ['OPP RANK', 'OPP PR', 'DVP'],
            'wind': ['WIND'], 'precip': ['PRECIP', 'RAIN']
        }
        
        # Site Specific Lines
        site_maps = {
            'prizepicks_line': ['PRIZEPICKS LINE', 'PRIZEPICKS'],
            'underdog_line': ['UNDERDOG LINE', 'UNDERDOG'],
            'sleeper_line': ['SLEEPER LINE', 'SLEEPER']
        }
        
        for target, sources in maps.items():
            for s in sources:
                if s in df.columns:
                    if target in ['projection', 'salary', 'prop_line', 'l3_fpts', 'avg_fpts', 'opp_rank', 'wind', 'precip']:
                        std[target] = df[s].apply(DataRefinery.clean_curr)
                    elif target == 'position':
                        std[target] = df[s].apply(DataRefinery.normalize_pos)
                    else:
                        std[target] = df[s].astype(str).str.strip()
                    break
        
        # Site specific logic
        for target, sources in site_maps.items():
            for s in sources:
                if s in df.columns:
                    std[target] = df[s].apply(DataRefinery.clean_curr)
                    break
            if target not in std.columns: std[target] = 0.0

        if 'name' not in std.columns: return pd.DataFrame()
        
        # Defaults
        defaults = {'projection':0.0, 'salary':0.0, 'prop_line':0.0, 'position':'FLEX', 'team':'N/A', 
                   'market':'Standard', 'status':'Active', 'sport': sport_tag}
        for k,v in defaults.items():
            if k not in std.columns: std[k] = v
            
        # Specific source handling
        if source_tag == 'PrizePicks' and std['prizepicks_line'].sum() == 0:
            std['prizepicks_line'] = std['prop_line']
        elif source_tag == 'Underdog' and std['underdog_line'].sum() == 0:
            std['underdog_line'] = std['prop_line']

        if 'salary' in std.columns and 'projection' in std.columns:
            std['value_score'] = np.where(std['salary'] > 0, (std['projection'] / std['salary']) * 1000, 0)

        return std

    @staticmethod
    def merge(base, new_df):
        """Merges multiple uploads into one pool"""
        if base.empty: return new_df
        if new_df.empty: return base
        combined = pd.concat([base, new_df])
        
        # Max Aggregation for numbers, Last for strings
        numeric_cols = ['projection', 'salary', 'prop_line', 'value_score', 'l3_fpts', 'avg_fpts', 'opp_rank', 'wind', 'precip', 'prizepicks_line', 'underdog_line', 'sleeper_line']
        meta_cols = ['position', 'team', 'status']
        
        agg_dict = {col: 'max' for col in numeric_cols if col in combined.columns}
        for col in meta_cols:
            if col in combined.columns: agg_dict[col] = 'last'
            
        try:
            fused = combined.groupby(['name', 'sport', 'market'], as_index=False).agg(agg_dict)
            return fused
        except:
            return combined.drop_duplicates(subset=['name', 'sport', 'market'], keep='last')

# ==========================================
# 🚀 6. AI WEB SCOUT
# ==========================================
def run_web_scout(sport):
    intel = {}
    if HAS_AI:
        try:
            with DDGS() as ddgs:
                queries = [f"{sport} dfs sleepers today", f"{sport} player props analysis"]
                for q in queries:
                    for r in list(ddgs.text(q, max_results=2)):
                        intel[r['title']] = r['body']
        except: pass
    return intel

# ==========================================
# 🏭 7. OPTIMIZER ENGINE (PULP)
# ==========================================
def get_roster_rules(sport, site):
    """Returns constraints based on Sport/Site"""
    rules = {'size': 6, 'cap': 50000, 'constraints': []}
    
    if site == 'DK': rules['cap'] = 50000
    elif site == 'FD': rules['cap'] = 60000
    
    if sport == 'NFL':
        if site == 'DK':
            rules['size'] = 9
            rules['constraints'] = [('QB', 1, 1), ('DST', 1, 1), ('RB', 2, 3), ('WR', 3, 4), ('TE', 1, 2)]
    elif sport == 'NBA':
        if site == 'DK':
            rules['size'] = 8
            rules['constraints'] = [('PG', 1, 3), ('SG', 1, 3), ('SF', 1, 3), ('PF', 1, 3), ('C', 1, 2)]
    
    return rules

def optimize_lineup(df, config):
    """DFS Lineup Builder using Linear Programming"""
    target_sport = config['sport']
    pool = df[df['sport'] == target_sport].copy()
    
    # Pre-Filtering
    if not config['ignore_salary']: pool = pool[pool['salary'] > 0]
    pool = pool[~pool['status'].str.contains('Out|IR|NA|Doubtful', case=False, na=False)]
    
    if pool.empty: return None

    # Brain Boosts
    brain = TitanBrain()
    applied = pool.apply(lambda row: brain.apply_strategic_boosts(row, target_sport), axis=1, result_type='expand')
    pool['projection'] = applied[0]
    pool['notes'] = applied[1]
    
    # Game Script / Chalk Fade logic
    if config['game_script']:
        pool['projection'] *= 1.02 # Simplified boost for demonstration

    pool = pool[pool['projection'] > 0].reset_index(drop=True)
    
    # Constraints Setup
    rules = get_roster_rules(config['sport'], config['site'])
    if config['ignore_salary']: rules['cap'] = 9999999
    
    lineups = []
    
    # Loop for multiple lineups
    for i in range(config['count']):
        prob = pulp.LpProblem("Titan", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("p", pool.index, cat='Binary')
        
        # Objective: Maximize Projection (with noise for simulation)
        noise = np.random.normal(0, 0.5, len(pool)) # Volatility
        prob += pulp.lpSum([(pool.loc[p, 'projection'] + noise[p]) * x[p] for p in pool.index])
        
        # Constraint: Salary
        prob += pulp.lpSum([pool.loc[p, 'salary'] * x[p] for p in pool.index]) <= rules['cap']
        
        # Constraint: Size
        prob += pulp.lpSum([x[p] for p in pool.index]) == rules['size']
        
        # Constraint: Positions
        for role, min_req, max_req in rules['constraints']:
            idx = pool[pool['position'] == role].index
            if not idx.empty:
                prob += pulp.lpSum([x[p] for p in idx]) >= min_req
                prob += pulp.lpSum([x[p] for p in idx]) <= max_req
        
        # Smart Stack (Milly Maker Logic) - NFL QB+WR
        if config['smart_stack'] and config['sport'] == 'NFL':
            qbs = pool[pool['position'] == 'QB'].index
            for qb in qbs:
                team = pool.loc[qb, 'team']
                teammates = pool[(pool['team'] == team) & (pool['position'].isin(['WR', 'TE']))].index
                if not teammates.empty:
                    # If QB selected, select at least 1 teammate
                    prob += pulp.lpSum([x[t] for t in teammates]) >= x[qb]

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == 1:
            sel = [p for p in pool.index if x[p].varValue == 1]
            lu = pool.loc[sel].copy()
            lu['Lineup_ID'] = i+1
            lineups.append(lu)
        
    return pd.concat(lineups) if lineups else None

def optimize_slips(df, config):
    """Prop Slip Builder (Greedy + Correlation)"""
    target_sport = config['sport']
    pool = df[df['sport'] == target_sport].copy()
    
    line_col = 'prizepicks_line' if config['book'] == 'PrizePicks' else 'underdog_line'
    if line_col not in pool.columns: line_col = 'prop_line'
    
    brain = TitanBrain()
    applied = pool.apply(lambda row: brain.apply_strategic_boosts(row, target_sport), axis=1, result_type='expand')
    pool['smart_projection'] = applied[0]
    
    # Calculate Edges
    res = pool.apply(lambda x: brain.calculate_prop_edge(x, line_col), axis=1, result_type='expand')
    pool['units'] = res[0]
    pool['rating'] = res[1]
    pool['win_prob'] = res[2]
    pool['titan_logic'] = res[3]
    pool['pick'] = np.where(pool['smart_projection'] > pool[line_col], "OVER", "UNDER")
    
    # Filter valid plays
    pool = pool[pool['rating'] != 'PASS'].sort_values('units', ascending=False)
    
    slips = []
    
    for i in range(config['count']):
        current_slip = []
        slip_teams = set()
        
        # Greedy Selection
        for idx, row in pool.iterrows():
            if len(current_slip) >= config['legs']: break
            if any(p['name'] == row['name'] for p in current_slip): continue
            
            # Correlation Boost (Same Team)
            score = row['units']
            if config['correlation'] and row['team'] in slip_teams:
                score *= 1.5 
            
            if score > 0:
                play = {
                    'name': row['name'], 'market': row['market'], 'pick': row['pick'], 
                    'line': row[line_col], 'proj': row['smart_projection'],
                    'Win Prob %': row['win_prob'], 'Rating': row['rating']
                }
                current_slip.append(play)
                slip_teams.add(row['team'])
        
        if len(current_slip) == config['legs']:
            slips.append(pd.DataFrame(current_slip))
            # Remove top play to force diversity
            pool = pool.iloc[1:]
            
    return slips

# ==========================================
# 🖥️ 8. UI DASHBOARD (Layout)
# ==========================================
conn = init_db()
st.sidebar.title("TITAN OMNI")
st.sidebar.caption("V38.0 | Hydra Engine")

# Sidebar Controls
api_key_input = st.sidebar.text_input("RapidAPI Key", type="password")
current_bank = get_bankroll(conn)
st.sidebar.metric("🏦 Bankroll", f"${current_bank:,.2f}")
if st.sidebar.button("Update Bankroll"):
    new_bal = st.sidebar.number_input("New Amount", value=current_bank)
    update_bankroll(conn, new_bal, "Manual Update")

sport = st.sidebar.selectbox("Sport", ["NBA", "NFL", "MLB", "NHL", "CBB"])
site = st.sidebar.selectbox("Platform", ["DK", "FD", "PrizePicks", "Underdog"])

# TABS
tabs = st.tabs(["1. 📡 Fusion", "2. 🏰 Optimizer", "3. 🔮 Vision", "4. 🚀 Props", "5. 🧩 Prop Opt"])

# --- TAB 1: DATA FUSION ---
with tabs[0]:
    st.markdown("### 📡 Data Fusion Command")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("☁️ AUTO-SYNC (API)"):
            with st.spinner("Handshaking with Multi-Verse..."):
                data = cached_api_fetch(sport, api_key_input)
                # Process boosts immediately
                brain = TitanBrain()
                data[['projection', 'notes']] = data.apply(lambda row: brain.apply_strategic_boosts(row, sport), axis=1, result_type='expand')
                st.session_state['dfs_pool'] = data
                st.session_state['prop_pool'] = data
                st.success(f"Synced {len(data)} Records.")
                
                # Run AI Scout
                intel = run_web_scout(sport)
                st.session_state['ai_intel'] = intel

    with col2:
        if st.button("🗑️ PURGE SYSTEM"):
            st.session_state['dfs_pool'] = pd.DataFrame()
            st.session_state['prop_pool'] = pd.DataFrame()
            st.warning("Memory Wiped.")

    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("🏰 **DFS UPLOAD** (Salaries)")
        files = st.file_uploader("Upload CSVs", accept_multiple_files=True, key="dfs")
        if files:
            ref = DataRefinery()
            merged = pd.DataFrame()
            for f in files:
                raw = pd.read_csv(f)
                merged = ref.merge(merged, ref.ingest(raw, sport, "Generic"))
            st.session_state['dfs_pool'] = merged
            st.success(f"DFS Pool: {len(merged)} Players")

    with c2:
        st.warning("🚀 **PROP UPLOAD** (Lines)")
        p_files = st.file_uploader("Upload CSVs", accept_multiple_files=True, key="props")
        if p_files:
            ref = DataRefinery()
            merged = pd.DataFrame()
            source = "PrizePicks" if site == "PrizePicks" else "Underdog"
            for f in p_files:
                raw = pd.read_csv(f)
                merged = ref.merge(merged, ref.ingest(raw, sport, source))
            st.session_state['prop_pool'] = merged
            st.success(f"Prop Pool: {len(merged)} Players")
            
    # AI Intel Display
    if st.session_state['ai_intel']:
        with st.expander("🧠 AI Scouting Report", expanded=True):
            for k, v in st.session_state['ai_intel'].items():
                st.markdown(f"**{k}**")
                st.caption(v[:200] + "...")

# --- TAB 2: OPTIMIZER ---
with tabs[1]:
    st.markdown("### 🏰 Lineup Generator")
    pool = st.session_state['dfs_pool']
    
    if pool.empty:
        st.error("No Data. Go to Tab 1.")
    else:
        c1, c2, c3 = st.columns(3)
        count = c1.slider("Lineups", 1, 20, 5)
        smart_stack = c2.checkbox("Smart Stack (QB+WR)", value=True)
        ignore_sal = c3.checkbox("Ignore Cap", value=False)
        
        if st.button("⚡ GENERATE LINEUPS"):
            cfg = {'sport':sport, 'site':site, 'count':count, 'smart_stack':smart_stack, 
                   'game_script':True, 'ignore_salary':ignore_sal, 'locks':[], 'bans':[]}
            res = optimize_lineup(pool, cfg)
            
            if res is not None:
                st.dataframe(res)
                # Download
                csv = res.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", csv, "titan_lineups.csv", "text/csv")
                
                # Analysis
                st.markdown("#### Top Exposure")
                st.bar_chart(res['name'].value_counts().head(10))
            else:
                st.error("Optimization Failed. Check Constraints.")

# --- TAB 3: VISION ---
with tabs[2]:
    st.markdown("### 🔮 Titan Vision Analytics")
    pool = st.session_state['dfs_pool']
    if not pool.empty:
        # Scatter: Salary vs Projection
        fig = px.scatter(pool, x='salary', y='projection', color='position', 
                         hover_data=['name', 'value_score'], title="Value Hunter: Salary vs Projection")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Load data to see visuals.")

# --- TAB 4: PROPS ---
with tabs[3]:
    st.markdown("### 🚀 Prop Analysis")
    pool = st.session_state['prop_pool']
    if not pool.empty:
        brain = TitanBrain()
        line_col = 'prizepicks_line' if site == 'PrizePicks' else 'underdog_line'
        
        # Analyze
        res = pool.apply(lambda x: brain.calculate_prop_edge(x, line_col), axis=1, result_type='expand')
        pool['units'] = res[0]
        pool['rating'] = res[1]
        pool['win_prob'] = res[2]
        pool['titan_logic'] = res[3]
        pool['pick'] = np.where(pool.get('smart_projection', pool['projection']) > pool.get(line_col, 0), "OVER", "UNDER")
        
        valid = pool[pool['rating'] != 'PASS'].sort_values('units', ascending=False)
        
        st.dataframe(valid[['name', 'market', line_col, 'pick', 'win_prob', 'rating', 'titan_logic']])

# --- TAB 5: PROP OPTIMIZER ---
with tabs[4]:
    st.markdown("### 🧩 Slip Builder")
    pool = st.session_state['prop_pool']
    
    if not pool.empty:
        legs = st.slider("Legs", 2, 6, 5)
        corr = st.checkbox("Correlation Boost", value=True)
        wager = st.number_input("Wager", value=20)
        
        if st.button("BUILD SLIPS"):
            cfg = {'sport':sport, 'book':site, 'count':3, 'legs':legs, 'correlation':corr}
            slips = optimize_slips(pool, cfg)
            
            for i, slip in enumerate(slips):
                brain = TitanBrain()
                win_pct, payout, ev, roi = brain.calculate_slip_ev(slip, site, wager)
                
                with st.expander(f"🎫 Slip #{i+1} | EV: ${ev:.2f} | ROI: {roi:.1f}%", expanded=True):
                    st.table(slip[['name', 'market', 'pick', 'line', 'Win Prob %', 'Rating']])
                    if ev > 0: st.success("✅ Positive EV")
                    else: st.error("❌ Negative EV")
