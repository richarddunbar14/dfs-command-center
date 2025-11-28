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
from duckduckgo_search import DDGS

# ==========================================
# ⚙️ 1. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="TITAN OMNI: V12.0", page_icon="🌌")

st.markdown("""
<style>
    /* THEME: Deep Space & Neon Blue */
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    .titan-card { background: #111; border: 1px solid #333; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #29b6f6; }
    div[data-testid="stDataFrame"] { border: 1px solid #333; border-radius: 5px; }
    .stButton>button { width: 100%; border-radius: 4px; font-weight: 800; text-transform: uppercase; background: linear-gradient(90deg, #111 0%, #222 100%); color: #29b6f6; border: 1px solid #29b6f6; transition: 0.3s; }
    .stButton>button:hover { background: #29b6f6; color: #000; box-shadow: 0 0 15px #29b6f6; }
    .value-badge { background-color: #00e676; color: black; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.8em; }
</style>
""", unsafe_allow_html=True)

# Session State
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
            if sport == 'NBA':
                url = "https://api-nba-v1.p.rapidapi.com/games"
                self.headers["X-RapidAPI-Host"] = "api-nba-v1.p.rapidapi.com"
                params = {"date": datetime.now().strftime("%Y-%m-%d")}
                res = requests.get(url, headers=self.headers, params=params)
                if res.status_code == 200:
                    for g in res.json().get('response', []):
                        data.append({'game': f"{g['teams']['home']['code']} vs {g['teams']['visitors']['code']}", 'sport': 'NBA'})
            elif sport == 'NFL':
                url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getDailyGameList"
                self.headers["X-RapidAPI-Host"] = "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
                params = {"gameDate": datetime.now().strftime("%Y%m%d")}
                res = requests.get(url, headers=self.headers, params=params)
                if res.status_code == 200:
                    for g in res.json().get('body', []):
                        data.append({'game': g.get('gameID'), 'sport': 'NFL'})
        except Exception as e: pass
        return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def cached_api_fetch(sport, key):
    gw = MultiVerseGateway(key)
    return gw.fetch_data(sport)

# ==========================================
# 💾 3. DATABASE
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

def get_bankroll_history(conn):
    try: return pd.read_sql("SELECT * FROM bankroll", conn)
    except: return pd.DataFrame({'date': [datetime.now().strftime("%Y-%m-%d")], 'amount': [1000.0]})

def update_bankroll(conn, amount, note):
    c = conn.cursor()
    c.execute("INSERT INTO bankroll (date, amount, notes) VALUES (?, ?, ?)", (datetime.now().strftime("%Y-%m-%d"), float(amount), note))
    conn.commit()

# ==========================================
# 🧠 4. TITAN BRAIN (ANALYSIS ENGINE)
# ==========================================
class TitanBrain:
    def __init__(self, bankroll):
        self.bankroll = bankroll

    def calculate_prop_edge(self, row, line_col='prop_line'):
        line = row.get(line_col, 0)
        if line == 0 or pd.isna(line): line = row.get('prop_line', 0)
        proj = row.get('projection', 0)
        
        if line <= 0 or proj <= 0: return 0, "No Data", 0.0, "Insufficient Data"
        
        edge_raw = abs(proj - line) / line
        win_prob = min(0.78, 0.52 + (edge_raw * 0.55))
        
        odds = 0.909
        kelly = ((odds * win_prob) - (1 - win_prob)) / odds
        units = max(0, kelly * 0.3) * 100 
        
        rating = "PASS"
        if units > 3.0: rating = "💎 ATOMIC"
        elif units > 1.5: rating = "🟢 NUCLEAR"
        elif units > 0.5: rating = "🟡 KINETIC"
        
        reasoning = []
        if edge_raw > 0.10: reasoning.append(f"🟢 High Edge ({edge_raw*100:.1f}%)")
        if row.get('factor_hit_rate', 0) > 55: reasoning.append(f"🔥 Hot Trend ({row['factor_hit_rate']}%)")
        if row.get('spike_score', 0) > 75: reasoning.append(f"⚡ High Spike Score")
        
        logic_text = " + ".join(reasoning) if reasoning else "Neutral Value"
        
        return units, rating, win_prob * 100, logic_text

    def analyze_lineup(self, lineup_df, sport, slate_size):
        msg = []
        stacks = lineup_df['team'].value_counts()
        heavy_stack = stacks[stacks >= 2].index.tolist()
        
        if heavy_stack:
            msg.append(f"🔗 **Stacked:** {', '.join(heavy_stack)} (Milly Maker Style)")
        else:
            msg.append(f"🧩 **Scatter:** Low Correlation")
            
        avg_proj = lineup_df['projection'].mean()
        msg.append(f"📊 **Proj:** {avg_proj:.1f}")
        
        # Slate Context Feedback
        if slate_size <= 4:
            if len(heavy_stack) < 1: msg.append("⚠️ **Warning:** Small Slate requires more correlation!")
        
        salary_used = lineup_df['salary'].sum()
        msg.append(f"💰 **Cap:** ${salary_used:,}")
        
        return " | ".join(msg)

# ==========================================
# 📂 5. DATA REFINERY (INTELLIGENT INGESTION)
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
        df.columns = df.columns.astype(str).str.upper().str.strip()
        std = pd.DataFrame()
        
        # MAPPINGS
        maps = {
            'name': ['PLAYER', 'NAME', 'WHO', 'ATHLETE'],
            'projection': ['ROTOWIRE PROJECTION', 'PROJECTION', 'FPTS', 'PROJ', 'AVG FPTS', 'PTS'], 
            'salary': ['SAL', 'SALARY', 'CAP', 'COST', 'PRICE'],
            'position': ['POS', 'POSITION', 'ROSTER POSITION', 'SLOT'],
            'team': ['TEAM', 'TM', 'SQUAD'],
            'prop_line': ['LINE', 'PROP', 'PLAYER PROP', 'STRIKE', 'TOTAL'], 
            'market': ['MARKET NAME', 'MARKET', 'STAT'],
            'game_info': ['GAME INFO', 'GAME', 'MATCHUP', 'OPPONENT'],
            'date': ['DATE', 'GAME DATE'],
            'time': ['TIME', 'GAME TIME', 'TIME (ET)'],
            'status': ['STATUS', 'INJURY', 'AVAILABILITY'], 
            'tm_score': ['TM SCORE', 'IMPLIED TOTAL', 'TEAM TOTAL'], # New for Game Script
            'game_total': ['O/U', 'GAME TOTAL', 'OVER/UNDER'], # New for Game Script
            'factor_pickem': ["DFS PICK'EM SITES FACTOR", "PICK'EM FACTOR"],
            'factor_sportsbook': ["SPORTSBOOKS FACTOR", "BOOKS FACTOR"],
            'factor_hit_rate': ["HIT RATE FACTOR", "HIT RATE"],
            'factor_proj': ["ROTOWIRE PROJECTION FACTOR", "PROJ FACTOR"]
        }
        
        site_maps = {
            'prizepicks_line': ['PRIZEPICKS LINE', 'PRIZEPICKS'],
            'underdog_line': ['UNDERDOG LINE', 'UNDERDOG'],
            'sleeper_line': ['SLEEPER LINE', 'SLEEPER'],
            'pick6_line': ['DRAFTKINGS PICK6 LINE', 'PICK6 LINE', 'PICK6']
        }
        
        for target, sources in maps.items():
            for s in sources:
                if s in df.columns:
                    if target in ['projection', 'salary', 'prop_line', 'tm_score', 'game_total', 'factor_pickem', 'factor_sportsbook', 'factor_hit_rate', 'factor_proj']:
                        std[target] = df[s].apply(DataRefinery.clean_curr)
                    elif target == 'name':
                        std[target] = df[s].astype(str).apply(lambda x: x.split('(')[0].strip())
                    elif target == 'position':
                        std[target] = df[s].apply(DataRefinery.normalize_pos)
                    else:
                        std[target] = df[s].astype(str).str.strip()
                    break
        
        for target, sources in site_maps.items():
            for s in sources:
                if s in df.columns:
                    std[target] = df[s].apply(DataRefinery.clean_curr)
                    break
            if target not in std.columns: std[target] = 0.0

        if 'name' not in std.columns: return pd.DataFrame()
        
        defaults = {'projection':0.0, 'salary':0.0, 'prop_line':0.0, 'position':'FLEX', 'team':'N/A', 
                   'market':'Standard', 'game_info':'All Games', 'date':datetime.now().strftime("%Y-%m-%d"), 'time':'TBD', 'status':'Active'}
        for k,v in defaults.items():
            if k not in std.columns: std[k] = v
        
        if 'SPORT' in df.columns: std['sport'] = df['SPORT'].str.strip().str.upper()
        else: std['sport'] = sport_tag.upper()
        
        if 'salary' in std.columns and 'projection' in std.columns:
            std['value_score'] = np.where(std['salary'] > 0, (std['projection'] / std['salary']) * 1000, 0)
        
        factors = ['factor_pickem', 'factor_sportsbook', 'factor_hit_rate', 'factor_proj']
        for f in factors:
            if f in std.columns: std[f] = pd.to_numeric(std[f], errors='coerce')
        std['spike_score'] = std[factors].mean(axis=1).fillna(0)
        
        if source_tag == 'PrizePicks' and std['prizepicks_line'].sum() == 0:
            std['prizepicks_line'] = std['prop_line']
        elif source_tag == 'Underdog' and std['underdog_line'].sum() == 0:
            std['underdog_line'] = std['prop_line']
        elif source_tag == 'Sleeper' and std['sleeper_line'].sum() == 0:
            std['sleeper_line'] = std['prop_line']
        elif source_tag == 'DraftKings Pick6' and std['pick6_line'].sum() == 0:
            std['pick6_line'] = std['prop_line']
            
        return std

    @staticmethod
    def merge(base, new_df):
        if base.empty: return new_df
        if new_df.empty: return base
        combined = pd.concat([base, new_df])
        
        numeric_cols = [
            'projection', 'salary', 'prop_line', 'value_score', 'tm_score', 'game_total',
            'prizepicks_line', 'underdog_line', 'sleeper_line', 'pick6_line',
            'factor_pickem', 'factor_sportsbook', 'factor_hit_rate', 'factor_proj', 'spike_score'
        ]
        meta_cols = ['position', 'team', 'game_info', 'date', 'time', 'status']
        
        agg_dict = {col: 'max' for col in numeric_cols if col in combined.columns}
        for col in meta_cols:
            if col in combined.columns: agg_dict[col] = 'last'
            
        try:
            fused = combined.groupby(['name', 'sport', 'market'], as_index=False).agg(agg_dict)
            return fused
        except:
            return combined.drop_duplicates(subset=['name', 'sport', 'market'], keep='last')

# ==========================================
# 📡 6. AI SCOUT
# ==========================================
def run_web_scout(sport):
    intel = {}
    try:
        with DDGS() as ddgs:
            q = f"{sport} dfs winning strategy correlation stacks today"
            for r in list(ddgs.text(q, max_results=5)):
                intel[(r['title'] + " " + r['body']).lower()] = 1
    except: pass
    return intel

# ==========================================
# 🏭 7. OPTIMIZER (SLATE INTELLIGENCE)
# ==========================================
def get_roster_rules(sport, site, mode):
    rules = {'size': 0, 'cap': 50000, 'constraints': []}
    
    if site in ['PrizePicks', 'Underdog', 'Sleeper', 'DraftKings Pick6']:
        rules['cap'] = 999999
    elif site == 'DK': rules['cap'] = 50000
    elif site == 'FD': rules['cap'] = 60000
    elif site == 'Yahoo': rules['cap'] = 200
    
    if mode == 'Showdown':
        if site == 'DK':
            rules['size'] = 6
            rules['constraints'].append(('CPT', 1, 1))
        elif site == 'FD':
            rules['size'] = 5
            rules['constraints'].append(('MVP', 1, 1))
        return rules

    if sport == 'NFL':
        if site == 'DK':
            rules['size'] = 9
            rules['constraints'] = [('QB', 1, 1), ('DST', 1, 1), ('RB', 2, 3), ('WR', 3, 4), ('TE', 1, 2)]
        elif site == 'FD':
            rules['size'] = 9
            rules['constraints'] = [('QB', 1, 1), ('DST', 1, 1), ('RB', 2, 3), ('WR', 3, 4), ('TE', 1, 2)]
            
    elif sport == 'NBA':
        if site == 'DK':
            rules['size'] = 8
            rules['constraints'] = [('PG', 1, 3), ('SG', 1, 3), ('SF', 1, 3), ('PF', 1, 3), ('C', 1, 2)]
        elif site == 'FD':
            rules['size'] = 9
            rules['constraints'] = [('PG', 2, 2), ('SG', 2, 2), ('SF', 2, 2), ('PF', 2, 2), ('C', 1, 1)]

    if rules['size'] == 0: rules['size'] = 6
    return rules

def optimize_lineup(df, config):
    target_sport = config['sport'].strip().upper()
    pool = df[df['sport'].str.strip().str.upper() == target_sport].copy()
    
    # 🩹 INJURY FILTER
    if 'status' in pool.columns:
        pool = pool[~pool['status'].str.contains('Out|IR|NA|Doubtful', case=False, na=False)]
    
    if pool.empty:
        st.error(f"❌ Pool is empty for {target_sport}.")
        return None

    pickem_sites = ['PrizePicks', 'Underdog', 'Sleeper', 'DraftKings Pick6']
    if config['site'] not in pickem_sites and pool['salary'].sum() == 0:
        st.error("⚠️ No Salary Data found. Cannot run DFS Optimizer.")
        return None

    # SLATE CONTEXT
    unique_games = pool['game_info'].nunique()
    is_small_slate = unique_games <= 4
    st.info(f"📊 **Slate Context:** {unique_games} Games detected. Strategy: {'Aggressive/Hyper-Correlated' if is_small_slate else 'Balanced/Standard'}")

    # GAME SCRIPT ADJUSTMENT (Boost players in high total games)
    if 'tm_score' in pool.columns and config['game_script']:
        pool['projection'] = np.where(pool['tm_score'] > 24, pool['projection'] * 1.05, pool['projection'])

    if config['slate_games']:
        pool = pool[pool['game_info'].isin(config['slate_games'])].reset_index(drop=True)
    
    if config['positions']:
        pool = pool[pool['position'].isin(config['positions'])].reset_index(drop=True)

    pool = pool[pool['projection'] > 0].reset_index(drop=True)
    if config['bans']: pool = pool[~pool['name'].isin(config['bans'])].reset_index(drop=True)
    
    if config['mode'] == 'Showdown':
        flex = pool.copy(); flex['pos_id'] = 'FLEX'
        cpt = pool.copy(); cpt['pos_id'] = 'CPT'
        cpt['name'] += " (CPT)"
        cpt['projection'] *= 1.5
        if config['site'] == 'DK': cpt['salary'] *= 1.5
        pool = pd.concat([cpt, flex]).reset_index(drop=True)
    else:
        pool['pos_id'] = pool['position'].replace({'D': 'DST', 'DEF': 'DST'})

    rules = get_roster_rules(config['sport'], config['site'], config['mode'])
    lineups = []
    player_exposure = {i: 0 for i in pool.index}
    
    for i in range(config['count']):
        prob = pulp.LpProblem("Titan", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("p", pool.index, cat='Binary')
        
        # HISTORICAL SIMULATION WEIGHTING
        # If player has high hit rate, use less variance (reliable). If low hit rate, more variance (boom/bust).
        sim_noise = 0.5
        if 'factor_hit_rate' in pool.columns:
            # Scale noise: High Hit Rate -> Low Noise (0.1), Low Hit Rate -> High Noise (0.9)
            sim_noise = 1.0 - (pool['factor_hit_rate'].fillna(50) / 100.0)
        
        randomness = np.random.uniform(0, sim_noise, len(pool))
        
        prob += pulp.lpSum([(pool.loc[p, 'projection'] + randomness[p]) * x[p] for p in pool.index])
        
        prob += pulp.lpSum([pool.loc[p, 'salary'] * x[p] for p in pool.index]) <= rules['cap']
        prob += pulp.lpSum([x[p] for p in pool.index]) == rules['size']
        
        for role, min_req, max_req in rules['constraints']:
            idx = pool[pool['pos_id'].str.contains(role, regex=True, na=False)].index
            if not idx.empty:
                prob += pulp.lpSum([x[p] for p in idx]) >= min_req
                prob += pulp.lpSum([x[p] for p in idx]) <= max_req
        
        for lock in config['locks']:
            l_idx = pool[pool['name'].str.contains(lock, regex=False)].index
            if not l_idx.empty: prob += pulp.lpSum([x[p] for p in l_idx]) >= 1

        # 🧠 SMART STACKING (SLATE DEPENDENT)
        if config['smart_stack'] and config['sport'] == 'NFL':
            qbs = pool[pool['pos_id'] == 'QB'].index
            for qb in qbs:
                team = pool.loc[qb, 'team']
                teammates = pool[(pool['team'] == team) & (pool['pos_id'].isin(['WR', 'TE']))].index
                
                # Rule 1: Always pair QB with at least 1 WR/TE
                prob += pulp.lpSum([x[t] for t in teammates]) >= x[qb]
                
                # Rule 2: If Small Slate, force Hyper-Correlation (QB + 2 WR/TE)
                if is_small_slate:
                     prob += pulp.lpSum([x[t] for t in teammates]) >= 2 * x[qb]

        if config['max_exposure'] < 100:
            max_lineups = max(1, int(config['count'] * (config['max_exposure'] / 100.0)))
            for p_idx in pool.index:
                if player_exposure[p_idx] >= max_lineups:
                    prob += x[p_idx] == 0

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == 1:
            sel = [p for p in pool.index if x[p].varValue == 1]
            lu = pool.loc[sel].copy()
            lu['Lineup_ID'] = i+1
            lineups.append(lu)
            for p in sel: player_exposure[p] += 1
            prob += pulp.lpSum([x[p] for p in sel]) <= rules['size'] - 1
            
    return pd.concat(lineups) if lineups else None

def get_html_report(df):
    html = f"""<html><body><h2>TITAN HYDRA REPORT</h2><hr>"""
    for _, row in df.head(20).iterrows():
        html += f"<div><b>{row['name']}</b> ({row['position']}) | Sal: ${row['salary']} | Proj: {row['projection']:.1f}</div>"
    return base64.b64encode(html.encode()).decode()

def get_csv_download(df):
    return df.to_csv(index=False).encode('utf-8')

# ==========================================
# 🖥️ 8. DASHBOARD
# ==========================================
conn = init_db()
st.sidebar.title("TITAN OMNI")
st.sidebar.caption("Hydra Edition 12.0 (Slate Intelligence)")

try: API_KEY = st.secrets["rapid_api_key"]
except: API_KEY = st.sidebar.text_input("Enter RapidAPI Key", type="password")

current_bank = get_bankroll(conn)
st.sidebar.metric("🏦 Bankroll", f"${current_bank:,.2f}")

sport = st.sidebar.selectbox("Sport", ["NBA", "NFL", "MLB", "CFB", "NHL", "SOCCER", "PGA", "TENNIS", "CS2"])
site = st.sidebar.selectbox("Site/Book", ["DK", "FD", "Yahoo", "PrizePicks", "Underdog", "Sleeper", "DraftKings Pick6"])

tabs = st.tabs(["1. 📡 Fusion", "2. 🏰 Optimizer", "3. 🔮 Simulation", "4. 🚀 Props", "5. 🧮 Parlay"])

# --- TAB 1: DATA ---
with tabs[0]:
    st.markdown("### 📡 Data Fusion")
    col_api, col_file = st.columns(2)
    with col_api:
        if st.button("☁️ AUTO-SYNC APIS"):
             if not API_KEY: st.error("No API Key Provided")
             else:
                 with st.spinner("Syncing..."):
                    cached_api_fetch(sport, API_KEY)
                    st.success("Sync Complete")
    with col_file:
        source_tag = st.selectbox("🏷️ Select Source for Upload:", ["Generic/Combined", "PrizePicks", "Underdog", "Sleeper", "DraftKings Pick6"])
        files = st.file_uploader(f"Upload {source_tag} CSVs", accept_multiple_files=True)
        
        if st.button("🧬 Fuse Files"):
            if files:
                ref = DataRefinery()
                new_data = pd.DataFrame()
                for f in files:
                    try:
                        try: raw = pd.read_csv(f, encoding='utf-8-sig')
                        except: raw = pd.read_excel(f)
                        st.caption(f"Ingesting {f.name} as {source_tag}...")
                        new_data = ref.merge(new_data, ref.ingest(raw, sport, source_tag))
                    except Exception as e: st.error(f"Error {f.name}: {e}")
                
                st.session_state['dfs_pool'] = ref.merge(st.session_state['dfs_pool'], new_data)
                st.session_state['prop_pool'] = ref.merge(st.session_state['prop_pool'], new_data)
                st.success(f"Fused {len(new_data)} records. Tagged as {source_tag}.")
    
    if st.button("🛰️ Run AI Scout"):
        st.session_state['ai_intel'] = run_web_scout(sport)
        st.success("AI Intel Gathered")

# --- TAB 2: OPTIMIZER ---
with tabs[1]:
    st.markdown("### 🏰 Lineup Generator")
    pool = st.session_state['dfs_pool']
    active = pool[pool['sport'].str.strip().str.upper() == sport.strip().upper()] if not pool.empty else pd.DataFrame()
    
    if active.empty:
        st.warning(f"No data for {sport}. Please Upload CSV.")
    else:
        # 🟢 BEST PLAYS (VALUE ENGINE)
        if 'value_score' in active.columns:
            top_value = active.sort_values('value_score', ascending=False).head(6)
            st.markdown("##### 💎 Core Plays (Best Value/Salary)")
            cols = st.columns(6)
            for i, (idx, row) in enumerate(top_value.iterrows()):
                cols[i].metric(row['name'], f"${row['salary']}", f"{row['value_score']:.1f}x")

        games = sorted(active['game_info'].astype(str).unique())
        slate = st.multiselect("🗓️ Filter Slate", games, default=games)
        
        c1, c2, c3 = st.columns(3)
        mode = c1.radio("Mode", ["Classic", "Showdown"])
        count = c2.slider("Lineups", 1, 50, 10)
        
        # 🟢 STRATEGY PANEL
        with st.expander("🧠 Milly Maker Strategy", expanded=True):
            smart_stack = st.checkbox("Milly Maker Logic (Correlation + Bring Back)", value=True)
            game_script = st.checkbox("Game Script Boost (High Totals)", value=True)
            max_exposure = st.slider("Max Player Exposure %", 10, 100, 60)
            locks = st.multiselect("Lock Players", sorted(active['name'].unique()))
            pos_filter = st.multiselect("Filter Positions", sorted(active['position'].unique()))
        
        if st.button("⚡ Generate & Analyze"):
            cfg = {
                'sport':sport, 'site':site, 'mode':mode, 'count':count, 'locks':locks, 'bans':[], 
                'slate_games':slate, 'sim':False, 'positions':pos_filter,
                'smart_stack': smart_stack, 'game_script': game_script, 'max_exposure': max_exposure
            }
            res = optimize_lineup(st.session_state['dfs_pool'], cfg)
            
            if res is not None:
                st.dataframe(res)
                top_lu = res[res['Lineup_ID']==1]
                brain = TitanBrain(current_bank)
                slate_size = len(slate) if slate else len(games)
                feedback = brain.analyze_lineup(top_lu, sport, slate_size)
                st.info(f"💡 **Lineup 1 Analysis:** {feedback}")
                
                csv_data = get_csv_download(res)
                st.download_button("📥 Export CSV", data=csv_data, file_name="titan_lineups.csv", mime="text/csv")
            else:
                st.error("Optimization Failed. Try loosening exposure limits.")

# --- TAB 3: SIMULATION ---
with tabs[2]:
    st.markdown("### 🔮 Monte Carlo Simulation")
    if st.button("🎲 Run Simulation (50 Lineups)"):
        with st.spinner("Simulating..."):
            cfg = {'sport':sport, 'site':site, 'mode':"Classic", 'count':50, 'locks':[], 'bans':[], 'stack':True, 'sim':True, 'slate_games':[], 'positions':[], 'smart_stack':False, 'game_script':False, 'max_exposure':100}
            res = optimize_lineup(st.session_state['dfs_pool'], cfg)
            if res is not None:
                exp = res['name'].value_counts(normalize=True).mul(100).reset_index()
                st.plotly_chart(px.bar(exp.head(15), x='proportion', y='name', orientation='h', title="Simulated Exposure"))

# --- TAB 4: PROPS ---
with tabs[3]:
    st.markdown("### 🚀 Prop Analyzer")
    pool = st.session_state['prop_pool']
    active = pool[pool['sport'].str.strip().str.upper() == sport.strip().upper()].copy() if not pool.empty else pd.DataFrame()
    
    if active.empty: st.warning(f"No props found for {sport}.")
    else:
        brain = TitanBrain(current_bank)
        target_line_col = 'prop_line'
        if site == 'Underdog': target_line_col = 'underdog_line'
        elif site == 'Sleeper': target_line_col = 'sleeper_line'
        elif site == 'PrizePicks': target_line_col = 'prizepicks_line'
        elif site == 'DraftKings Pick6': target_line_col = 'pick6_line'
        
        st.info(f"🔎 Analyzing for **{site}**. Using column: `{target_line_col}`.")
        
        active['final_line'] = active.apply(lambda x: x[target_line_col] if x.get(target_line_col, 0) > 0 else x['prop_line'], axis=1)
        active['pick'] = np.where(active['projection'] > active['final_line'], "OVER", "UNDER")
        
        res = active.apply(lambda x: brain.calculate_prop_edge(x, 'final_line'), axis=1, result_type='expand')
        active['units'] = res[0]
        active['rating'] = res[1]
        active['win_prob'] = res[2]
        active['titan_logic'] = res[3]
        
        valid = active[(active['projection'] > 0) & (active['final_line'] > 0)].copy()
        
        show_spikes = st.checkbox("🔥 Show Only 'Spiked' Players (High Factor Score)")
        if show_spikes:
            valid = valid[valid['spike_score'] > 75]
        
        cols = ['date', 'time', 'name', 'market', 'team', 'final_line', 'projection', 'pick', 'win_prob', 'rating', 'titan_logic', 'spike_score']
        final_cols = [c for c in cols if c in valid.columns]
        
        st.dataframe(
            valid[final_cols]
            .sort_values('win_prob', ascending=False)
            .style.format({'final_line':'{:.1f}', 'projection':'{:.1f}', 'win_prob':'{:.1f}%', 'spike_score':'{:.0f}'})
        )

# --- TAB 5: PARLAY ---
with tabs[4]:
    st.markdown("### 🧮 Parlay & Correlation Architect")
    pool = st.session_state['prop_pool']
    if not pool.empty:
        active = pool[pool['sport'].str.strip().str.upper() == sport.strip().upper()].copy()
        target_line_col = 'prop_line'
        if site == 'Underdog': target_line_col = 'underdog_line'
        elif site == 'Sleeper': target_line_col = 'sleeper_line'
        
        active['final_line'] = active.apply(lambda x: x.get(target_line_col, 0) if x.get(target_line_col, 0) > 0 else x.get('prop_line', 0), axis=1)
        active = active[active['final_line'] > 0]
        active['pick'] = np.where(active['projection'] > active['final_line'], "OVER", "UNDER")
        active['uid'] = active['name'] + " (" + active['market'] + ": " + active['pick'] + ")"
        
        st.subheader("🔗 Correlation Station")
        teams = active['team'].unique()
        selected_team = st.selectbox("Find Correlated Pairs for Team:", teams)
        
        team_props = active[active['team'] == selected_team]
        if not team_props.empty:
            st.dataframe(team_props[['name', 'market', 'pick', 'win_prob', 'titan_logic']])
        
        st.divider()
        selection = st.multiselect("Build Custom Slip", active['uid'].unique())
        if selection:
            total_prob = 1.0
            for item in selection:
                row = active[active['uid'] == item].iloc[0]
                edge = abs(row['projection'] - row['final_line']) / row['final_line']
                prob = min(0.75, 0.52 + (edge * 0.5))
                total_prob *= prob
            st.metric("Total Probability", f"{total_prob*100:.1f}%")
