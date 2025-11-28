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
st.set_page_config(layout="wide", page_title="TITAN OMNI: MULTI-VERSE", page_icon="🌌")

st.markdown("""
<style>
    /* THEME: Deep Space & Neon Blue */
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    
    /* CARDS */
    .titan-card { 
        background: #111; border: 1px solid #333; padding: 15px; border-radius: 8px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 10px; border-left: 4px solid #29b6f6;
    }
    
    /* DATAFRAME STYLING */
    div[data-testid="stDataFrame"] { border: 1px solid #333; border-radius: 5px; }
    
    /* BUTTONS */
    .stButton>button { 
        width: 100%; border-radius: 4px; font-weight: 800; letter-spacing: 1px; text-transform: uppercase;
        background: linear-gradient(90deg, #111 0%, #222 100%); color: #29b6f6; border: 1px solid #29b6f6; transition: 0.3s;
    }
    .stButton>button:hover { background: #29b6f6; color: #000; box-shadow: 0 0 15px #29b6f6; }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if 'dfs_pool' not in st.session_state: st.session_state['dfs_pool'] = pd.DataFrame()
if 'prop_pool' not in st.session_state: st.session_state['prop_pool'] = pd.DataFrame()
if 'api_log' not in st.session_state: st.session_state['api_log'] = []

# ==========================================
# 📡 2. MULTI-VERSE API ROUTER
# ==========================================

class MultiVerseGateway:
    def __init__(self, api_key):
        self.key = api_key
        self.headers = {"X-RapidAPI-Key": self.key, "Content-Type": "application/json"}

    def fetch_data(self, sport):
        data = []
        log_msg = ""
        try:
            # --- NBA ---
            if sport == 'NBA':
                url = "https://api-nba-v1.p.rapidapi.com/games"
                self.headers["X-RapidAPI-Host"] = "api-nba-v1.p.rapidapi.com"
                params = {"date": datetime.now().strftime("%Y-%m-%d")}
                res = requests.get(url, headers=self.headers, params=params)
                if res.status_code == 200:
                    games = res.json().get('response', [])
                    for g in games:
                        data.append({'game': f"{g['teams']['home']['code']} vs {g['teams']['visitors']['code']}", 'sport': 'NBA'})
            # --- NFL ---
            elif sport == 'NFL':
                url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getDailyGameList"
                self.headers["X-RapidAPI-Host"] = "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
                params = {"gameDate": datetime.now().strftime("%Y%m%d")}
                res = requests.get(url, headers=self.headers, params=params)
                if res.status_code == 200:
                    games = res.json().get('body', [])
                    for g in games:
                        data.append({'game': g.get('gameID'), 'sport': 'NFL'})
            # Add other sports as needed...
        except Exception as e:
            log_msg = f"Error: {e}"

        st.session_state['api_log'].append(f"{datetime.now().time()} - {sport}: {log_msg}")
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
        c.execute("INSERT INTO bankroll (date, amount, notes) VALUES (?, ?, ?)", 
                  (datetime.now().strftime("%Y-%m-%d"), 1000.0, 'Genesis'))
        conn.commit()
    return conn

def get_bankroll(conn):
    try:
        df = pd.read_sql("SELECT * FROM bankroll ORDER BY id DESC LIMIT 1", conn)
        return df.iloc[0]['amount'] if not df.empty else 1000.0
    except: return 1000.0

def get_bankroll_history(conn):
    try:
        return pd.read_sql("SELECT * FROM bankroll", conn)
    except:
        return pd.DataFrame({'date': [datetime.now().strftime("%Y-%m-%d")], 'amount': [1000.0]})

def update_bankroll(conn, amount, note):
    c = conn.cursor()
    c.execute("INSERT INTO bankroll (date, amount, notes) VALUES (?, ?, ?)", 
              (datetime.now().strftime("%Y-%m-%d"), float(amount), note))
    conn.commit()

# ==========================================
# 🧠 4. TITAN BRAIN
# ==========================================

class TitanBrain:
    def __init__(self, bankroll):
        self.bankroll = bankroll

    def calculate_prop_edge(self, row, line_col='prop_line'):
        # Get the specific line for the selected site, or fallback to generic
        line = row.get(line_col, 0)
        # If specific line is missing (0), use the generic 'prop_line'
        if line == 0 or pd.isna(line): line = row.get('prop_line', 0)
        
        proj = row.get('projection', 0)
        
        if line <= 0 or proj <= 0: return 0, "No Data", 0.0
        
        edge_raw = abs(proj - line) / line
        # Win probability formula
        win_prob = min(0.78, 0.52 + (edge_raw * 0.55))
        
        # Kelly Criterion for Bankroll Management
        odds = 0.909 # Standard -110 odds implied
        kelly = ((odds * win_prob) - (1 - win_prob)) / odds
        units = max(0, kelly * 0.3) * 100 
        
        rating = "PASS"
        if units > 3.0: rating = "💎 ATOMIC"
        elif units > 1.5: rating = "🟢 NUCLEAR"
        elif units > 0.5: rating = "🟡 KINETIC"
        
        return units, rating, win_prob * 100

# ==========================================
# 📂 5. DATA REFINERY (UPDATED FOR FACTORS & DATES)
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
    def ingest(df, sport_tag):
        # Normalize columns to uppercase for easier matching
        df.columns = df.columns.astype(str).str.upper().str.strip()
        std = pd.DataFrame()
        
        # 1. BASIC FIELDS MAPPING
        maps = {
            'name': ['PLAYER', 'NAME', 'WHO', 'ATHLETE'],
            'projection': ['ROTOWIRE PROJECTION', 'PROJECTION', 'FPTS', 'PROJ', 'AVG FPTS', 'PTS'], 
            'salary': ['SAL', 'SALARY', 'CAP', 'COST'],
            'position': ['POS', 'POSITION', 'ROSTER POSITION'],
            'team': ['TEAM', 'TM', 'SQUAD'],
            'prop_line': ['LINE', 'O/U', 'PROP', 'STRIKE', 'TOTAL'], # Generic Line
            'market': ['MARKET NAME', 'MARKET', 'PROP TYPE', 'STAT'],
            'game_info': ['GAME INFO', 'GAME', 'MATCHUP', 'OPPONENT'],
            'date': ['DATE', 'GAME DATE'],
            'time': ['TIME', 'GAME TIME'],
            # "Far Cry" Specs (Factors)
            'factor_pickem': ["DFS PICK'EM SITES FACTOR", "PICK'EM FACTOR"],
            'factor_sportsbook': ["SPORTSBOOKS FACTOR", "BOOKS FACTOR"],
            'factor_hit_rate': ["HIT RATE FACTOR", "HIT RATE"],
            'factor_proj': ["ROTOWIRE PROJECTION FACTOR", "PROJ FACTOR"]
        }
        
        # 2. SITE SPECIFIC LINES MAPPING
        site_maps = {
            'prizepicks_line': ['PRIZEPICKS LINE', 'PRIZEPICKS'],
            'underdog_line': ['UNDERDOG LINE', 'UNDERDOG'],
            'sleeper_line': ['SLEEPER LINE', 'SLEEPER'],
            'pick6_line': ['DRAFTKINGS PICK6 LINE', 'PICK6 LINE', 'PICK6']
        }
        
        # Ingest Basic Fields
        for target, sources in maps.items():
            for s in sources:
                if s in df.columns:
                    if target in ['projection', 'salary', 'prop_line', 'factor_pickem', 'factor_sportsbook', 'factor_hit_rate', 'factor_proj']:
                        std[target] = df[s].apply(DataRefinery.clean_curr)
                    elif target == 'name':
                        std[target] = df[s].astype(str).apply(lambda x: x.split('(')[0].strip())
                    else:
                        std[target] = df[s].astype(str).str.strip()
                    break
        
        # Ingest Site Specific Lines
        for target, sources in site_maps.items():
            for s in sources:
                if s in df.columns:
                    std[target] = df[s].apply(DataRefinery.clean_curr)
                    break
            if target not in std.columns: std[target] = 0.0

        if 'name' not in std.columns: return pd.DataFrame()
        
        # Set Defaults if missing
        if 'projection' not in std.columns: std['projection'] = 0.0
        if 'salary' not in std.columns: std['salary'] = 0.0
        if 'prop_line' not in std.columns: std['prop_line'] = 0.0
        if 'position' not in std.columns: std['position'] = 'FLEX'
        if 'team' not in std.columns: std['team'] = 'N/A'
        if 'market' not in std.columns: std['market'] = 'Standard'
        if 'game_info' not in std.columns: std['game_info'] = 'All Games'
        if 'date' not in std.columns: std['date'] = datetime.now().strftime("%Y-%m-%d")
        if 'time' not in std.columns: std['time'] = "TBD"
        
        # Sport Detection
        if 'SPORT' in df.columns:
            std['sport'] = df['SPORT'].str.strip().str.upper()
        else:
            std['sport'] = sport_tag.upper()
            
        return std

    @staticmethod
    def merge(base, new_df):
        if base.empty: return new_df
        if new_df.empty: return base
        # Merge key: Name + Market + Sport to avoid duplicates but keep different props
        return pd.concat([base, new_df]).drop_duplicates(subset=['name', 'sport', 'market'], keep='last').reset_index(drop=True)

# ==========================================
# 🏭 7. OPTIMIZER
# ==========================================

def get_roster_rules(sport, site, mode):
    rules = {'size': 0, 'cap': 50000, 'constraints': []}
    
    # Cap logic
    if site == 'DK' or site == 'DraftKings Pick6': rules['cap'] = 50000
    elif site == 'FD': rules['cap'] = 60000
    elif site == 'Yahoo': rules['cap'] = 200

    # NFL Rules
    if sport == 'NFL':
        rules['size'] = 9
        if site == 'DK': rules['constraints'] = [('QB', 1, 1), ('DST', 1, 1), ('RB', 2, 3), ('WR', 3, 4), ('TE', 1, 2), ('RB|WR|TE', 7, 7)]
        elif site == 'FD': rules['constraints'] = [('QB', 1, 1), ('DEF', 1, 1), ('RB', 2, 3), ('WR', 3, 4), ('TE', 1, 2), ('RB|WR|TE', 7, 7)]
    # NBA Rules
    elif sport == 'NBA':
        if site == 'DK':
            rules['size'] = 8
            rules['constraints'] = [('PG', 1, 3), ('SG', 1, 3), ('SF', 1, 3), ('PF', 1, 3), ('C', 1, 2), ('PG|SG', 3, 4), ('SF|PF', 3, 4)]
        elif site == 'FD':
            rules['size'] = 9
            rules['constraints'] = [('PG', 2, 2), ('SG', 2, 2), ('SF', 2, 2), ('PF', 2, 2), ('C', 1, 1)]
    
    if rules['size'] == 0: rules['size'] = 6
    return rules

def optimize_lineup(df, config):
    # FILTER BY SPORT (STRICT)
    target_sport = config['sport'].strip().upper()
    pool = df[df['sport'].str.strip().str.upper() == target_sport].copy()
    
    if pool.empty: return None

    if config['slate_games']:
        pool = pool[pool['game_info'].isin(config['slate_games'])].reset_index(drop=True)
        
    pool = pool[pool['projection'] > 0].reset_index(drop=True)
    if config['bans']: pool = pool[~pool['name'].isin(config['bans'])].reset_index(drop=True)
    if pool.empty: return None

    if config['mode'] == 'Showdown':
        flex = pool.copy(); flex['pos_id'] = 'FLEX'
        cpt = pool.copy(); cpt['pos_id'] = 'CPT'
        cpt['name'] += f" (CPT)"
        cpt['projection'] *= 1.5
        cpt['salary'] *= 1.5
        pool = pd.concat([cpt, flex]).reset_index(drop=True)
    else:
        pool['pos_id'] = pool['position'].replace({'D': 'DST', 'DEF': 'DST'})

    rules = get_roster_rules(config['sport'], config['site'], config['mode'])
    lineups = []
    
    for i in range(config['count']):
        prob = pulp.LpProblem("Titan", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("p", pool.index, cat='Binary')
        
        prob += pulp.lpSum([pool.loc[p, 'projection'] * x[p] for p in pool.index])
        prob += pulp.lpSum([pool.loc[p, 'salary'] * x[p] for p in pool.index]) <= rules['cap']
        prob += pulp.lpSum([x[p] for p in pool.index]) == rules['size']
        
        for role, min_req, max_req in rules['constraints']:
            idx = pool[pool['pos_id'].str.contains(role, regex=True, na=False)].index
            if not idx.empty:
                prob += pulp.lpSum([x[p] for p in idx]) >= min_req
                prob += pulp.lpSum([x[p] for p in idx]) <= max_req

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == 1:
            sel = [p for p in pool.index if x[p].varValue == 1]
            lu = pool.loc[sel].copy()
            lu['Lineup_ID'] = i+1
            lineups.append(lu)
            prob += pulp.lpSum([x[p] for p in sel]) <= rules['size'] - 1
            
    return pd.concat(lineups) if lineups else None

def get_html_report(df):
    html = f"""<html><body><h2>TITAN HYDRA REPORT</h2><hr>"""
    for _, row in df.head(20).iterrows():
        html += f"<div><b>{row['name']}</b> ({row['position']}) | Sal: ${row['salary']} | Proj: {row['projection']:.1f}</div>"
    return base64.b64encode(html.encode()).decode()

# ==========================================
# 🖥️ 8. DASHBOARD
# ==========================================

conn = init_db()
st.sidebar.title("TITAN OMNI")
st.sidebar.caption("Hydra Edition 3.0 (Bolt-On Complete)")

# API Key handling
try: API_KEY = st.secrets["rapid_api_key"]
except: API_KEY = st.sidebar.text_input("Enter RapidAPI Key", type="password")

history = get_bankroll_history(conn)
current_bank = get_bankroll(conn)
st.sidebar.metric("🏦 Bankroll", f"${current_bank:,.2f}")

# GLOBAL FILTERS
sport = st.sidebar.selectbox("Sport", ["NBA", "NFL", "MLB", "CFB", "NHL", "SOCCER", "PGA", "CS2"])
site = st.sidebar.selectbox("Site/Book", ["PrizePicks", "Underdog", "Sleeper", "DraftKings Pick6", "DK", "FD"])

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
        files = st.file_uploader("Upload CSVs (Sleeper/Underdog/RotoWire)", accept_multiple_files=True)
        if st.button("🧬 Fuse Files"):
            if files:
                ref = DataRefinery()
                new_data = pd.DataFrame()
                for f in files:
                    try:
                        try: raw = pd.read_csv(f, encoding='utf-8-sig')
                        except: raw = pd.read_excel(f)
                        st.caption(f"Ingesting {f.name}...")
                        new_data = ref.merge(new_data, ref.ingest(raw, sport))
                    except Exception as e: st.error(f"Error {f.name}: {e}")
                
                st.session_state['dfs_pool'] = ref.merge(st.session_state['dfs_pool'], new_data)
                st.session_state['prop_pool'] = ref.merge(st.session_state['prop_pool'], new_data)
                st.success(f"Fused {len(new_data)} records successfully.")

# --- TAB 2: OPTIMIZER ---
with tabs[1]:
    pool = st.session_state['dfs_pool']
    active = pool[pool['sport'].str.strip().str.upper() == sport.strip().upper()] if not pool.empty else pd.DataFrame()
    
    if active.empty:
        st.warning(f"No data for {sport}. Upload CSV or Sync.")
    else:
        st.write(f"Pool Size: {len(active)} players")
        games = sorted(active['game_info'].astype(str).unique())
        slate = st.multiselect("🗓️ Filter Slate", games, default=games)
        
        c1, c2 = st.columns(2)
        mode = c1.radio("Mode", ["Classic", "Showdown"])
        count = c2.slider("Lineups", 1, 50, 10)
        
        if st.button("⚡ Generate Lineups"):
            cfg = {'sport':sport, 'site':site, 'mode':mode, 'count':count, 'locks':[], 'bans':[], 'slate_games':slate, 'sim':False}
            res = optimize_lineup(st.session_state['dfs_pool'], cfg)
            if res is not None:
                st.dataframe(res)
            else:
                st.error("Optimizer found no valid solution.")

# --- TAB 4: PROPS (FIXED FOR FACTORS & DATES) ---
with tabs[3]:
    st.markdown("### 🚀 Prop Analyzer")
    pool = st.session_state['prop_pool']
    
    if not pool.empty:
        # STRICT SPORT FILTER
        active = pool[pool['sport'].str.strip().str.upper() == sport.strip().upper()].copy()
    else:
        active = pd.DataFrame()
    
    if active.empty:
        st.warning(f"No props found for {sport}.")
    else:
        brain = TitanBrain(current_bank)
        
        # DETERMINE LINE BASED ON SITE SELECTION
        target_line_col = 'prop_line'
        if site == 'Underdog': target_line_col = 'underdog_line'
        elif site == 'Sleeper': target_line_col = 'sleeper_line'
        elif site == 'PrizePicks': target_line_col = 'prizepicks_line'
        elif site == 'DraftKings Pick6': target_line_col = 'pick6_line'
        
        # Fallback Logic: If specific line is 0, use generic line
        active['final_line'] = active.apply(lambda x: x[target_line_col] if x.get(target_line_col, 0) > 0 else x['prop_line'], axis=1)
        
        # Logic
        active['pick'] = np.where(active['projection'] > active['final_line'], "OVER", "UNDER")
        
        res = active.apply(lambda x: brain.calculate_prop_edge(x, 'final_line'), axis=1, result_type='expand')
        active['units'] = res[0]
        active['rating'] = res[1]
        active['win_prob'] = res[2]
        
        # Filter valid rows
        valid_props = active[(active['projection'] > 0) & (active['final_line'] > 0)].copy()
        
        # Define Columns to Show (including Date, Time, Factors)
        cols_to_show = [
            'date', 'time', 'name', 'market', 'team', 'final_line', 'projection', 
            'pick', 'win_prob', 'rating', 
            'factor_pickem', 'factor_sportsbook', 'factor_hit_rate', 'factor_proj'
        ]
        # Only include columns that actually exist in the dataframe
        final_cols = [c for c in cols_to_show if c in valid_props.columns]
        
        # Display Table (Sorting by Win Probability)
        st.dataframe(
            valid_props[final_cols]
            .sort_values('win_prob', ascending=False)
            .style.format({'final_line': '{:.1f}', 'projection': '{:.1f}', 'win_prob': '{:.1f}%'})
        )

# --- TAB 5: PARLAY ---
with tabs[4]:
    pool = st.session_state['prop_pool']
    if not pool.empty:
        active = pool[pool['sport'].str.strip().str.upper() == sport.strip().upper()].copy()
    else:
        active = pd.DataFrame()

    if active.empty: st.warning("No props.")
    else:
        st.markdown("### 🧮 Parlay Architect")
        # Ensure final_line exists
        target_line_col = 'prop_line'
        if site == 'Underdog': target_line_col = 'underdog_line'
        elif site == 'Sleeper': target_line_col = 'sleeper_line'
        
        active['final_line'] = active.apply(lambda x: x.get(target_line_col, 0) if x.get(target_line_col, 0) > 0 else x.get('prop_line', 0), axis=1)
        active = active[active['final_line'] > 0]
        
        active['pick'] = np.where(active['projection'] > active['final_line'], "OVER", "UNDER")
        active['uid'] = active['name'] + " (" + active['market'] + ": " + active['pick'] + ")"
        
        selection = st.multiselect("Build Slip", active['uid'].unique())
        
        if selection:
            total_prob = 1.0
            for item in selection:
                row = active[active['uid'] == item].iloc[0]
                edge = abs(row['projection'] - row['final_line']) / row['final_line']
                prob = min(0.75, 0.52 + (edge * 0.5))
                total_prob *= prob
            
            c1, c2 = st.columns(2)
            c1.metric("Legs", len(selection))
            c2.metric("True Probability", f"{total_prob*100:.1f}%")
