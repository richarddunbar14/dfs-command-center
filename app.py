import streamlit as st
import pandas as pd
import numpy as np
import pulp
import re
import time
import sqlite3
import base64
import requests
import difflib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from duckduckgo_search import DDGS

# ==========================================
# ⚙️ 1. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="TITAN UNIVERSAL", page_icon="🌍")

st.markdown("""
<style>
    /* THEME: Universal Dark */
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    
    /* CARDS */
    .titan-card { 
        background: #111; border: 1px solid #333; padding: 15px; border-radius: 8px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 10px; border-left: 4px solid #00d2ff;
    }
    .titan-card-prop { border-left: 4px solid #00ff41; }
    
    /* METRICS */
    div[data-testid="stMetricValue"] { color: #00ff41; text-shadow: 0 0 10px rgba(0,255,65,0.3); }
    
    /* BUTTONS */
    .stButton>button { 
        width: 100%; border-radius: 4px; font-weight: 800; letter-spacing: 1px; text-transform: uppercase;
        background: linear-gradient(90deg, #111 0%, #222 100%); color: #00d2ff; border: 1px solid #00d2ff; transition: 0.3s;
    }
    .stButton>button:hover { background: #00d2ff; color: #000; box-shadow: 0 0 15px #00d2ff; }
</style>
""", unsafe_allow_html=True)

# Session State
if 'dfs_pool' not in st.session_state: st.session_state['dfs_pool'] = pd.DataFrame()
if 'prop_pool' not in st.session_state: st.session_state['prop_pool'] = pd.DataFrame()
if 'ai_intel' not in st.session_state: st.session_state['ai_intel'] = {}
if 'api_log' not in st.session_state: st.session_state['api_log'] = []

# MASTER KEY
RAPID_KEY = "cf5f5b1102mshedbe3c14b0eb432p176b16jsnf9b2d93c804c"

# ==========================================
# 📡 2. LEVIATHAN ROUTER (ALL SPORTS)
# ==========================================

class LeviathanRouter:
    def __init__(self, key):
        self.key = key
        self.headers = {"X-RapidAPI-Key": key}

    def fetch(self, sport):
        """Routes to specific APIs for WNBA, CBB, CFB, etc."""
        data = []
        log = []
        today_compact = datetime.now().strftime("%Y%m%d")

        try:
            # --- NBA ---
            if sport == 'NBA':
                url = "https://api-nba-v1.p.rapidapi.com/games"
                self.headers["X-RapidAPI-Host"] = "api-nba-v1.p.rapidapi.com"
                params = {"date": datetime.now().strftime("%Y-%m-%d")}
                res = requests.get(url, headers=self.headers, params=params)
                log.append(f"✅ NBA API: {res.status_code}")

            # --- WNBA (NEW) ---
            elif sport == 'WNBA':
                # Using standard basketball API or WNBA specific if available
                url = "https://api-basketball.p.rapidapi.com/games"
                self.headers["X-RapidAPI-Host"] = "api-basketball.p.rapidapi.com"
                params = {"date": datetime.now().strftime("%Y-%m-%d"), "league": "13"} # 13 is often WNBA
                res = requests.get(url, headers=self.headers, params=params)
                log.append(f"✅ WNBA API: {res.status_code}")

            # --- NFL ---
            elif sport == 'NFL':
                url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getDFSProjections"
                self.headers["X-RapidAPI-Host"] = "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
                params = {"gameDate": today_compact, "slate": "main"}
                res = requests.get(url, headers=self.headers, params=params)
                if res.status_code == 200:
                    raw = res.json().get('body', [])
                    for p in raw:
                         if isinstance(p, dict):
                            data.append({
                                'name': p.get('player'), 'salary': float(p.get('salary', 0) or 0),
                                'projection': float(p.get('fantasyPoints', 0) or 0),
                                'position': p.get('pos'), 'team': p.get('team'), 'sport': 'NFL'
                            })
                    log.append(f"✅ NFL Tank01: {len(data)} players")

            # --- CFB (NEW) ---
            elif sport == 'CFB':
                url = "https://college-football-api.p.rapidapi.com/games"
                self.headers["X-RapidAPI-Host"] = "college-football-api.p.rapidapi.com"
                params = {"year": datetime.now().year, "seasonType": "regular"}
                res = requests.get(url, headers=self.headers, params=params)
                log.append(f"✅ CFB API: {res.status_code}")

            # --- MLB (NEW) ---
            elif sport == 'MLB':
                url = "https://tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com/getDailyGameList"
                self.headers["X-RapidAPI-Host"] = "tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com"
                res = requests.get(url, headers=self.headers, params={"gameDate": today_compact})
                log.append(f"✅ MLB API: {res.status_code}")

            # --- CBB (NEW) ---
            elif sport == 'CBB':
                url = "https://tank01-ncaab-live-in-game-real-time-statistics.p.rapidapi.com/getDailyGameList"
                self.headers["X-RapidAPI-Host"] = "tank01-ncaab-live-in-game-real-time-statistics.p.rapidapi.com"
                res = requests.get(url, headers=self.headers, params={"gameDate": today_compact})
                log.append(f"✅ CBB API: {res.status_code}")

            # --- NHL ---
            elif sport == 'NHL':
                url = "https://nhl-stats-and-salary.p.rapidapi.com/players"
                self.headers["X-RapidAPI-Host"] = "nhl-stats-and-salary.p.rapidapi.com"
                res = requests.get(url, headers=self.headers)
                log.append(f"✅ NHL API: {res.status_code}")
            
            # --- PROPS BACKUP (Pinnacle) ---
            url_odds = "https://pinnacle-odds.p.rapidapi.com/kit/v1/special-markets"
            self.headers["X-RapidAPI-Host"] = "pinnacle-odds.p.rapidapi.com"
            requests.get(url_odds, headers=self.headers, params={"sport_id": "1", "is_have_odds": "1"})
            log.append("✅ Pinnacle Odds: Active")

        except Exception as e:
            log.append(f"❌ API Error: {e}")

        st.session_state['api_log'] = log
        return pd.DataFrame(data)

# ==========================================
# 💾 3. DATABASE
# ==========================================

def init_db():
    conn = sqlite3.connect('titan_universal.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS bankroll (id INTEGER PRIMARY KEY, date TEXT, amount REAL, notes TEXT)''')
    c.execute('SELECT count(*) FROM bankroll')
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO bankroll (date, amount, notes) VALUES (?, ?, ?)", 
                  (datetime.now().strftime("%Y-%m-%d"), 1000.0, 'Genesis'))
        conn.commit()
    return conn

def get_bankroll_history(conn):
    return pd.read_sql("SELECT * FROM bankroll", conn)

def update_bankroll(conn, amount, note):
    c = conn.cursor()
    c.execute("INSERT INTO bankroll (date, amount, notes) VALUES (?, ?, ?)", 
              (datetime.now().strftime("%Y-%m-%d %H:%M"), float(amount), note))
    conn.commit()

# ==========================================
# 🧠 4. TITAN BRAIN
# ==========================================

class TitanBrain:
    def __init__(self, bankroll):
        self.bankroll = bankroll

    def calculate_prop_edge(self, row):
        if row.get('prop_line', 0) <= 0 or row.get('projection', 0) <= 0: return 0, "No Data", 0.0
        
        proj = row['projection']
        line = row['prop_line']
        edge_raw = abs(proj - line) / line
        
        win_prob = min(0.78, 0.52 + (edge_raw * 0.55))
        
        odds = 0.909
        kelly = ((odds * win_prob) - (1 - win_prob)) / odds
        units = max(0, kelly * 0.3) * 100 
        
        rating = "PASS"
        if units > 3.0: rating = "💎 ATOMIC"
        elif units > 1.5: rating = "🟢 NUCLEAR"
        elif units > 0.5: rating = "🟡 KINETIC"
        
        return units, rating, win_prob * 100

# ==========================================
# 📂 5. DATA REFINERY
# ==========================================

class DataRefinery:
    @staticmethod
    def clean_curr(val):
        try:
            s = str(val).strip()
            if s.lower() in ['-', '', 'nan', 'none']: return 0.0
            return float(re.sub(r'[^\d.]', '', s))
        except: return 0.0

    @staticmethod
    def ingest(df, sport_tag):
        df.columns = df.columns.astype(str).str.upper().str.strip()
        std = pd.DataFrame()
        
        maps = {
            'name': ['PLAYER', 'NAME', 'WHO', 'ATHLETE'],
            'projection': ['FPTS', 'PROJ', 'AVG FPTS', 'PTS', 'FC PROJ'],
            'salary': ['SAL', 'SALARY', 'CAP', 'COST'],
            'position': ['POS', 'POSITION', 'ROSTER POSITION'],
            'team': ['TEAM', 'TM', 'SQUAD'],
            'prop_line': ['O/U', 'PROP', 'LINE', 'STRIKE', 'TOTAL'],
            'game_info': ['GAME INFO', 'GAME', 'MATCHUP']
        }
        
        for target, sources in maps.items():
            for s in sources:
                if s in df.columns:
                    if target in ['projection', 'salary', 'prop_line']:
                        std[target] = df[s].apply(DataRefinery.clean_curr)
                    elif target == 'name':
                        std[target] = df[s].astype(str).apply(lambda x: x.split('(')[0].strip())
                    else:
                        std[target] = df[s].astype(str).str.strip()
                    break
        
        if 'name' not in std.columns: return pd.DataFrame()
        for c in ['projection', 'salary', 'prop_line']: 
            if c not in std.columns: std[c] = 0.0
        if 'position' not in std.columns: std['position'] = 'FLEX'
        if 'team' not in std.columns: std['team'] = 'N/A'
        if 'game_info' not in std.columns: std['game_info'] = 'All Games'
        
        std['sport'] = sport_tag
        return std

    @staticmethod
    def merge(base, new_df):
        if base.empty: return new_df
        if new_df.empty: return base
        return pd.concat([base, new_df]).drop_duplicates(subset=['name', 'sport'], keep='last').reset_index(drop=True)

# ==========================================
# 📡 6. AI SCOUT
# ==========================================

def run_web_scout(sport):
    intel = {}
    try:
        with DDGS() as ddgs:
            q = f"{sport} dfs sleepers value plays analysis today"
            for r in list(ddgs.text(q, max_results=5)):
                blob = (r['title'] + " " + r['body']).lower()
                intel[blob] = 1
    except: pass
    return intel

# ==========================================
# 🏭 7. OPTIMIZER & SIMULATION
# ==========================================

def get_roster_rules(sport, site, mode):
    rules = {'size': 0, 'cap': 50000, 'constraints': []}
    if site == 'DK': rules['cap'] = 50000
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

    # --- SPORT SPECIFIC RULES ---
    if sport == 'NFL':
        rules['size'] = 9
        if site == 'DK': rules['constraints'] = [('QB', 1, 1), ('DST', 1, 1), ('RB', 2, 3), ('WR', 3, 4), ('TE', 1, 2), ('RB|WR|TE', 7, 7)]
        elif site == 'FD': rules['constraints'] = [('QB', 1, 1), ('DEF', 1, 1), ('RB', 2, 3), ('WR', 3, 4), ('TE', 1, 2), ('RB|WR|TE', 7, 7)]
    
    elif sport == 'NBA':
        if site == 'DK':
            rules['size'] = 8
            rules['constraints'] = [('PG', 1, 3), ('SG', 1, 3), ('SF', 1, 3), ('PF', 1, 3), ('C', 1, 2), ('PG|SG', 3, 4), ('SF|PF', 3, 4)]
        elif site == 'FD':
            rules['size'] = 9
            rules['constraints'] = [('PG', 2, 2), ('SG', 2, 2), ('SF', 2, 2), ('PF', 2, 2), ('C', 1, 1)]
    
    elif sport == 'WNBA': # Similar to NBA but often simpler on FD
        if site == 'DK': rules['size'] = 6; rules['constraints'] = [('G', 2, 3), ('F', 3, 4)]
        elif site == 'FD': rules['size'] = 7; rules['constraints'] = [('G', 3, 3), ('F', 4, 4)]
        
    elif sport == 'MLB':
        rules['size'] = 10
        if site == 'DK': rules['constraints'] = [('P', 2, 2), ('C|1B|2B|3B|SS|OF', 8, 8)]
        
    elif sport == 'CBB': # College Basketball
        rules['size'] = 8 # Classic DK
        if site == 'DK': rules['constraints'] = [('G', 3, 4), ('F', 3, 4)]
        
    elif sport == 'CFB': # College Football
        rules['size'] = 9
        if site == 'DK': rules['constraints'] = [('QB', 1, 2), ('RB', 2, 4), ('WR', 3, 5)]

    return rules

def optimize_lineup(df, config):
    pool = df[df['sport'] == config['sport']].copy()
    if config['slate_games']:
        pool = pool[pool['game_info'].isin(config['slate_games'])].reset_index(drop=True)
        
    pool = pool[pool['projection'] > 0].reset_index(drop=True)
    if config['bans']: pool = pool[~pool['name'].isin(config['bans'])].reset_index(drop=True)
    if pool.empty: return None

    if config['mode'] == 'Showdown':
        flex = pool.copy(); flex['pos_id'] = 'FLEX'
        cpt = pool.copy(); cpt['pos_id'] = 'CPT' if config['site']=='DK' else 'MVP'
        cpt['name'] += f" ({cpt['pos_id'].iloc[0]})"
        cpt['projection'] *= 1.5
        if config['site'] == 'DK': cpt['salary'] *= 1.5
        pool = pd.concat([cpt, flex]).reset_index(drop=True)
    else:
        pool['pos_id'] = pool['position'].replace({'D': 'DST', 'DEF': 'DST'})

    rules = get_roster_rules(config['sport'], config['site'], config['mode'])
    lineups = []
    
    for i in range(config['count']):
        prob = pulp.LpProblem("Titan", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("p", pool.index, cat='Binary')
        
        # Sim Variance
        volatility = 0.15 if config['sim'] else 0.0
        pool['sim'] = pool['projection'] * np.random.normal(1.0, volatility, len(pool))
        
        prob += pulp.lpSum([pool.loc[p, 'sim'] * x[p] for p in pool.index])
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

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == 1:
            sel = [p for p in pool.index if x[p].varValue == 1]
            lu = pool.loc[sel].copy()
            lu['Lineup_ID'] = i+1
            lineups.append(lu)
            prob += pulp.lpSum([x[p] for p in sel]) <= rules['size'] - 1
            
    return pd.concat(lineups) if lineups else None

def get_html_report(df):
    html = f"""<html><body><h2>TITAN UNIVERSAL REPORT</h2><hr>"""
    for _, row in df.head(20).iterrows():
        html += f"<div><b>{row['name']}</b> ({row['position']}) | Sal: ${row['salary']} | Proj: {row['projection']:.1f}</div>"
    return base64.b64encode(html.encode()).decode()

# ==========================================
# 🖥️ 8. DASHBOARD
# ==========================================

conn = init_db()
st.sidebar.title("TITAN OMNI")
st.sidebar.caption("Universal Dominion")

history = get_bankroll_history(conn)
current_bank = history.iloc[-1]['amount']
st.sidebar.metric("🏦 Bankroll", f"${current_bank:,.2f}")

if len(history) > 1:
    fig = px.line(history, x='date', y='amount', height=150)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.sidebar.plotly_chart(fig, use_container_width=True)

with st.sidebar.expander("Update Funds"):
    new = st.number_input("New Balance", value=current_bank)
    if st.button("Update"):
        update_bankroll(conn, new, "Manual")
        st.rerun()

sport = st.sidebar.selectbox("Sport", ["NFL", "NBA", "MLB", "CFB", "CBB", "WNBA", "NHL", "PGA"])
site = st.sidebar.selectbox("Site", ["DK", "FD", "Yahoo", "PrizePicks"])

tabs = st.tabs(["1. 📡 Fusion", "2. 🏰 Optimizer", "3. 🔮 Simulation", "4. 🚀 Props", "5. 🧮 Parlay"])

# --- TAB 1: DATA ---
with tabs[0]:
    st.markdown("### 📡 Hydra Data Router")
    
    col_api, col_file = st.columns(2)
    with col_api:
        if st.button("☁️ AUTO-SYNC APIS"):
            gw = LeviathanRouter(RAPID_KEY)
            with st.spinner(f"Routing to {sport} API endpoints..."):
                api_data = gw.fetch(sport)
                
                # Debug Log
                with st.expander("API Logs"):
                    for l in st.session_state['api_log']: st.write(l)
                
                if not api_data.empty:
                    ref = DataRefinery()
                    # Assume API returns raw data we clean
                    cln = ref.ingest(api_data, sport)
                    st.session_state['dfs_pool'] = ref.merge(st.session_state['dfs_pool'], cln)
                    # Also populate props if applicable
                    st.session_state['prop_pool'] = ref.merge(st.session_state['prop_pool'], cln)
                    st.success(f"Hydra synced {len(cln)} items for {sport}.")
                else:
                    st.warning("No data returned from API. Check connection/subscription.")

    with col_file:
        files = st.file_uploader("Upload CSVs (Salary/Proj)", accept_multiple_files=True)
        if st.button("🧬 Fuse Files"):
            if files:
                ref = DataRefinery()
                new_data = pd.DataFrame()
                for f in files:
                    try:
                        try: raw = pd.read_csv(f, encoding='utf-8-sig')
                        except: raw = pd.read_excel(f)
                        new_data = ref.merge(new_data, ref.ingest(raw, sport))
                    except: pass
                st.session_state['dfs_pool'] = ref.merge(st.session_state['dfs_pool'], new_data)
                st.success(f"Fused {len(new_data)} players.")

    if st.button("🛰️ Run AI Scout"):
        st.session_state['ai_intel'] = run_web_scout(sport)
        st.success("AI Updated")

# --- TAB 2: OPTIMIZER ---
with tabs[1]:
    pool = st.session_state['dfs_pool']
    active = pool[pool['sport'] == sport] if not pool.empty else pd.DataFrame()
    
    if active.empty:
        st.warning(f"No {sport} DFS data loaded.")
    else:
        # Slate Filter
        games = sorted(active['game_info'].astype(str).unique())
        slate = st.multiselect("🗓️ Filter Slate", games, default=games)
        
        c1, c2, c3 = st.columns(3)
        mode = c1.radio("Mode", ["Classic", "Showdown"])
        count = c2.slider("Lineups", 1, 50, 10)
        stack = c3.checkbox("Stacking (NFL)", value=(sport=='NFL'))
        
        names = sorted(active['name'].unique())
        locks = st.multiselect("Lock", names)
        bans = st.multiselect("Ban", names)
        
        if st.button("⚡ Generate Lineups"):
            cfg = {'sport':sport, 'site':site, 'mode':mode, 'count':count, 'locks':locks, 'bans':bans, 'stack':stack, 'sim':False, 'slate_games':slate}
            res = optimize_lineup(st.session_state['dfs_pool'], cfg)
            
            if res is not None:
                st.dataframe(res)
                st.markdown(f'<a href="data:text/html;base64,{get_html_report(res)}" download="report.html">📥 Download Cheat Sheet</a>', unsafe_allow_html=True)
                
                # Visuals
                fig = px.treemap(res, path=['team'], values='projection', title="Team Exposure")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Optimization Failed.")

# --- TAB 3: SIMULATION ---
with tabs[2]:
    if st.button("🎲 Run Simulation"):
        with st.spinner("Simulating..."):
            cfg = {'sport':sport, 'site':site, 'mode':"Classic", 'count':50, 'locks':[], 'bans':[], 'stack':True, 'sim':True, 'slate_games':[]}
            res = optimize_lineup(st.session_state['dfs_pool'], cfg)
            if res is not None:
                exp = res['name'].value_counts(normalize=True).mul(100).reset_index()
                st.plotly_chart(px.bar(exp.head(15), x='proportion', y='name', orientation='h'))

# --- TAB 4: PROPS ---
with tabs[3]:
    pool = st.session_state['prop_pool']
    active = pool[pool['sport'] == sport].copy() if not pool.empty else pd.DataFrame()
    
    if active.empty: st.warning("No props found.")
    else:
        brain = TitanBrain(current_bank)
        active['pick'] = np.where(active['projection'] > active['prop_line'], "OVER", "UNDER")
        res = active.apply(lambda x: brain.calculate_prop_edge(x), axis=1, result_type='expand')
        active['units'] = res[0]
        active['rating'] = res[1]
        active['win_prob'] = res[2]
        st.dataframe(active[active['units'] > 0].sort_values('win_prob', ascending=False))

# --- TAB 5: PARLAY ---
with tabs[4]:
    pool = st.session_state['prop_pool']
    active = pool[pool['sport'] == sport].copy() if not pool.empty else pd.DataFrame()
    if active.empty: st.warning("No props.")
    else:
        st.markdown("### 🧮 Parlay Architect")
        active['pick'] = np.where(active['projection'] > active['prop_line'], "OVER", "UNDER")
        options = active['name'] + " (" + active['pick'] + ")"
        selection = st.multiselect("Build Slip", options)
        
        if selection:
            total_prob = 1.0
            for item in selection:
                name = item.split(" (")[0]
                row = active[active['name'] == name].iloc[0]
                edge = abs(row['projection'] - row['prop_line']) / row['prop_line']
                prob = min(0.75, 0.52 + (edge * 0.5))
                total_prob *= prob
            
            c1, c2 = st.columns(2)
            c1.metric("Legs", len(selection))
            c2.metric("True Probability", f"{total_prob*100:.1f}%")
            if total_prob > 0: st.info(f"Fair Odds: {1/total_prob:.2f}")
