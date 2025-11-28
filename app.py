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
st.set_page_config(layout="wide", page_title="TITAN OMNI: SINGULARITY", page_icon="🌌")

st.markdown("""
<style>
    /* THEME: Deep Space */
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
if 'api_data' not in st.session_state: st.session_state['api_data'] = {}

# API KEYS
SPORTSDATAIO_KEY = "dda563b328d34b80a38c26cd43223614"
ODDS_API_KEY = "a31f629075c27927fe99b097b51e1717"

# ==========================================
# 💾 2. DATABASE & HISTORY
# ==========================================

def init_db():
    conn = sqlite3.connect('titan_final.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS bankroll 
                 (id INTEGER PRIMARY KEY, date TEXT, amount REAL, notes TEXT)''')
    c.execute('SELECT count(*) FROM bankroll')
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO bankroll (date, amount, notes) VALUES (?, ?, ?)", 
                  (datetime.now().strftime("%Y-%m-%d %H:%M"), 1000.0, 'Genesis'))
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
# 📡 3. UNIVERSAL API GATEWAY
# ==========================================

class UniversalGateway:
    def __init__(self):
        self.sdio_key = SPORTSDATAIO_KEY
        self.odds_key = ODDS_API_KEY

    def fetch_live_props(self, sport):
        """Fetches Live Betting Lines (TheOddsAPI)."""
        maps = {'NFL': 'americanfootball_nfl', 'NBA': 'basketball_nba', 'MLB': 'baseball_mlb', 'NHL': 'icehockey_nhl'}
        key = maps.get(sport)
        if not key: return pd.DataFrame()
        
        url = f"https://api.the-odds-api.com/v4/sports/{key}/odds"
        params = {'apiKey': self.odds_key, 'regions': 'us', 'markets': 'player_points,h2h', 'oddsFormat': 'american'}
        
        try:
            res = requests.get(url, params=params).json()
            # Simple parser for demonstration
            # In a full app, this would iterate through every outcome
            return pd.DataFrame(res) 
        except: return pd.DataFrame()

    def fetch_team_momentum(self, sport):
        """Fetches Standings (SportsDataIO)."""
        year = datetime.now().year
        # Adjust for NFL season start
        if sport == 'NFL' and datetime.now().month < 8: year -= 1
        
        url = f"https://api.sportsdata.io/v3/{sport.lower()}/scores/json/Standings/{year}?key={self.sdio_key}"
        try:
            res = requests.get(url).json()
            # Return map {Team: WinPct}
            return {t.get('Team'): (t.get('Wins', 0)/max(1, t.get('Wins', 0)+t.get('Losses', 0))) for t in res}
        except: return {}

# ==========================================
# 🧠 4. TITAN BRAIN (MATH)
# ==========================================

class TitanBrain:
    def __init__(self, bankroll):
        self.bankroll = bankroll

    def calculate_prop_edge(self, row):
        if row.get('prop_line', 0) <= 0 or row.get('projection', 0) <= 0: return 0, "No Data", 0.0
        
        proj = row['projection']
        line = row['prop_line']
        edge_raw = abs(proj - line) / line
        
        # Win Prob (Logistic)
        win_prob = 0.52 + (edge_raw * 0.55)
        win_prob = min(0.78, win_prob)
        
        # Kelly
        odds = 0.909 # -110
        kelly = ((odds * win_prob) - (1 - win_prob)) / odds
        units = max(0, kelly * 0.3) * 100 
        
        rating = "PASS"
        if units > 3.0: rating = "💎 ATOMIC"
        elif units > 1.5: rating = "🟢 NUCLEAR"
        elif units > 0.5: rating = "🟡 KINETIC"
        
        return units, rating, win_prob * 100

# ==========================================
# 📂 5. DATA REFINERY (INGESTION)
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
    def parse_game_info(df):
        if 'Game Info' in df.columns: return df['Game Info']
        elif 'GAME INFO' in df.columns: return df['GAME INFO']
        return "Unknown"

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
        
        # Fill
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
    return rules

def optimize_lineup(df, config):
    # SLATE FILTERING
    pool = df[df['sport'] == config['sport']].copy()
    if config['slate_games']:
        pool = pool[pool['game_info'].isin(config['slate_games'])].reset_index(drop=True)
        
    pool = pool[pool['projection'] > 0].reset_index(drop=True)
    if config['bans']: pool = pool[~pool['name'].isin(config['bans'])].reset_index(drop=True)
    if pool.empty: return None

    # SHOWDOWN LOGIC
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
    html = f"""<html><body><h2>TITAN REPORT</h2><hr>"""
    for _, row in df.head(15).iterrows():
        html += f"<div><b>{row['name']}</b> ({row['position']}) ${row['salary']}</div>"
    html += "</body></html>"
    return base64.b64encode(html.encode()).decode()

# ==========================================
# 🖥️ 8. DASHBOARD
# ==========================================

conn = init_db()
st.sidebar.title("TITAN OMNI")
st.sidebar.caption("The Singularity")

# BANKROLL
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

sport = st.sidebar.selectbox("Sport", ["NFL", "NBA", "MLB", "CFB", "NHL", "PGA"])
site = st.sidebar.selectbox("Site", ["DK", "FD", "Yahoo", "PrizePicks"])

tabs = st.tabs(["1. 📡 Fusion", "2. 🏰 Optimizer", "3. 🔮 Simulation", "4. 🚀 Props", "5. 🧮 Parlay", "6. 🤖 Intel"])

# --- TAB 1: DATA ---
with tabs[0]:
    st.markdown("### 📡 Data Command")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("DFS Data")
        files = st.file_uploader("Upload Salary CSVs", accept_multiple_files=True, key="dfs")
        if st.button("🧬 Fuse DFS Data"):
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
                st.success(f"DFS Pool: {len(st.session_state['dfs_pool'])} players.")

    with col2:
        st.subheader("Prop Data")
        if st.button("☁️ Sync APIs"):
            gw = UniversalGateway()
            with st.spinner("Fetching data..."):
                # 1. Props
                props = gw.fetch_live_props(sport)
                if not props.empty:
                    ref = DataRefinery()
                    cln = ref.ingest(props, sport)
                    st.session_state['prop_pool'] = ref.merge(st.session_state['prop_pool'], cln)
                
                # 2. Team Momentum
                mom = gw.fetch_team_momentum(sport)
                st.session_state['api_data'] = mom
                st.success("API Sync Complete.")

        prop_files = st.file_uploader("Upload Prop CSVs", accept_multiple_files=True, key="props")
        if st.button("📥 Load Prop CSV"):
            if prop_files:
                ref = DataRefinery()
                new_data = pd.DataFrame()
                for f in prop_files:
                    try:
                        try: raw = pd.read_csv(f, encoding='utf-8-sig')
                        except: raw = pd.read_excel(f)
                        new_data = ref.merge(new_data, ref.ingest(raw, sport))
                    except: pass
                st.session_state['prop_pool'] = ref.merge(st.session_state['prop_pool'], new_data)
                st.success("Prop Pool Updated.")

# --- TAB 2: OPTIMIZER ---
with tabs[1]:
    pool = st.session_state['dfs_pool']
    active = pool[pool['sport'] == sport] if not pool.empty else pd.DataFrame()
    
    if active.empty:
        st.warning(f"No {sport} DFS data. Go to Tab 1.")
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
        
        if st.button("⚡ Build Lineups"):
            cfg = {'sport':sport, 'site':site, 'mode':mode, 'count':count, 'locks':locks, 'bans':bans, 'stack':stack, 'sim':False, 'slate_games':slate}
            res = optimize_lineup(st.session_state['dfs_pool'], cfg)
            
            if res is not None:
                st.dataframe(res)
                st.markdown(f'<a href="data:text/html;base64,{get_html_report(res)}" download="report.html">📥 Download Cheat Sheet</a>', unsafe_allow_html=True)
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
    
    if active.empty: st.warning("No props.")
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
        active['pick'] = np.where(active['projection'] > active['prop_line'], "OVER", "UNDER")
        options = active['name'] + " (" + active['pick'] + ")"
        selection = st.multiselect("Build Slip", options)
        if selection:
            prob = 1.0
            for s in selection:
                row = active[active['name'] == s.split(" (")[0]].iloc[0]
                edge = abs(row['projection'] - row['prop_line']) / row['prop_line']
                prob *= min(0.75, 0.52 + (edge * 0.5))
            st.metric("Win Probability", f"{prob*100:.1f}%")

# --- TAB 6: INTEL ---
with tabs[5]:
    if st.button("Run AI Scout"):
        st.session_state['ai_intel'] = run_web_scout(sport)
        st.success("Updated")
    st.write(st.session_state['ai_intel'])
