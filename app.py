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
st.set_page_config(layout="wide", page_title="TITAN OMNI: GOD KING", page_icon="👑")

st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    
    /* COMPONENTS */
    .titan-card { background: #111; border: 1px solid #333; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #00d2ff; }
    .titan-card-prop { border-left: 4px solid #00ff41; }
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

# KEYS
SPORTSDATAIO_KEY = "dda563b328d34b80a38c26cd43223614"
ODDS_API_KEY = "a31f629075c27927fe99b097b51e1717"

# ==========================================
# 📡 2. EXTERNAL DATA ENGINE (APIs)
# ==========================================

class SportsDataIO:
    def __init__(self, api_key):
        self.key = api_key
        self.base = "https://api.sportsdata.io/v3"

    def get_team_stats(self, sport):
        """Fetches Standings/Stats to calculate 'Team Momentum'."""
        # Note: Developer keys often restricted to NFL. We try gracefully.
        if sport == 'NFL':
            url = f"{self.base}/nfl/scores/json/Standings/2024?key={self.key}" # Adjust year as needed
        elif sport == 'NBA':
            url = f"{self.base}/nba/scores/json/Standings/2025?key={self.key}"
        else:
            return {}

        try:
            res = requests.get(url)
            if res.status_code == 200:
                data = res.json()
                # Create Dictionary: {Team: WinPercentage}
                stats = {}
                for team in data:
                    # Normalize keys based on sport response structure
                    name = team.get('Team') or team.get('Key')
                    wins = team.get('Wins', 0)
                    losses = team.get('Losses', 1)
                    if name:
                        stats[name] = wins / (wins + losses) if (wins+losses) > 0 else 0.5
                return stats
            return {}
        except: return {}

class OddsAPI:
    def __init__(self, api_key):
        self.key = api_key
    
    def get_lines(self, sport):
        sport_map = {'NFL': 'americanfootball_nfl', 'NBA': 'basketball_nba', 'MLB': 'baseball_mlb'}
        key = sport_map.get(sport)
        if not key: return pd.DataFrame()
        
        url = f"https://api.the-odds-api.com/v4/sports/{key}/odds/?regions=us&markets=h2h,totals&apiKey={self.key}"
        try:
            res = requests.get(url)
            if res.status_code == 200:
                # Simplified processing: Just return a dataframe of events
                return pd.DataFrame(res.json())
            return pd.DataFrame()
        except: return pd.DataFrame()

# ==========================================
# 💾 3. DATABASE (BANKROLL)
# ==========================================

def init_db():
    conn = sqlite3.connect('titan_god.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS bankroll (id INTEGER PRIMARY KEY, date TEXT, amount REAL, notes TEXT)''')
    c.execute('SELECT count(*) FROM bankroll')
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO bankroll (date, amount, notes) VALUES (?, ?, ?)", 
                  (datetime.now().strftime("%Y-%m-%d"), 1000.0, 'Genesis'))
        conn.commit()
    return conn

def get_bankroll(conn):
    return pd.read_sql("SELECT * FROM bankroll", conn).iloc[-1]['amount']

def update_bankroll(conn, amount, note):
    conn.execute("INSERT INTO bankroll (date, amount, notes) VALUES (?, ?, ?)", 
                 (datetime.now().strftime("%Y-%m-%d"), float(amount), note))
    conn.commit()

# ==========================================
# 🧠 4. TITAN BRAIN (MATH)
# ==========================================

class TitanBrain:
    def __init__(self, bankroll): self.bankroll = bankroll

    def calculate_prop_edge(self, row):
        if row.get('prop_line', 0) <= 0 or row.get('projection', 0) <= 0: return 0, "No Data", 0.0
        
        # Apply Team Momentum Boost if available
        boost = row.get('team_momentum', 0.5) # Default neutral
        proj = row['projection']
        
        # If team is winning > 60%, slight boost to players
        if boost > 0.60: proj *= 1.05
        
        edge_raw = abs(proj - row['prop_line']) / row['prop_line']
        win_prob = min(0.75, 0.52 + (edge_raw * 0.5))
        
        odds = 0.909
        kelly = ((odds * win_prob) - (1 - win_prob)) / odds
        units = max(0, kelly * 0.3) * 100
        
        rating = "PASS"
        if units > 3.0: rating = "💎 ATOMIC"
        elif units > 1.5: rating = "🟢 NUCLEAR"
        elif units > 0.5: rating = "🟡 KINETIC"
        
        return units, rating, win_prob

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
            'prop_line': ['O/U', 'PROP', 'LINE', 'STRIKE', 'TOTAL']
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
            q = f"{sport} dfs sleepers value plays injury report today"
            for r in list(ddgs.text(q, max_results=5)):
                blob = (r['title'] + " " + r['body']).lower()
                intel[blob] = 1
    except: pass
    return intel

# ==========================================
# 🏭 7. STRUCTURAL OPTIMIZER
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
    pool = df[df['sport'] == config['sport']].copy()
    pool = pool[pool['projection'] > 0].reset_index(drop=True)
    
    if config['bans']: pool = pool[~pool['name'].isin(config['bans'])].reset_index(drop=True)
    if pool.empty: return None

    # Showdown Expansion
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
        
        volatility = 0.15 if config['sim'] else 0.0
        # Boost projection if Team Momentum is high
        if 'team_momentum' in pool.columns:
            pool['sim'] = (pool['projection'] * pool['team_momentum'].clip(0.9, 1.1)) * np.random.normal(1.0, volatility, len(pool))
        else:
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

        # Stacking
        if config['mode'] == 'Classic' and config['sport'] == 'NFL' and config.get('stack'):
            qbs = pool[pool['pos_id'] == 'QB']
            for qb in qbs.index:
                tm = pool.loc[qb, 'team']
                mates = pool[(pool['team'] == tm) & (pool['pos_id'].str.contains("WR|TE"))]
                if not mates.empty: prob += pulp.lpSum([x[m] for m in mates.index]) >= x[qb]

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == 1:
            sel = [p for p in pool.index if x[p].varValue == 1]
            lu = pool.loc[sel].copy()
            lu['Lineup_ID'] = i+1
            lineups.append(lu)
            prob += pulp.lpSum([x[p] for p in sel]) <= rules['size'] - 1
            
    return pd.concat(lineups) if lineups else None

def get_html_report(df):
    html = f"""<html><body><h2>TITAN GOD KING REPORT</h2><hr>"""
    for _, row in df.head(15).iterrows():
        html += f"<div><b>{row['name']}</b> ({row['position']}) ${row['salary']}</div>"
    html += "</body></html>"
    return base64.b64encode(html.encode()).decode()

# ==========================================
# 🖥️ 8. DASHBOARD
# ==========================================

conn = init_db()
st.sidebar.title("TITAN OMNI")
st.sidebar.caption("God King Edition")

current_bank = get_bankroll(conn)
st.sidebar.metric("🏦 Bankroll", f"${current_bank:,.2f}")
with st.sidebar.expander("Update Funds"):
    new = st.number_input("New Balance", value=current_bank)
    if st.button("Update"):
        update_bankroll(conn, new, "Manual")
        st.rerun()

sport = st.sidebar.selectbox("Sport", ["NFL", "NBA", "MLB", "CFB", "NHL", "PGA"])
site = st.sidebar.selectbox("Site", ["DK", "FD", "Yahoo", "PrizePicks"])

tabs = st.tabs(["1. 📡 Fusion", "2. 🏰 Optimizer", "3. 🔮 Simulation", "4. 🚀 Props", "5. 🧮 Parlay"])

# --- TAB 1 ---
with tabs[0]:
    st.markdown("### 📡 Data Fusion Engine")
    
    col_api, col_file = st.columns(2)
    with col_api:
        if st.button("☁️ Sync SportsDataIO"):
            with st.spinner("Fetching Standings & Momentum..."):
                sd = SportsDataIO(SPORTSDATAIO_KEY)
                team_data = sd.get_team_stats(sport)
                if team_data:
                    st.session_state['api_data'] = team_data
                    st.success(f"Synced {len(team_data)} teams from API.")
                else:
                    st.warning("API Sync Empty (Check Season/Key)")

    with col_file:
        files = st.file_uploader("Upload CSVs", accept_multiple_files=True)
    
    if st.button("🧬 Fuse & Process"):
        if files:
            ref = DataRefinery()
            new_data = pd.DataFrame()
            for f in files:
                try:
                    try: raw = pd.read_csv(f, encoding='utf-8-sig')
                    except: raw = pd.read_excel(f)
                    new_data = ref.merge(new_data, ref.ingest(raw, sport))
                except: pass
            
            # Enrich with API Data
            if st.session_state['api_data']:
                # Map team momentum to players
                new_data['team_momentum'] = new_data['team'].map(st.session_state['api_data']).fillna(0.5)
            
            st.session_state['dfs_pool'] = ref.merge(st.session_state['dfs_pool'], new_data)
            st.session_state['prop_pool'] = st.session_state['dfs_pool'][st.session_state['dfs_pool']['prop_line'] > 0]
            st.success(f"Fused {len(new_data)} players.")

    if st.button("🛰️ Run AI Scout"):
        st.session_state['ai_intel'] = run_web_scout(sport)
        st.success("AI Updated")

# --- TAB 2 ---
with tabs[1]:
    pool = st.session_state['dfs_pool']
    if pool.empty or 'sport' not in pool.columns:
        st.warning("No data.")
    else:
        active = pool[pool['sport'] == sport]
        if active.empty: st.warning(f"No {sport} data.")
        else:
            c1, c2, c3 = st.columns(3)
            mode = c1.radio("Mode", ["Classic", "Showdown"])
            count = c2.slider("Lineups", 1, 50, 10)
            stack = c3.checkbox("Stacking (NFL)", value=(sport=='NFL'))
            
            names = sorted(active['name'].unique())
            locks = st.multiselect("Lock", names)
            bans = st.multiselect("Ban", names)
            
            if st.button("⚡ Generate Lineups"):
                cfg = {'sport':sport, 'site':site, 'mode':mode, 'count':count, 'locks':locks, 'bans':bans, 'stack':stack, 'sim':False}
                res = optimize_lineup(st.session_state['dfs_pool'], cfg)
                
                if res is not None:
                    st.dataframe(res)
                    st.markdown(f'<a href="data:text/html;base64,{get_html_report(res)}" download="report.html">📥 Download Report</a>', unsafe_allow_html=True)
                    
                    fig = px.treemap(res, path=['team'], values='projection', title="Team Exposure")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Optimization Failed.")

# --- TAB 3 ---
with tabs[2]:
    if st.button("🎲 Run Simulation"):
        with st.spinner("Simulating..."):
            cfg = {'sport':sport, 'site':site, 'mode':"Classic", 'count':50, 'locks':[], 'bans':[], 'stack':True, 'sim':True}
            res = optimize_lineup(st.session_state['dfs_pool'], cfg)
            if res is not None:
                exp = res['name'].value_counts(normalize=True).mul(100).reset_index()
                st.plotly_chart(px.bar(exp.head(15), x='proportion', y='name', orientation='h'))

# --- TAB 4 ---
with tabs[3]:
    pool = st.session_state['prop_pool']
    if pool.empty or 'sport' not in pool.columns: st.warning("No props.")
    else:
        active = pool[pool['sport'] == sport].copy()
        if active.empty: st.warning("No props for sport.")
        else:
            brain = TitanBrain(current_bank)
            active['pick'] = np.where(active['projection'] > active['prop_line'], "OVER", "UNDER")
            res = active.apply(lambda x: brain.calculate_prop_edge(x), axis=1, result_type='expand')
            active['units'] = res[0]
            active['rating'] = res[1]
            active['win_prob'] = res[2]
            st.dataframe(active[active['units'] > 0].sort_values('win_prob', ascending=False))

# --- TAB 5 ---
with tabs[4]:
    pool = st.session_state['prop_pool']
    if pool.empty or 'sport' not in pool.columns: st.warning("No props.")
    else:
        active = pool[pool['sport'] == sport].copy()
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
