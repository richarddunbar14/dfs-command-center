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
st.set_page_config(layout="wide", page_title="TITAN OMNI: FINAL", page_icon="🏛️")

st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    .titan-card { background: #111; border: 1px solid #333; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 10px; border-left: 4px solid #00d2ff; }
    .titan-card-prop { border-left: 4px solid #00ff41; }
    div[data-testid="stMetricValue"] { color: #00ff41; text-shadow: 0 0 10px rgba(0,255,65,0.3); }
    .stButton>button { width: 100%; border-radius: 4px; font-weight: 800; background: linear-gradient(90deg, #111 0%, #222 100%); color: #00d2ff; border: 1px solid #00d2ff; }
    .stButton>button:hover { background: #00d2ff; color: #000; }
</style>
""", unsafe_allow_html=True)

# Session State
if 'dfs_pool' not in st.session_state: st.session_state['dfs_pool'] = pd.DataFrame()
if 'prop_pool' not in st.session_state: st.session_state['prop_pool'] = pd.DataFrame()
if 'ai_intel' not in st.session_state: st.session_state['ai_intel'] = {}
if 'api_log' not in st.session_state: st.session_state['api_log'] = []

# KEYCHAIN
RAPID_KEY = "cf5f5b1102mshedbe3c14b0eb432p176b16jsnf9b2d93c804c"
SPORTSDATAIO_KEY = "dda563b328d34b80a38c26cd43223614"
ODDS_API_KEY = "a31f629075c27927fe99b097b51e1717"

# ==========================================
# 📡 2. OMNISCIENT GATEWAY (RapidAPI + ESPN Fallback)
# ==========================================

class OmniscientGateway:
    def __init__(self):
        self.rapid_headers = {"X-RapidAPI-Key": RAPID_KEY, "Content-Type": "application/json"}
        self.odds_key = ODDS_API_KEY

    def fetch_master_data(self, sport):
        log = []
        dfs_data = []
        today_compact = datetime.now().strftime("%Y%m%d")
        
        # --- 1. RAPID API (PRIMARY) ---
        try:
            if sport == 'NFL':
                url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getDFSProjections"
                self.rapid_headers["X-RapidAPI-Host"] = "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
                res = requests.get(url, headers=self.rapid_headers, params={"gameDate": today_compact, "slate": "main"})
                if res.status_code == 200:
                    raw = res.json().get('body', [])
                    for p in raw:
                        if isinstance(p, dict):
                            dfs_data.append({
                                'name': p.get('player'), 'salary': float(p.get('salary', 0) or 0),
                                'projection': float(p.get('fantasyPoints', 0) or 0), 
                                'position': p.get('pos', 'FLEX'), 'team': p.get('team', 'FA'), 
                                'sport': 'NFL', 'source': 'RapidAPI'
                            })

            elif sport == 'NBA':
                url = "https://api-nba-v1.p.rapidapi.com/games"
                self.rapid_headers["X-RapidAPI-Host"] = "api-nba-v1.p.rapidapi.com"
                requests.get(url, headers=self.rapid_headers, params={"date": datetime.now().strftime("%Y-%m-%d")})
                # (NBA requires specific game ID iteration, simplified here to ping check)
                
            # Add other sports here...
            
        except Exception as e: log.append(f"RapidAPI Error: {e}")

        # --- 2. ESPN FALLBACK (If RapidAPI Empty) ---
        if not dfs_data:
            log.append(f"⚠️ RapidAPI empty for {sport}. Engaging ESPN Fallback...")
            try:
                espn_map = {
                    'NFL': ('football', 'nfl'), 'NBA': ('basketball', 'nba'), 
                    'MLB': ('baseball', 'mlb'), 'NHL': ('hockey', 'nhl'),
                    'CFB': ('football', 'college-football'), 'CBB': ('basketball', 'mens-college-basketball'),
                    'WNBA': ('basketball', 'wnba')
                }
                if sport in espn_map:
                    path = espn_map[sport]
                    url = f"https://site.api.espn.com/apis/site/v2/sports/{path[0]}/{path[1]}/scoreboard"
                    res = requests.get(url).json()
                    for event in res.get('events', []):
                        comp = event['competitions'][0]
                        home = comp['competitors'][0]['team']['displayName']
                        away = comp['competitors'][1]['team']['displayName']
                        matchup = f"{away} @ {home}"
                        # Generate Placeholder Rosters for Optimizer functionality
                        dfs_data.append({'name': f"{home} Stack", 'position': 'FLEX', 'salary': 5000, 'projection': 20, 'team': home, 'game_info': matchup, 'sport': sport})
                        dfs_data.append({'name': f"{away} Stack", 'position': 'FLEX', 'salary': 5000, 'projection': 20, 'team': away, 'game_info': matchup, 'sport': sport})
                    log.append(f"✅ ESPN: Loaded Schedule for {sport}")
            except Exception as e: log.append(f"ESPN Error: {e}")

        # --- 3. ODDS API (PROPS) ---
        prop_data = []
        try:
            odds_map = {'NFL': 'americanfootball_nfl', 'NBA': 'basketball_nba', 'MLB': 'baseball_mlb', 'NHL': 'icehockey_nhl', 'CFB': 'americanfootball_ncaaf'}
            key = odds_map.get(sport)
            if key:
                url = f"https://api.the-odds-api.com/v4/sports/{key}/odds"
                params = {'apiKey': self.odds_key, 'regions': 'us', 'markets': 'player_points,h2h', 'oddsFormat': 'american'}
                res = requests.get(url, params=params).json()
                if isinstance(res, list):
                    for event in res:
                        game = f"{event['home_team']} vs {event['away_team']}"
                        for book in event['bookmakers']:
                            if book['key'] in ['draftkings', 'fanduel']:
                                for m in book['markets']:
                                    if 'player' in m['key']:
                                        for o in m['outcomes']:
                                            prop_data.append({
                                                'name': o['description'], 'prop_line': o.get('point', 0), 
                                                'odds': o.get('price', 0), 'game_info': game, 'sport': sport
                                            })
        except: pass

        st.session_state['api_log'] = log
        return pd.DataFrame(dfs_data), pd.DataFrame(prop_data)

# ==========================================
# 💾 3. DATABASE
# ==========================================

def init_db():
    conn = sqlite3.connect('titan_final.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS bankroll (id INTEGER PRIMARY KEY, date TEXT, amount REAL, notes TEXT)''')
    c.execute('SELECT count(*) FROM bankroll')
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO bankroll (date, amount, notes) VALUES (?, ?, ?)", (datetime.now().strftime("%Y-%m-%d"), 1000.0, 'Init'))
        conn.commit()
    return conn

def get_bankroll_history(conn): return pd.read_sql("SELECT * FROM bankroll", conn)
def update_bankroll(conn, amount, note):
    conn.execute("INSERT INTO bankroll (date, amount, notes) VALUES (?, ?, ?)", (datetime.now().strftime("%Y-%m-%d %H:%M"), float(amount), note))
    conn.commit()

# ==========================================
# 🧠 4. TITAN BRAIN
# ==========================================

class TitanBrain:
    def __init__(self, bankroll): self.bankroll = bankroll
    def calculate_prop_edge(self, row):
        if row.get('prop_line', 0) <= 0: return 0, "No Data", 0.0
        proj = row.get('projection', row['prop_line']) # Fallback
        edge_raw = abs(proj - row['prop_line']) / row['prop_line']
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
                    if target in ['projection', 'salary', 'prop_line']: std[target] = df[s].apply(DataRefinery.clean_curr)
                    elif target == 'name': std[target] = df[s].astype(str).apply(lambda x: x.split('(')[0].strip())
                    else: std[target] = df[s].astype(str).str.strip()
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
# 🏭 6. OPTIMIZER
# ==========================================

def get_roster_rules(sport, site, mode):
    rules = {'size': 0, 'cap': 50000, 'constraints': []}
    if site == 'DK': rules['cap'] = 50000
    elif site == 'FD': rules['cap'] = 60000
    elif site == 'Yahoo': rules['cap'] = 200

    if mode == 'Showdown':
        rules['size'] = 6 if site == 'DK' else 5
        rules['constraints'].append(('CPT', 1, 1))
        return rules

    # --- ALL SPORTS RULES (UPDATED) ---
    if sport == 'NFL':
        rules['size'] = 9
        if site == 'DK': rules['constraints'] = [('QB', 1, 1), ('DST', 1, 1), ('RB', 2, 3), ('WR', 3, 4), ('TE', 1, 2)]
        elif site == 'FD': rules['constraints'] = [('QB', 1, 1), ('DEF', 1, 1), ('RB', 2, 3), ('WR', 3, 4), ('TE', 1, 2)]
    elif sport == 'NBA':
        rules['size'] = 8
        if site == 'DK': rules['constraints'] = [('PG', 1, 3), ('SG', 1, 3), ('SF', 1, 3), ('PF', 1, 3), ('C', 1, 2)]
    elif sport == 'MLB':
        rules['size'] = 10
        if site == 'DK': rules['constraints'] = [('P', 2, 2)]
    elif sport == 'NHL':
        rules['size'] = 9
        if site == 'DK': rules['constraints'] = [('C', 2, 3), ('W', 3, 4), ('D', 2, 3), ('G', 1, 1)]
    elif sport == 'CFB':
        rules['size'] = 9
        if site == 'DK': rules['constraints'] = [('QB', 1, 2), ('RB', 2, 4), ('WR', 3, 5)]
    elif sport == 'CBB':
        rules['size'] = 8
        if site == 'DK': rules['constraints'] = [('G', 3, 5), ('F', 3, 5)]
    elif sport == 'WNBA':
        rules['size'] = 6
        if site == 'DK': rules['constraints'] = [('G', 2, 3), ('F', 3, 4)]
    elif sport == 'PGA':
        rules['size'] = 6
    
    # Fallback
    if rules['size'] == 0: rules['size'] = 6
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
        
        pool['sim'] = pool['projection'] * np.random.normal(1.0, 0.15 if config['sim'] else 0.0, len(pool))
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
    for _, row in df.head(20).iterrows():
        html += f"<div><b>{row['name']}</b> ({row['position']}) | Sal: ${row['salary']} | Proj: {row['projection']:.1f}</div>"
    return base64.b64encode(html.encode()).decode()

# ==========================================
# 📡 7. AI SCOUT
# ==========================================
def run_web_scout(sport):
    intel = {}
    try:
        with DDGS() as ddgs:
            q = f"{sport} dfs sleepers value plays analysis today"
            for r in list(ddgs.text(q, max_results=5)):
                intel[(r['title'] + " " + r['body']).lower()] = 1
    except: pass
    return intel

# ==========================================
# 🖥️ 8. DASHBOARD
# ==========================================
conn = init_db()
st.sidebar.title("TITAN OMNI")
st.sidebar.caption("Final Architect Edition")

history = get_bankroll_history(conn)
current_bank = history.iloc[-1]['amount']
st.sidebar.metric("🏦 Bankroll", f"${current_bank:,.2f}")
if len(history) > 1:
    fig = px.line(history, x='date', y='amount', height=150)
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.sidebar.plotly_chart(fig, use_container_width=True)

with st.sidebar.expander("Update Funds"):
    new = st.number_input("New Balance", value=current_bank)
    if st.button("Update"): update_bankroll(conn, new, "Manual"); st.rerun()

sport = st.sidebar.selectbox("Sport", ["NFL", "NBA", "MLB", "CFB", "CBB", "WNBA", "NHL", "PGA"])
site = st.sidebar.selectbox("Site", ["DK", "FD", "Yahoo", "PrizePicks"])

tabs = st.tabs(["1. 📡 Fusion", "2. 🏰 Optimizer", "3. 🔮 Simulation", "4. 🚀 Props", "5. 🧮 Parlay"])

# --- TAB 1 ---
with tabs[0]:
    st.markdown("### 📡 Data Fusion Engine")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("☁️ AUTO-SYNC APIS"):
            gw = OmniscientGateway()
            with st.spinner(f"Routing to {sport} API endpoints..."):
                dfs, props = gw.fetch_master_data(sport)
                
                with st.expander("API Logs"):
                    for l in st.session_state['api_log']: st.write(l)
                
                if not dfs.empty:
                    ref = DataRefinery()
                    cln = ref.ingest(dfs, sport)
                    st.session_state['dfs_pool'] = ref.merge(st.session_state['dfs_pool'], cln)
                    st.success(f"DFS: {len(cln)} items loaded.")
                
                if not props.empty:
                    ref = DataRefinery()
                    cln = ref.ingest(props, sport)
                    st.session_state['prop_pool'] = ref.merge(st.session_state['prop_pool'], cln)
                    st.success(f"Props: {len(cln)} lines loaded.")

    with col2:
        files = st.file_uploader("Upload CSVs", accept_multiple_files=True)
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
                st.success("Fused.")

    if st.button("🛰️ Run AI Scout"):
        st.session_state['ai_intel'] = run_web_scout(sport)
        st.success("AI Updated")

# --- TAB 2 ---
with tabs[1]:
    pool = st.session_state['dfs_pool']
    if pool.empty or 'sport' not in pool.columns: st.warning("No data.")
    else:
        active = pool[pool['sport'] == sport]
        if active.empty: st.warning(f"No {sport} data.")
        else:
            games = sorted(active['game_info'].astype(str).unique())
            slate = st.multiselect("🗓️ Filter Slate", games, default=games)
            c1, c2 = st.columns(2)
            mode = c1.radio("Mode", ["Classic", "Showdown"])
            count = c2.slider("Lineups", 1, 50, 10)
            stack = st.checkbox("Stacking (NFL)", value=(sport=='NFL'))
            
            names = sorted(active['name'].unique())
            locks = st.multiselect("Lock", names)
            bans = st.multiselect("Ban", names)
            
            if st.button("⚡ Generate"):
                cfg = {'sport':sport, 'site':site, 'mode':mode, 'count':count, 'locks':locks, 'bans':bans, 'stack':stack, 'sim':False, 'slate_games':slate}
                res = optimize_lineup(st.session_state['dfs_pool'], cfg)
                if res is not None:
                    st.dataframe(res)
                    st.markdown(f'<a href="data:text/html;base64,{get_html_report(res)}" download="report.html">📥 Download</a>', unsafe_allow_html=True)
                else: st.error("Optimization Failed.")

# --- TAB 3 ---
with tabs[2]:
    if st.button("🎲 Simulate"):
        cfg = {'sport':sport, 'site':site, 'mode':"Classic", 'count':50, 'locks':[], 'bans':[], 'stack':True, 'sim':True, 'slate_games':[]}
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
            active['units'] = res[0]; active['rating'] = res[1]; active['win_prob'] = res[2]
            st.dataframe(active[active['units'] > 0].sort_values('win_prob', ascending=False))

# --- TAB 5 ---
with tabs[4]:
    pool = st.session_state['prop_pool']
    active = pool[pool['sport'] == sport].copy() if not pool.empty else pd.DataFrame()
    if active.empty: st.warning("No props.")
    else:
        active['pick'] = np.where(active['projection'] > active['prop_line'], "OVER", "UNDER")
        sel = st.multiselect("Build Slip", active['name'] + " (" + active['pick'] + ")")
        if sel:
            prob = 1.0
            for s in sel:
                row = active[active['name'] == s.split(" (")[0]].iloc[0]
                edge = abs(row['projection'] - row['prop_line']) / row['prop_line']
                prob *= min(0.75, 0.52 + (edge * 0.5))
            st.metric("Win Prob", f"{prob*100:.1f}%")
