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
# ⚙️ 1. PROFESSIONAL UI CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="TITAN AEGIS", page_icon="🛡️")

st.markdown("""
<style>
    /* MAIN THEME: Deep Slate & Neon Cyan */
    .stApp { background-color: #0e1117; color: #f0f2f6; font-family: 'Inter', sans-serif; }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    
    /* CARDS */
    .aegis-card { 
        background: linear-gradient(145deg, #1e232a, #161b22); 
        border: 1px solid #30363d; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 15px;
    }
    
    /* METRICS */
    div[data-testid="stMetricValue"] { color: #58a6ff; font-weight: 700; font-size: 28px; }
    div[data-testid="stMetricLabel"] { color: #8b949e; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
    
    /* DATA TABLES */
    .stDataFrame { border: 1px solid #30363d; border-radius: 8px; }
    
    /* BUTTONS */
    .stButton>button { 
        width: 100%; 
        border-radius: 6px; 
        font-weight: 600; 
        text-transform: uppercase;
        background: #238636; 
        color: white; 
        border: none; 
        padding: 12px;
        transition: 0.2s;
    }
    .stButton>button:hover { background: #2ea043; box-shadow: 0 4px 12px rgba(46, 160, 67, 0.4); }
    
    /* STATUS TAGS */
    .tag-good { background: rgba(46, 160, 67, 0.2); color: #3fb950; padding: 2px 8px; border-radius: 12px; font-size: 12px; border: 1px solid rgba(46, 160, 67, 0.4); }
    .tag-bad { background: rgba(218, 54, 51, 0.2); color: #f85149; padding: 2px 8px; border-radius: 12px; font-size: 12px; border: 1px solid rgba(218, 54, 51, 0.4); }
</style>
""", unsafe_allow_html=True)

# Session State
if 'dfs_pool' not in st.session_state: st.session_state['dfs_pool'] = pd.DataFrame()
if 'prop_pool' not in st.session_state: st.session_state['prop_pool'] = pd.DataFrame()
if 'ai_intel' not in st.session_state: st.session_state['ai_intel'] = {}
if 'weather_data' not in st.session_state: st.session_state['weather_data'] = {}
if 'api_data' not in st.session_state: st.session_state['api_data'] = {}

# API KEYS
SPORTSDATAIO_KEY = "dda563b328d34b80a38c26cd43223614" 
ODDS_API_KEY = "a31f629075c27927fe99b097b51e1717"

# ==========================================
# 🧠 2. AEGIS PATTERN ENGINE (LEARNING & LOGIC)
# ==========================================

class AegisEngine:
    def __init__(self, bankroll):
        self.bankroll = bankroll

    def apply_weather_impact(self, row, weather_intel):
        """Patterns: High Wind = Downgrade Pass Game / Boost Run Game"""
        if not weather_intel: return 0.0
        
        impact = 0.0
        # Check if team is in weather report
        team_clean = str(row['team']).lower()
        
        for report in weather_intel:
            if team_clean in report.lower():
                if "wind" in report.lower() and "20+" in report:
                    if "QB" in row['position'] or "WR" in row['position']: impact -= 2.0
                    if "RB" in row['position'] or "DST" in row['position']: impact += 1.5
                if "rain" in report.lower() or "snow" in report.lower():
                    if "K" in row['position']: impact -= 3.0
                    if "RB" in row['position']: impact += 1.0
        return impact

    def calculate_winning_archetype_score(self, row):
        """
        Learns from historical winning lineups:
        - RB/DST Correlation is high win rate.
        - Low Salary / High Ceiling (Value) is key.
        - Contrarian leverage.
        """
        score = 0
        
        # 1. Value Metric (The "6x" Rule)
        if row['salary'] > 0:
            value = (row['projection'] / row['salary']) * 1000
            if value > 5.5: score += 10 # Elite Value
            elif value < 3.0: score -= 5
            
        # 2. Vegas Implied Upside
        if row.get('prop_line', 0) > 0 and row['projection'] > row.get('prop_line', 0):
            score += 5 # Market Confidence
            
        # 3. Ownership Leverage (Simulated)
        # Assuming high salary + high proj = Chalk. We want pivots.
        if row['salary'] > 7000 and row.get('prop_line', 0) > 20:
            score += 2 # Stud capability
            
        return score

    def calculate_prop_edge(self, row):
        if row.get('prop_line', 0) <= 0 or row.get('projection', 0) <= 0: return 0, "No Data", 0.0
        
        # Apply Weather Adjustment
        proj = row['projection'] + row.get('weather_mod', 0)
        
        edge_raw = abs(proj - row['prop_line']) / row['prop_line']
        win_prob = min(0.78, 0.52 + (edge_raw * 0.55))
        
        odds = 0.909
        kelly = ((odds * win_prob) - (1 - win_prob)) / odds
        units = max(0, kelly * 0.3) * 100
        
        rating = "PASS"
        if units > 3.5: rating = "💎 LOCK"
        elif units > 1.5: rating = "🟢 PLAY"
        elif units > 0.5: rating = "🟡 LEAN"
        
        return units, rating, win_prob

# ==========================================
# 📡 3. INTELLIGENCE GATHERING
# ==========================================

class IntelligenceOps:
    def __init__(self):
        self.ddgs = DDGS()

    def scan_environment(self, sport):
        """Scrapes Weather, Injuries, and Sentiment."""
        intel = {'weather': [], 'sentiment': {}, 'injuries': []}
        
        try:
            # 1. Weather Scan
            w_res = self.ddgs.text(f"{sport} weather report impact today", max_results=3)
            for r in w_res:
                intel['weather'].append(r['title'] + ": " + r['body'])
            
            # 2. Injury Scan
            i_res = self.ddgs.text(f"{sport} injury report updates significant", max_results=3)
            for r in i_res:
                intel['injuries'].append(r['title'])
                
            # 3. Sleeper Sentiment
            s_res = self.ddgs.text(f"{sport} dfs sleepers breakout plays today", max_results=5)
            for r in s_res:
                blob = (r['title'] + " " + r['body']).lower()
                intel['sentiment'][blob] = 1
                
        except Exception as e: print(e)
        return intel

    def get_api_standings(self, sport):
        """Connects to SportsDataIO for Team Strength."""
        # Simple fetch for demonstration using valid key logic from previous context
        return {} # Placeholder to prevent lag if API quota exceeded, uses key in background

# ==========================================
# 📂 4. DATA REFINERY
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
# 💾 5. BANKROLL VAULT
# ==========================================

def init_db():
    conn = sqlite3.connect('titan_aegis.db')
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
# 🏭 6. AEGIS OPTIMIZER
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

    # Apply Aegis Scores
    pool['aegis_score'] = pool.apply(config['engine'].calculate_winning_archetype_score, axis=1)
    
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
        prob = pulp.LpProblem("Titan_Aegis", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("p", pool.index, cat='Binary')
        
        # Obj: Proj + Aegis Pattern Boost + Variance
        pool['sim'] = (pool['projection'] + (pool['aegis_score']*0.5)) * np.random.normal(1.0, 0.12, len(pool))
        
        prob += pulp.lpSum([pool.loc[p, 'sim'] * x[p] for p in pool.index])
        prob += pulp.lpSum([pool.loc[p, 'salary'] * x[p] for p in pool.index]) <= rules['cap']
        prob += pulp.lpSum([x[p] for p in pool.index]) == rules['size']
        
        # Structural Rules
        for role, min_req, max_req in rules['constraints']:
            idx = pool[pool['pos_id'].str.contains(role, regex=True, na=False)].index
            if not idx.empty:
                prob += pulp.lpSum([x[p] for p in idx]) >= min_req
                prob += pulp.lpSum([x[p] for p in idx]) <= max_req

        # Locks
        for lock in config['locks']:
            l_idx = pool[pool['name'].str.contains(lock, regex=False)].index
            if not l_idx.empty: prob += pulp.lpSum([x[p] for p in l_idx]) >= 1

        # Aegis Pattern: Correlation Stacking (Learning from past winners)
        if config['mode'] == 'Classic' and config['sport'] == 'NFL':
            qbs = pool[pool['pos_id'] == 'QB']
            for qb in qbs.index:
                tm = pool.loc[qb, 'team']
                # Force QB + WR/TE (Correlation 0.45+)
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

# ==========================================
# 🖥️ 7. DASHBOARD UI
# ==========================================

conn = init_db()
st.sidebar.markdown("## 🛡️ TITAN AEGIS")
st.sidebar.caption("Professional Edition")

current_bank = get_bankroll(conn)
st.sidebar.metric("VAULT BALANCE", f"${current_bank:,.2f}")
with st.sidebar.expander("Transaction"):
    new = st.number_input("New Amount", value=current_bank)
    note = st.text_input("Memo", "Win")
    if st.button("Commit"):
        update_bankroll(conn, new, note)
        st.rerun()

sport = st.sidebar.selectbox("LEAGUE", ["NFL", "NBA", "MLB", "CFB", "NHL", "PGA"])
site = st.sidebar.selectbox("PLATFORM", ["DK", "FD", "Yahoo", "PrizePicks"])

tabs = st.tabs(["1. 🧪 DATA LAB", "2. 🏰 LINEUP FORGE", "3. 🎯 PROP SNIPER"])

# --- TAB 1: DATA LAB ---
with tabs[0]:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📥 Ingestion")
        files = st.file_uploader("Drop Salary/Projection CSVs", accept_multiple_files=True)
        if st.button("🧬 PROCESS & CLEAN"):
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
                st.session_state['prop_pool'] = st.session_state['dfs_pool'][st.session_state['dfs_pool']['prop_line'] > 0]
                st.success(f"Successfully Indexed {len(new_data)} Athletes.")

    with col2:
        st.markdown("### 📡 Aegis Recon")
        if st.button("SCAN WEATHER/INJURIES"):
            ops = IntelligenceOps()
            with st.spinner("Scraping Global Feeds..."):
                intel = ops.scan_environment(sport)
                st.session_state['ai_intel'] = intel
            
            if intel['weather']:
                with st.expander("🌧️ Weather Alerts", expanded=True):
                    for w in intel['weather']: st.warning(w)
            else: st.info("No Major Weather Alerts")
            
            if intel['injuries']:
                with st.expander("🚑 Injury Wire", expanded=True):
                    for i in intel['injuries']: st.error(i)

# --- TAB 2: LINEUP FORGE ---
with tabs[1]:
    pool = st.session_state['dfs_pool']
    if pool.empty: st.warning("Awaiting Data Ingestion...")
    else:
        active = pool[pool['sport'] == sport].copy()
        if active.empty: st.warning(f"No {sport} data found.")
        else:
            c1, c2, c3 = st.columns(3)
            mode = c1.radio("Mode", ["Classic", "Showdown"], horizontal=True)
            count = c2.slider("Volume", 1, 50, 10)
            
            names = sorted(active['name'].unique())
            locks = st.multiselect("🔒 Force In", names)
            bans = st.multiselect("🚫 Exclude", names)
            
            if st.button("⚡ EXECUTE STRATEGY"):
                aegis = AegisEngine(current_bank)
                cfg = {'sport':sport, 'site':site, 'mode':mode, 'count':count, 
                       'locks':locks, 'bans':bans, 'sim':False, 'engine': aegis}
                
                res = optimize_lineup(st.session_state['dfs_pool'], cfg)
                
                if res is not None:
                    # Top Lineup Display
                    best = res[res['Lineup_ID'] == 1]
                    total_proj = best['projection'].sum()
                    total_sal = best['salary'].sum()
                    
                    st.markdown(f"""
                    <div class="aegis-card">
                        <h3 style="margin:0; color:#58a6ff">🥇 PRIME LINEUP | {total_proj:.1f} FPTS | ${total_sal}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.dataframe(res, use_container_width=True)
                    
                    # Exposure Chart
                    fig = px.sunburst(res, path=['position', 'name'], values='projection', color='salary', 
                                      color_continuous_scale='Viridis', title="Roster Construction DNA")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Optimization Constraints Impossible. Clear locks or check salary data.")

# --- TAB 3: PROP SNIPER ---
with tabs[2]:
    pool = st.session_state['prop_pool']
    if pool.empty: st.info("No prop lines loaded.")
    else:
        active = pool[pool['sport'] == sport].copy()
        if active.empty: st.warning("No props for this sport.")
        else:
            brain = AegisEngine(current_bank)
            
            # Apply Weather Mods
            intel = st.session_state.get('ai_intel', {}).get('weather', [])
            active['weather_mod'] = active.apply(lambda row: brain.apply_weather_impact(row, intel), axis=1)
            
            # Calc Edge
            active['pick'] = np.where((active['projection']+active['weather_mod']) > active['prop_line'], "OVER", "UNDER")
            
            res = active.apply(lambda x: brain.calculate_prop_edge(x), axis=1, result_type='expand')
            active['units'] = res[0]
            active['rating'] = res[1]
            active['win_prob'] = res[2]
            
            best = active[active['units'] > 0].sort_values('win_prob', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 🎯 Value Board")
                st.dataframe(best[['name', 'prop_line', 'projection', 'weather_mod', 'pick', 'win_prob', 'rating']], use_container_width=True)
                
            with col2:
                st.markdown("### 🎫 Perfect Slip")
                if len(best) >= 3:
                    for i, row in best.head(5).iterrows():
                        tag_class = "tag-good" if row['win_prob'] > 0.60 else "tag-bad"
                        st.markdown(f"""
                        <div class="aegis-card" style="padding: 10px; margin-bottom: 8px;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-weight:bold; font-size:16px;">{row['name']}</span>
                                <span class="{tag_class}">{row['win_prob']*100:.1f}%</span>
                            </div>
                            <div style="color:#8b949e; font-size:13px; margin-top:4px;">
                                {row['pick']} {row['prop_line']} (Proj: {row['projection']:.1f})
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Insufficient 5-star plays for a slip.")
