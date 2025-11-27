import streamlit as st
import pandas as pd
import numpy as np
import pulp
import re
import time
import sqlite3
import base64
import difflib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from duckduckgo_search import DDGS

# ==========================================
# ⚙️ 1. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="TITAN OMNI: QUANTUM", page_icon="⚛️")

st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    
    /* CARDS & METRICS */
    .titan-card { 
        background: #111; border: 1px solid #333; padding: 15px; border-radius: 8px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 10px; border-left: 4px solid #00d2ff;
    }
    .titan-card-prop { border-left: 4px solid #00ff41; }
    div[data-testid="stMetricValue"] { color: #00ff41; text-shadow: 0 0 10px rgba(0,255,65,0.3); }
    
    /* BUTTONS */
    .stButton>button { 
        width: 100%; border-radius: 4px; font-weight: 800; letter-spacing: 1px; text-transform: uppercase;
        background: linear-gradient(90deg, #111 0%, #222 100%); color: #00d2ff; border: 1px solid #00d2ff; transition: 0.3s;
    }
    .stButton>button:hover { background: #00d2ff; color: #000; box-shadow: 0 0 15px #00d2ff; }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #1a1a1a; border-radius: 4px; color: #888; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #00d2ff; color: #000; }
</style>
""", unsafe_allow_html=True)

# Session State
if 'dfs_pool' not in st.session_state: st.session_state['dfs_pool'] = pd.DataFrame()
if 'prop_pool' not in st.session_state: st.session_state['prop_pool'] = pd.DataFrame()
if 'ai_intel' not in st.session_state: st.session_state['ai_intel'] = {}

# ==========================================
# 💾 2. PERSISTENT DATABASE (BANKROLL)
# ==========================================

def init_db():
    conn = sqlite3.connect('titan_core.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS bankroll 
                 (id INTEGER PRIMARY KEY, date TEXT, amount REAL, notes TEXT)''')
    # Init if empty
    c.execute('SELECT count(*) FROM bankroll')
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO bankroll (date, amount, notes) VALUES (?, ?, ?)", 
                  (datetime.now().strftime("%Y-%m-%d"), 1000.0, 'Genesis Block'))
        conn.commit()
    return conn

def get_bankroll(conn):
    c = conn.cursor()
    c.execute("SELECT amount FROM bankroll ORDER BY id DESC LIMIT 1")
    return c.fetchone()[0]

def update_bankroll(conn, amount, note):
    c = conn.cursor()
    c.execute("INSERT INTO bankroll (date, amount, notes) VALUES (?, ?, ?)", 
              (datetime.now().strftime("%Y-%m-%d"), float(amount), note))
    conn.commit()

# ==========================================
# 🧠 3. TITAN BRAIN (MATH ENGINE)
# ==========================================

class TitanBrain:
    def __init__(self, bankroll):
        self.bankroll = bankroll

    def calculate_prop_edge(self, row):
        if row.get('prop_line', 0) <= 0 or row.get('projection', 0) <= 0: 
            return 0, "No Data", 0.0
        
        proj = row['projection']
        line = row['prop_line']
        edge_raw = abs(proj - line) / line
        
        # Win Prob Model (Logistic)
        win_prob = 0.52 + (edge_raw * 0.5)
        win_prob = min(0.72, win_prob) # Cap at 72%
        
        # Kelly Criterion
        odds = 0.909 # -110 implied
        kelly = ((odds * win_prob) - (1 - win_prob)) / odds
        units = max(0, kelly * 0.3) * 100 # 30% Fractional Kelly
        
        rating = "PASS"
        if units > 3.0: rating = "💎 ATOMIC"
        elif units > 1.5: rating = "🟢 NUCLEAR"
        elif units > 0.5: rating = "🟡 KINETIC"
        
        return units, rating, win_prob * 100

# ==========================================
# 📂 4. AGGRESSIVE DATA REFINERY
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
        
        # Mappings
        maps = {
            'name': ['PLAYER', 'NAME', 'WHO', 'ATHLETE'],
            'projection': ['FPTS', 'PROJ', 'AVG FPTS', 'PTS', 'FC PROJ'],
            'salary': ['SAL', 'SALARY', 'CAP', 'COST'],
            'position': ['POS', 'POSITION', 'ROSTER POSITION'],
            'team': ['TEAM', 'TM', 'SQUAD'],
            'prop_line': ['O/U', 'PROP', 'LINE', 'STRIKE', 'TOTAL', 'STAT']
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
        
        std['sport'] = sport_tag
        return std

    @staticmethod
    def merge(base, new_df):
        if base.empty: return new_df
        if new_df.empty: return base
        # Merge on Name+Sport to prevent sport pollution
        return pd.concat([base, new_df]).drop_duplicates(subset=['name', 'sport'], keep='last').reset_index(drop=True)

# ==========================================
# 📡 5. AI WEB SCOUT
# ==========================================

def run_web_scout(sport):
    intel = {}
    try:
        with DDGS() as ddgs:
            q = f"{sport} dfs sleepers value plays injury news today"
            for r in list(ddgs.text(q, max_results=5)):
                blob = (r['title'] + " " + r['body']).lower()
                intel[blob] = 1
    except: pass
    return intel

# ==========================================
# 🏭 6. SIMULATION & OPTIMIZER
# ==========================================

def optimize_lineup(df, config):
    # 1. Filter
    pool = df[df['sport'] == config['sport']].copy()
    pool = pool[pool['projection'] > 0].reset_index(drop=True)
    
    # 2. Exclusions
    if config['bans']: pool = pool[~pool['name'].isin(config['bans'])].reset_index(drop=True)
    if pool.empty: return None

    # 3. Rules
    cap = 50000 if config['site'] == 'DK' else 60000
    if config['site'] == 'Yahoo': cap = 200
    
    size = 8
    if config['sport'] == 'NFL': size = 9
    elif config['sport'] == 'MLB': size = 10
    
    if config['mode'] == 'Showdown':
        size = 6 if config['site'] == 'DK' else 5
        # Expand Captains
        flex = pool.copy(); flex['pos_id'] = 'FLEX'
        cpt = pool.copy(); cpt['pos_id'] = 'CPT'
        cpt['name'] += " (CPT)"
        cpt['projection'] *= 1.5
        if config['site'] == 'DK': cpt['salary'] *= 1.5
        pool = pd.concat([cpt, flex]).reset_index(drop=True)

    lineups = []
    
    # 4. Iterative Solver
    for i in range(config['count']):
        prob = pulp.LpProblem("Titan_Omni", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("p", pool.index, cat='Binary')
        
        # Objective: Sim Variance
        volatility = 0.15 if config['sim_mode'] else 0.0
        pool['sim'] = pool['projection'] * np.random.normal(1.0, volatility, len(pool))
        prob += pulp.lpSum([pool.loc[p, 'sim'] * x[p] for p in pool.index])
        
        # Constraints
        prob += pulp.lpSum([pool.loc[p, 'salary'] * x[p] for p in pool.index]) <= cap
        prob += pulp.lpSum([x[p] for p in pool.index]) == size
        
        # Locks
        for lock in config['locks']:
            l_idx = pool[pool['name'].str.contains(lock)].index
            if not l_idx.empty: prob += pulp.lpSum([x[p] for p in l_idx]) >= 1

        # Sport Rules
        if config['mode'] == 'Classic':
            if config['sport'] == 'NFL':
                qbs = pool[pool['position'].str.contains("QB")]
                dst = pool[pool['position'].str.contains("DST|DEF|^D$")]
                if not qbs.empty: prob += pulp.lpSum([x[p] for p in qbs.index]) == 1
                if not dst.empty: prob += pulp.lpSum([x[p] for p in dst.index]) == 1
                
                if config['stack']:
                    for qb in qbs.index:
                        tm = pool.loc[qb, 'team']
                        mates = pool[(pool['team'] == tm) & (pool['position'].str.contains("WR|TE"))]
                        if not mates.empty: prob += pulp.lpSum([x[m] for m in mates.index]) >= x[qb]

            elif config['sport'] == 'NBA':
                guards = pool[pool['position'].str.contains("G")]
                if not guards.empty: prob += pulp.lpSum([x[p] for p in guards.index]) >= 3
                
        elif config['mode'] == 'Showdown':
            cpts = pool[pool['pos_id'] == 'CPT']
            prob += pulp.lpSum([x[p] for p in cpts.index]) == 1

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == 1:
            sel = [p for p in pool.index if x[p].varValue == 1]
            lu = pool.loc[sel].copy()
            lu['Lineup_ID'] = i+1
            lineups.append(lu)
            prob += pulp.lpSum([x[p] for p in sel]) <= size - 1
            
    return pd.concat(lineups) if lineups else None

def get_html_report(df):
    html = f"""
    <html><head><style>
        body {{ font-family: sans-serif; background: #f0f0f0; padding: 20px; }}
        .card {{ background: white; padding: 15px; margin: 10px; border-radius: 8px; border-left: 5px solid #00d2ff; }}
        h1 {{ color: #333; }}
    </style></head><body>
    <h1>TITAN OMNI: CHEAT SHEET</h1>
    <h3>Top Plays Generated {datetime.now()}</h3>
    """
    for _, row in df.head(10).iterrows():
        html += f"<div class='card'><b>{row['name']}</b> ({row['position']})<br>Salary: {row['salary']} | Proj: {row['projection']}</div>"
    html += "</body></html>"
    return base64.b64encode(html.encode()).decode()

# ==========================================
# 🖥️ 7. DASHBOARD
# ==========================================

conn = init_db()
st.sidebar.title("TITAN QUANTUM")
st.sidebar.caption("Singularity Edition")

# BANKROLL
current_bank = get_bankroll(conn)
st.sidebar.metric("🏦 Bankroll", f"${current_bank:,.2f}")
with st.sidebar.expander("Update Funds"):
    new_bal = st.number_input("New Balance", value=current_bank)
    note = st.text_input("Note", "Win/Loss")
    if st.button("Update"):
        update_bankroll(conn, new_bal, note)
        st.rerun()

# SETTINGS
sport = st.sidebar.selectbox("Sport", ["NFL", "NBA", "MLB", "CFB", "NHL", "PGA"])
site = st.sidebar.selectbox("Site", ["DK", "FD", "Yahoo", "PrizePicks", "Underdog"])

tabs = st.tabs(["1. 📡 Fusion", "2. 🏰 Optimizer", "3. 🔮 Simulation", "4. 🚀 Props"])

# --- TAB 1: FUSION ---
with tabs[0]:
    st.markdown("### 📡 Data Fusion Engine")
    files = st.file_uploader("Upload CSVs", accept_multiple_files=True)
    
    if st.button("🧬 Fuse & Analyze"):
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
            st.success(f"Ingested {len(new_data)} records for {sport}.")

    if st.button("🛰️ Run AI Scout"):
        with st.spinner("Analyzing Sentiment..."):
            st.session_state['ai_intel'] = run_web_scout(sport)
        st.success(f"Intel Updated: {len(st.session_state['ai_intel'])} sources.")

# --- TAB 2: OPTIMIZER ---
with tabs[1]:
    pool = st.session_state['dfs_pool']
    active = pool[pool['sport'] == sport]
    
    if active.empty:
        st.warning(f"No {sport} data found.")
    else:
        c1, c2, c3 = st.columns(3)
        mode = c1.radio("Mode", ["Classic", "Showdown"])
        count = c2.slider("Lineups", 1, 50, 10)
        stack = c3.checkbox("Force Stacking (NFL)", value=(sport=='NFL'))
        
        names = sorted(active['name'].unique())
        locks = st.multiselect("🔒 Lock", names)
        bans = st.multiselect("🚫 Ban", names)
        
        if st.button("⚡ Generate Lineups"):
            cfg = {'sport':sport, 'site':site, 'mode':mode, 'count':count, 'locks':locks, 'bans':bans, 'stack':stack, 'sim_mode':True}
            res = optimize_lineup(st.session_state['dfs_pool'], cfg)
            
            if res is not None:
                st.dataframe(res)
                # HTML Export
                href = f'<a href="data:text/html;base64,{get_html_report(res)}" download="titan_cheat_sheet.html">📥 Download Rich Report</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.error("Optimizer Failed. Check constraints.")

# --- TAB 3: SIMULATION ---
with tabs[2]:
    st.markdown("### 🔮 Slate Simulator")
    st.caption("Run 50 simulations to find the highest 'Win Equity' players.")
    
    if st.button("🎲 Run Simulation"):
        with st.spinner("Simulating 50 Slates..."):
            cfg = {'sport':sport, 'site':site, 'mode':"Classic", 'count':50, 'locks':[], 'bans':[], 'stack':True, 'sim_mode':True}
            res = optimize_lineup(st.session_state['dfs_pool'], cfg)
            
            if res is not None:
                exposure = res['name'].value_counts(normalize=True).mul(100).reset_index()
                exposure.columns = ['Player', 'Win Equity %']
                
                fig = px.bar(exposure.head(15), x='Win Equity %', y='Player', orientation='h', 
                             title="Top Simulated Plays", color='Win Equity %', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)

# --- TAB 4: PROPS ---
with tabs[3]:
    pool = st.session_state['prop_pool']
    active = pool[pool['sport'] == sport].copy()
    
    if active.empty:
        st.warning("No props found.")
    else:
        brain = TitanBrain(current_bank)
        active['pick'] = np.where(active['projection'] > active['prop_line'], "OVER", "UNDER")
        
        res = active.apply(lambda x: brain.calculate_prop_edge(x), axis=1, result_type='expand')
        active['units'] = res[0]
        active['rating'] = res[1]
        active['win_prob'] = res[2]
        
        best = active[active['units'] > 0].sort_values('win_prob', ascending=False)
        
        col1, col2 = st.columns([2,1])
        with col1:
            st.dataframe(best[['name', 'prop_line', 'projection', 'pick', 'win_prob', 'units']])
        with col2:
            st.markdown("#### 🎫 Power Slip")
            if len(best) >= 5:
                for i, row in best.head(5).iterrows():
                    color = "#00ff41" if row['pick'] == "OVER" else "#ff0055"
                    st.markdown(f"""
                    <div style="background:#222; padding:10px; margin-bottom:5px; border-left:5px solid {color}; border-radius:5px;">
                        <b>{row['name']}</b>
                        <div style="display:flex; justify-content:space-between;">
                            <span style="color:{color}">{row['pick']} {row['prop_line']}</span>
                            <span>{row['win_prob']:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Need 5+ strong plays for a slip.")
