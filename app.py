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
st.set_page_config(layout="wide", page_title="TITAN OMNI: OMEGA", page_icon="🦁")

st.markdown("""
<style>
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

# ==========================================
# 💾 2. DATABASE & HISTORY
# ==========================================

def init_db():
    conn = sqlite3.connect('titan_omega.db')
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
# 🧠 3. TITAN BRAIN (MATH ENGINE)
# ==========================================

class TitanBrain:
    def __init__(self, bankroll):
        self.bankroll = bankroll

    def calculate_prop_edge(self, row):
        if row.get('prop_line', 0) <= 0 or row.get('projection', 0) <= 0: return 0, "No Data", 0.0
        
        proj, line = row['projection'], row['prop_line']
        edge_raw = abs(proj - line) / line
        
        # Logistic Win Prob Model
        win_prob = 0.52 + (edge_raw * 0.5)
        win_prob = min(0.75, win_prob)
        
        # Kelly Criterion
        odds = 0.909 # -110
        kelly = ((odds * win_prob) - (1 - win_prob)) / odds
        units = max(0, kelly * 0.3) * 100 
        
        rating = "PASS"
        if units > 3.0: rating = "💎 ATOMIC"
        elif units > 1.5: rating = "🟢 NUCLEAR"
        elif units > 0.5: rating = "🟡 KINETIC"
        
        return units, rating, win_prob

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
            'projection': ['FPTS', 'PROJ', 'AVG FPTS', 'FC PROJ', 'PTS'],
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
# 📡 5. AI SCOUT
# ==========================================

def run_web_scout(sport):
    intel = {}
    try:
        with DDGS() as ddgs:
            q = f"{sport} dfs sleepers value prop bets analysis today"
            for r in list(ddgs.text(q, max_results=5)):
                blob = (r['title'] + " " + r['body']).lower()
                intel[blob] = 1
    except: pass
    return intel

# ==========================================
# 🏭 6. OPTIMIZER & SIMULATION
# ==========================================

def optimize_lineup(df, config):
    pool = df[df['sport'] == config['sport']].copy()
    pool = pool[pool['projection'] > 0].reset_index(drop=True)
    
    if config['bans']: pool = pool[~pool['name'].isin(config['bans'])].reset_index(drop=True)
    if pool.empty: return None

    cap = 50000 if config['site'] == 'DK' else 60000
    if config['site'] == 'Yahoo': cap = 200
    
    size = 8
    if config['sport'] == 'NFL': size = 9
    elif config['sport'] == 'MLB': size = 10
    
    if config['mode'] == 'Showdown':
        size = 6 if config['site'] == 'DK' else 5
        flex = pool.copy(); flex['pos_id'] = 'FLEX'
        cpt = pool.copy(); cpt['pos_id'] = 'CPT'
        cpt['name'] += " (CPT)"
        cpt['projection'] *= 1.5
        if config['site'] == 'DK': cpt['salary'] *= 1.5
        pool = pd.concat([cpt, flex]).reset_index(drop=True)

    lineups = []
    
    for i in range(config['count']):
        prob = pulp.LpProblem("Titan_Omega", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("p", pool.index, cat='Binary')
        
        # Sim Variance (Monte Carlo)
        volatility = 0.15 if config['sim'] else 0.0
        pool['sim'] = pool['projection'] * np.random.normal(1.0, volatility, len(pool))
        prob += pulp.lpSum([pool.loc[p, 'sim'] * x[p] for p in pool.index])
        
        prob += pulp.lpSum([pool.loc[p, 'salary'] * x[p] for p in pool.index]) <= cap
        prob += pulp.lpSum([x[p] for p in pool.index]) == size
        
        # Locks
        for lock in config.get('locks', []):
            l_idx = pool[pool['name'].str.contains(lock)].index
            if not l_idx.empty: prob += pulp.lpSum([x[p] for p in l_idx]) >= 1

        if config['mode'] == 'Classic':
            if config['sport'] == 'NFL':
                qbs = pool[pool['position'].str.contains("QB")]
                dst = pool[pool['position'].str.contains("DST|DEF|^D$")]
                if not qbs.empty: prob += pulp.lpSum([x[p] for p in qbs.index]) == 1
                if not dst.empty: prob += pulp.lpSum([x[p] for p in dst.index]) == 1
                
                if config.get('stack', False):
                    for qb in qbs.index:
                        tm = pool.loc[qb, 'team']
                        mates = pool[(pool['team'] == tm) & (pool['position'].str.contains("WR|TE"))]
                        if not mates.empty: prob += pulp.lpSum([x[m] for m in mates.index]) >= x[qb]

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
        body {{ font-family: monospace; background: #f4f4f4; padding: 20px; }}
        .play {{ background: #fff; padding: 10px; margin: 5px; border-left: 5px solid #0072ff; }}
        .header {{ font-size: 24px; font-weight: bold; color: #333; }}
    </style></head><body>
    <div class="header">TITAN OMEGA CHEAT SHEET</div>
    <hr>
    """
    for _, row in df.head(15).iterrows():
        html += f"<div class='play'><b>{row['name']}</b> ({row['position']}) | Sal: ${row['salary']} | Proj: {row['projection']}</div>"
    html += "</body></html>"
    return base64.b64encode(html.encode()).decode()

# ==========================================
# 🖥️ 7. DASHBOARD
# ==========================================

conn = init_db()
st.sidebar.title("TITAN OMNI")
st.sidebar.caption("Omega Edition")

# --- BANKROLL GRAPH ---
history = get_bankroll_history(conn)
current_bank = history.iloc[-1]['amount']
st.sidebar.metric("🏦 Bankroll", f"${current_bank:,.2f}")

if len(history) > 1:
    fig = px.line(history, x='date', y='amount', height=150)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.sidebar.plotly_chart(fig, use_container_width=True)

with st.sidebar.expander("Update Funds"):
    new_bal = st.number_input("New Balance", value=current_bank)
    note = st.text_input("Note", "Win/Loss")
    if st.button("Update"):
        update_bankroll(conn, new_bal, note)
        st.rerun()

sport = st.sidebar.selectbox("Sport", ["NFL", "NBA", "MLB", "CFB", "NHL", "PGA"])
site = st.sidebar.selectbox("Site", ["DK", "FD", "Yahoo", "PrizePicks"])

tabs = st.tabs(["1. 📡 Ingest", "2. 🏰 Optimizer", "3. 🔮 Simulation", "4. 🚀 Props", "5. 🧮 Parlay"])

# --- TAB 1: INGEST ---
with tabs[0]:
    st.markdown("### 📡 Data Fusion")
    files = st.file_uploader("Upload CSVs", accept_multiple_files=True)
    
    if st.button("🧬 Fuse Data"):
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
        with st.spinner("Analyzing..."):
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
        stack = c3.checkbox("Stacking (NFL)", value=(sport=='NFL'))
        
        names = sorted(active['name'].unique())
        locks = st.multiselect("🔒 Lock", names)
        bans = st.multiselect("🚫 Ban", names)
        
        if st.button("⚡ Generate Lineups"):
            cfg = {'sport':sport, 'site':site, 'mode':mode, 'count':count, 'locks':locks, 'bans':bans, 'stack':stack, 'sim':False}
            res = optimize_lineup(st.session_state['dfs_pool'], cfg)
            
            if res is not None:
                st.dataframe(res)
                
                # HTML Cheat Sheet
                st.markdown(f'<a href="data:text/html;base64,{get_html_report(res)}" download="cheat_sheet.html">📥 Download Cheat Sheet (HTML)</a>', unsafe_allow_html=True)
                
                # Heatmap
                st.markdown("### 🔥 Exposure Heatmap")
                exp = res['team'].value_counts().reset_index()
                exp.columns = ['Team', 'Count']
                fig = px.treemap(exp, path=['Team'], values='Count', color='Count', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Optimization Failed.")

# --- TAB 3: SIMULATION ---
with tabs[2]:
    st.markdown("### 🔮 Slate Simulator")
    st.caption("Run 50 Monte Carlo simulations to find 'Win Equity'.")
    if st.button("🎲 Simulate Slate"):
        with st.spinner("Simulating..."):
            cfg = {'sport':sport, 'site':site, 'mode':"Classic", 'count':50, 'locks':[], 'bans':[], 'stack':True, 'sim':True}
            res = optimize_lineup(st.session_state['dfs_pool'], cfg)
            if res is not None:
                exposure = res['name'].value_counts(normalize=True).mul(100).reset_index()
                exposure.columns = ['Player', 'Win Equity %']
                fig = px.bar(exposure.head(15), x='Win Equity %', y='Player', orientation='h', title="Top Simulated Plays")
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
        st.dataframe(best[['name', 'prop_line', 'projection', 'pick', 'win_prob', 'units']])

# --- TAB 5: PARLAY ---
with tabs[4]:
    st.markdown("### 🧮 Parlay Architect")
    pool = st.session_state['prop_pool']
    active = pool[pool['sport'] == sport].copy()
    
    if not active.empty:
        active['pick'] = np.where(active['projection'] > active['prop_line'], "OVER", "UNDER")
        options = active['name'] + " (" + active['pick'] + ")"
        selection = st.multiselect("Build Slip", options)
        
        if selection:
            total_prob = 1.0
            for item in selection:
                name = item.split(" (")[0]
                row = active[active['name'] == name].iloc[0]
                edge_raw = abs(row['projection'] - row['prop_line']) / row['prop_line']
                prob = min(0.75, 0.52 + (edge_raw * 0.5))
                total_prob *= prob
            
            c1, c2 = st.columns(2)
            c1.metric("Legs", len(selection))
            c2.metric("True Win Probability", f"{total_prob*100:.1f}%")
            if total_prob > 0: st.info(f"Fair Odds: {1/total_prob:.2f}")
    else:
        st.warning("Load props first.")
