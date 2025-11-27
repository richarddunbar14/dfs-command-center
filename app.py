import streamlit as st
import pandas as pd
import numpy as np
import pulp
import time
import difflib
import re
import base64
from duckduckgo_search import DDGS

# ==========================================
# ⚙️ 1. SYSTEM CONFIGURATION & STYLING
# ==========================================
st.set_page_config(layout="wide", page_title="TITAN OMNI: DEFINITIVE", page_icon="⚡")

st.markdown("""
<style>
    /* TITAN DARK THEME */
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    
    /* COMPONENT STYLING */
    .titan-card { 
        background: #111; border: 1px solid #333; padding: 15px; border-radius: 8px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 10px; border-left: 4px solid #00d2ff;
    }
    .titan-card-prop { border-left: 4px solid #00ff41; } /* Green for props */
    
    div[data-testid="stMetricValue"] { color: #00d2ff; text-shadow: 0 0 10px rgba(0,210,255,0.4); }
    
    .stButton>button { 
        width: 100%; border-radius: 4px; font-weight: 900; letter-spacing: 1px; text-transform: uppercase;
        background: #222; color: #00d2ff; border: 1px solid #00d2ff; transition: 0.3s;
    }
    .stButton>button:hover { background: #00d2ff; color: #000; }
    
    .stDataFrame { border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if 'dfs_pool' not in st.session_state: st.session_state['dfs_pool'] = pd.DataFrame()
if 'prop_pool' not in st.session_state: st.session_state['prop_pool'] = pd.DataFrame()
if 'ai_intel' not in st.session_state: st.session_state['ai_intel'] = {}

# ==========================================
# 🧠 2. THE TITAN BRAIN (LOGIC CORE)
# ==========================================

class TitanBrain:
    def __init__(self, sport, bankroll=1000):
        self.sport = sport
        self.bankroll = bankroll

    def calculate_kelly_bet(self, row):
        """Calculates bet size using Kelly Criterion."""
        if row.get('prop_line', 0) == 0: return 0, "No Line", 0.0
        
        proj = row['projection']
        line = row['prop_line']
        edge_raw = (proj - line) / line
        
        # Win Prob Model: Base 50% + (Edge * 0.5)
        win_prob = 0.50 + (edge_raw * 0.5) 
        win_prob = max(0.40, min(0.75, win_prob)) 
        
        # Kelly Formula (assuming -110 odds)
        odds = 0.909
        kelly_fraction = ((odds * win_prob) - (1 - win_prob)) / odds
        
        # Half-Kelly for safety
        safe_fraction = max(0, kelly_fraction * 0.5)
        
        units = safe_fraction * 100
        wager_amt = safe_fraction * self.bankroll
        
        rating = "PASS"
        if units > 2.5: rating = "💎 MAX PLAY"
        elif units > 1.0: rating = "🟢 STRONG"
        elif units > 0.1: rating = "🟡 LEAN"
        
        return units, rating, wager_amt

    def generate_narrative(self, row, context="PROP"):
        """Generates plain-English analysis."""
        txt = ""
        if context == "PROP":
            if row['pick'] == "OVER":
                txt = f"ADVANTAGE: Projection ({row['projection']}) exceeds line ({row['prop_line']}) by {row['edge_pct']:.1f}%."
            else:
                txt = f"ADVANTAGE: Projection ({row['projection']}) is well under line ({row['prop_line']}) by {abs(row['edge_pct']):.1f}%."
            
            if row.get('ai_hype', 0) > 0:
                txt += " [AI BOOST] Web search detects positive sleeper/breakout sentiment."
        
        elif context == "DFS":
            txt = f"VALUE: {row['value_score']:.1f}x Pts/$."
            if row.get('ai_hype', 0) > 0:
                txt += " AI SCOUT: Identified as active news target."
                
        return txt

# ==========================================
# 📂 3. DATA REFINERY (INGEST & CLEAN)
# ==========================================

class DataRefinery:
    @staticmethod
    def clean_currency(val):
        try: return float(re.sub(r'[^\d.]', '', str(val)))
        except: return 0

    @staticmethod
    def detect_and_clean(df):
        """Auto-classifies files as DFS Slate, Projections, or Props."""
        cols = [c.lower() for c in df.columns]
        df.columns = df.columns.str.lower().str.strip()
        standardized = pd.DataFrame()
        
        # 1. NAME
        for c in ['player', 'name', 'athlete', 'player name', 'nickname']:
            if c in df.columns:
                # Clean DK Names "Name (ID)"
                standardized['name'] = df[c].astype(str).apply(lambda x: x.split('(')[0].strip())
                break
        
        # 2. PROJECTIONS
        for c in ['fpts', 'proj', 'projection', 'avg', 'points', 'fppg']:
            if c in df.columns:
                standardized['projection'] = df[c].apply(DataRefinery.clean_currency)
                break
        if 'projection' not in standardized.columns: standardized['projection'] = 0.0

        # 3. SALARY (DFS Trigger)
        is_dfs = False
        for c in ['salary', 'cost', 'price']:
            if c in df.columns:
                standardized['salary'] = df[c].apply(DataRefinery.clean_currency)
                is_dfs = True
                break
        
        # 4. PROPS (Prop Trigger)
        is_prop = False
        for c in ['prop', 'line', 'strike', 'total', 'ou']:
            if c in df.columns:
                standardized['prop_line'] = df[c].apply(DataRefinery.clean_currency)
                is_prop = True
                break
                
        # 5. METADATA
        if 'team' in df.columns: standardized['team'] = df['team']
        elif 'teamabbrev' in df.columns: standardized['team'] = df['teamabbrev']
        
        if 'position' in df.columns: standardized['position'] = df['position']
        elif 'roster position' in df.columns: standardized['position'] = df['roster position']

        file_type = "DFS" if is_dfs else ("PROPS" if is_prop else "PROJECTIONS")
        return standardized, file_type

    @staticmethod
    def smart_merge(base_df, new_df):
        """Fuzzy merges data."""
        if base_df.empty: return new_df
        base_names = base_df['name'].unique()
        name_map = {}
        for name in new_df['name'].unique():
            matches = difflib.get_close_matches(name, base_names, n=1, cutoff=0.85)
            name_map[name] = matches[0] if matches else name
                
        new_df['merge_key'] = new_df['name'].map(name_map)
        cols_to_add = [c for c in new_df.columns if c not in base_df.columns and c != 'merge_key' and c != 'name']
        merged = base_df.merge(new_df[['merge_key'] + cols_to_add], left_on='name', right_on='merge_key', how='left')
        return merged

# ==========================================
# 📡 4. AI WEB SCOUT
# ==========================================

def run_ai_scout(sport):
    intel = {}
    queries = [
        f"{sport} dfs sleepers value plays analysis",
        f"{sport} player prop bets best over under today",
        f"{sport} injury news impact report"
    ]
    try:
        with DDGS() as ddgs:
            for q in queries:
                results = list(ddgs.text(q, max_results=5))
                for r in results:
                    text_blob = (r['title'] + " " + r['body']).lower()
                    intel[text_blob] = 1 
        return intel
    except: return {}

# ==========================================
# 🏭 5. OPTIMIZER ENGINE (DUAL MODE)
# ==========================================

def optimize_dfs(df, config):
    site, cap, mode = config['site'], config['cap'], config['mode']
    pool = df[df['projection'] > 0].reset_index(drop=True)
    if pool.empty: return None
    
    lineups = []
    num_lineups = config['num_lineups']
    
    # Progress Bar
    bar = st.progress(0)
    
    for i in range(num_lineups):
        prob = pulp.LpProblem("Titan_DFS", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("player", pool.index, cat='Binary')
        
        # MODE SELECTION
        if mode == "Slate Breaker (GPP)":
            # Monte Carlo Variance: Randomize projection by +/- 15%
            volatility = np.random.normal(1.0, 0.15, size=len(pool))
            pool['sim_pts'] = pool['projection'] * volatility
            # Bonus for AI Hype
            pool['sim_pts'] += (pool.get('ai_hype', 0) * 3.0)
            target_col = 'sim_pts'
        else:
            # Optimal Cash Mode
            target_col = 'projection'
        
        prob += pulp.lpSum([pool.loc[i, target_col] * x[i] for i in pool.index])
        prob += pulp.lpSum([pool.loc[i, 'salary'] * x[i] for i in pool.index]) <= cap
        
        # Roster Constraints
        roster_size = 9 # Default
        prob += pulp.lpSum([x[i] for i in pool.index]) == roster_size
        
        # NFL Stack Logic (Simple)
        if config['sport'] == 'NFL' and config['stacking']:
            qbs = pool[pool['position'].str.contains("QB", na=False)]
            if len(qbs) > 0: prob += pulp.lpSum([x[i] for i in qbs.index]) == 1
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == 1:
            sel = [i for i in pool.index if x[i].varValue == 1]
            lu = pool.loc[sel].copy()
            lu['Lineup_ID'] = i + 1
            lu['Total_Proj'] = lu['projection'].sum()
            lineups.append(lu)
            
            # Constraint: Don't pick exact same lineup again
            prob += pulp.lpSum([x[i] for i in sel]) <= roster_size - 1
            
        bar.progress((i+1)/num_lineups)
        
    return pd.concat(lineups) if lineups else None

def get_csv_download(df, filename="titan_export.csv"):
    csv = df.to_csv(index=False).encode('utf-8')
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="stButton" style="text-decoration:none; color:#00d2ff; border:1px solid #00d2ff; padding:5px; border-radius:5px;">📥 DOWNLOAD CSV</a>'

# ==========================================
# 🖥️ 6. DASHBOARD INTERFACE
# ==========================================

st.sidebar.title("TITAN OMNI")
st.sidebar.caption("Definitive Edition")
sport = st.sidebar.selectbox("Sport", ["NFL", "NBA", "MLB", "NHL"])
bankroll = st.sidebar.number_input("Bankroll ($)", 1000)

tabs = st.tabs(["1. 📡 Intel & Data", "2. 🏰 DFS Factory", "3. 🎫 Prop Sniper", "4. 📊 Analysis"])

# --- TAB 1: DATA INGEST ---
with tabs[0]:
    st.header("Data Ingestion Hub")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("### 1. Web Scout")
        if st.button("🛰️ Launch AI Satellites"):
            with st.spinner("Scraping Global Feeds..."):
                intel = run_ai_scout(sport)
                st.session_state['ai_intel'] = intel
            st.success(f"Intel Gathered: {len(intel)} Sources")
            
    with c2:
        st.markdown("### 2. File Fusion")
        files = st.file_uploader("Upload CSVs (DK/FD/Props/Projections)", accept_multiple_files=True)
        if st.button("🧬 Fuse Data"):
            refinery = DataRefinery()
            
            dfs_temp = pd.DataFrame()
            prop_temp = pd.DataFrame()
            proj_temp = pd.DataFrame()
            
            for f in files:
                try:
                    if f.name.endswith('.csv'): raw = pd.read_csv(f)
                    else: raw = pd.read_excel(f)
                    clean_df, ftype = refinery.detect_and_clean(raw)
                    if ftype == "DFS": dfs_temp = clean_df
                    elif ftype == "PROPS": prop_temp = clean_df
                    elif ftype == "PROJECTIONS": proj_temp = clean_df
                except: pass
                
            # Merge Logic
            if not proj_temp.empty:
                if not dfs_temp.empty: dfs_temp = refinery.smart_merge(dfs_temp, proj_temp)
                if not prop_temp.empty: prop_temp = refinery.smart_merge(prop_temp, proj_temp)
            
            # Apply AI Hype
            for text in st.session_state['ai_intel']:
                if not dfs_temp.empty: dfs_temp['ai_hype'] = dfs_temp['name'].apply(lambda x: 1 if str(x).lower() in text else 0)
                if not prop_temp.empty: prop_temp['ai_hype'] = prop_temp['name'].apply(lambda x: 1 if str(x).lower() in text else 0)

            st.session_state['dfs_pool'] = dfs_temp
            st.session_state['prop_pool'] = prop_temp
            st.success(f"Data Fused. DFS: {len(dfs_temp)} | Props: {len(prop_temp)}")

# --- TAB 2: DFS FACTORY ---
with tabs[1]:
    st.header("DFS Lineup Builder")
    df = st.session_state['dfs_pool']
    
    if df.empty:
        st.warning("No DFS Data (Salaries) Found.")
    else:
        df['value_score'] = (df['projection'] / df['salary']) * 1000
        
        c1, c2, c3 = st.columns(3)
        site = c1.selectbox("Site", ["DK", "FD"])
        cap = 50000 if site == 'DK' else 60000
        mode = c2.selectbox("Mode", ["Optimal (Cash)", "Slate Breaker (GPP)"])
        stack = c3.checkbox("Stack QB+WR (NFL)", value=True)
        count = st.slider("Lineups to Build", 1, 50, 5)
        
        if st.button("⚡ Run Optimizer"):
            config = {'site': site, 'cap': cap, 'sport': sport, 'stacking': stack, 'mode': mode, 'num_lineups': count}
            res = optimize_dfs(df, config)
            
            if res is not None:
                st.success(f"Generated {count} Lineups")
                
                # Show CSV Download
                st.markdown(get_csv_download(res, "titan_lineups.csv"), unsafe_allow_html=True)
                
                # Display Top Lineup
                top_lu = res[res['Lineup_ID'] == 1]
                st.dataframe(top_lu[['name', 'position', 'salary', 'projection', 'value_score']])
                
                # Narrative
                qb = top_lu[top_lu['position'].str.contains("QB")].iloc[0]['name'] if 'QB' in top_lu['position'].values else "None"
                st.info(f"STRATEGY: Anchored by {qb}. Mode: {mode}.")
            else:
                st.error("Optimization Failed. Check constraints.")

# --- TAB 3: PROP SNIPER ---
with tabs[2]:
    st.header("Prop Betting Desk")
    df = st.session_state['prop_pool']
    
    if df.empty:
        st.warning("No Prop Data Found.")
    else:
        brain = TitanBrain(sport, bankroll)
        
        df['edge_pct'] = ((df['projection'] - df['prop_line']) / df['prop_line']) * 100
        df['pick'] = np.where(df['projection'] > df['prop_line'], 'OVER', 'UNDER')
        
        # Kelly Calc
        results = df.apply(lambda x: brain.calculate_kelly_bet(x), axis=1, result_type='expand')
        df['units'] = results[0]
        df['rating'] = results[1]
        df['wager_amt'] = results[2]
        df['analysis'] = df.apply(lambda x: brain.generate_narrative(x), axis=1)
        
        # Export
        st.markdown(get_csv_download(df, "titan_bets.csv"), unsafe_allow_html=True)
        
        # Display Cards
        plays = df[df['units'] > 0].sort_values('units', ascending=False)
        for idx, row in plays.iterrows():
            color = "#00ff41" if "MAX" in row['rating'] else "#00d2ff"
            st.markdown(f"""
            <div class="titan-card titan-card-prop" style="border-left: 5px solid {color};">
                <div style="display:flex; justify-content:space-between;">
                    <h3 style="margin:0;">{row['name']}</h3>
                    <h3 style="margin:0; color:{color};">{row['pick']} {row['prop_line']}</h3>
                </div>
                <div style="font-size:12px; color:#888;">{row['analysis']}</div>
                <div style="margin-top:10px; font-weight:bold; font-size:18px;">
                    BET: {row['units']:.2f} UNITS <span style="font-size:12px; color:#666;">(${row['wager_amt']:.2f})</span>
                    <span style="float:right; background:{color}; color:#000; padding:2px 8px; border-radius:4px;">{row['rating']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 4: ANALYSIS ---
with tabs[3]:
    st.header("Raw Intelligence")
    if not st.session_state['dfs_pool'].empty:
        st.write("DFS Pool:", st.session_state['dfs_pool'])
    if not st.session_state['prop_pool'].empty:
        st.write("Prop Pool:", st.session_state['prop_pool'])
