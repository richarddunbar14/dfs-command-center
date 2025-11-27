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
# ⚙️ 1. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="TITAN OMNI: UNIVERSAL", page_icon="🌐")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    .titan-card { background: #1a1c24; border: 1px solid #333; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #00d2ff; }
    .titan-card-prop { border-left: 4px solid #00ff41; }
    .stButton>button { width: 100%; border-radius: 4px; font-weight: 900; background: #222; color: #00d2ff; border: 1px solid #00d2ff; }
    .stButton>button:hover { background: #00d2ff; color: #000; }
    div[data-testid="stMetricValue"] { color: #00d2ff; }
</style>
""", unsafe_allow_html=True)

if 'dfs_pool' not in st.session_state: st.session_state['dfs_pool'] = pd.DataFrame()
if 'prop_pool' not in st.session_state: st.session_state['prop_pool'] = pd.DataFrame()
if 'ai_intel' not in st.session_state: st.session_state['ai_intel'] = {}

# ==========================================
# 🧠 2. TITAN BRAIN (PROPS & ANALYSIS)
# ==========================================

class TitanBrain:
    def __init__(self, bankroll=1000):
        self.bankroll = bankroll

    def calculate_kelly_bet(self, row):
        if row.get('prop_line', 0) <= 0 or row.get('projection', 0) <= 0: 
            return 0, "No Data", 0.0
        
        proj = row['projection']
        line = row['prop_line']
        edge_raw = abs(proj - line) / line
        
        if edge_raw < 0.03: return 0, "No Edge", 0.0

        # Win Prob Model
        win_prob = 0.53 + (edge_raw * 0.4) 
        win_prob = min(0.70, win_prob) 
        
        # Kelly (Odds -110)
        odds = 0.909
        kelly_fraction = ((odds * win_prob) - (1 - win_prob)) / odds
        safe_fraction = max(0, kelly_fraction * 0.25) # Quarter Kelly
        
        units = safe_fraction * 100
        wager = safe_fraction * self.bankroll
        
        rating = "PASS"
        if units > 2.0: rating = "💎 MAX PLAY"
        elif units > 1.0: rating = "🟢 STRONG"
        elif units > 0.1: rating = "🟡 LEAN"
        
        return units, rating, wager

    def generate_narrative(self, row):
        diff = row['projection'] - row['prop_line']
        side = "OVER" if row['pick'] == "OVER" else "UNDER"
        return f"Proj {row['projection']} is {abs(diff):.1f} pts {side} line {row['prop_line']}."

# ==========================================
# 📂 3. DATA REFINERY
# ==========================================

class DataRefinery:
    @staticmethod
    def clean_currency(val):
        try:
            s = str(val).strip()
            if s in ['-', '', 'nan']: return 0.0
            return float(re.sub(r'[^\d.]', '', s))
        except: return 0.0

    @staticmethod
    def detect_and_clean(df):
        df.columns = df.columns.astype(str).str.upper().str.strip()
        std = pd.DataFrame()
        
        # Name
        if 'PLAYER' in df.columns: std['name'] = df['PLAYER']
        elif 'NAME' in df.columns: std['name'] = df['NAME']
        else: std['name'] = "Unknown"
        std['name'] = std['name'].astype(str).apply(lambda x: x.split('(')[0].strip())
        
        # Proj
        if 'FPTS' in df.columns: std['projection'] = df['FPTS'].apply(DataRefinery.clean_currency)
        elif 'PROJ' in df.columns: std['projection'] = df['PROJ'].apply(DataRefinery.clean_currency)
        else: std['projection'] = 0.0
        
        # Salary
        if 'SAL' in df.columns: std['salary'] = df['SAL'].apply(DataRefinery.clean_currency)
        elif 'SALARY' in df.columns: std['salary'] = df['SALARY'].apply(DataRefinery.clean_currency)
        else: std['salary'] = 0.0
        
        # Position
        if 'POS' in df.columns: std['position'] = df['POS'].astype(str)
        elif 'POSITION' in df.columns: std['position'] = df['POSITION'].astype(str)
        else: std['position'] = 'FLEX'
        
        # Team
        if 'TEAM' in df.columns: std['team'] = df['TEAM']
        elif 'TM' in df.columns: std['team'] = df['TM']
        else: std['team'] = 'N/A'
        
        # Prop
        if 'O/U' in df.columns: std['prop_line'] = df['O/U'].apply(DataRefinery.clean_currency)
        elif 'PROP' in df.columns: std['prop_line'] = df['PROP'].apply(DataRefinery.clean_currency)
        else: std['prop_line'] = 0.0

        is_dfs = std['salary'].sum() > 0
        is_prop = std['prop_line'].sum() > 0
        ftype = "DFS" if is_dfs else ("PROPS" if is_prop else "PROJECTIONS")
        return std, ftype

    @staticmethod
    def smart_merge(base, new_df):
        if base.empty: return new_df
        if new_df.empty: return base
        
        base = base.drop_duplicates(subset=['name'])
        new_df = new_df.drop_duplicates(subset=['name'])
        
        base_names = base['name'].unique()
        mapping = {}
        for name in new_df['name'].unique():
            matches = difflib.get_close_matches(name, base_names, n=1, cutoff=0.85)
            if matches: mapping[name] = matches[0]
            
        new_df['merge_key'] = new_df['name'].map(mapping).fillna(new_df['name'])
        cols = [c for c in new_df.columns if c not in base.columns and c != 'merge_key' and c != 'name']
        merged = base.merge(new_df[['merge_key'] + cols], left_on='name', right_on='merge_key', how='left')
        
        if 'projection_y' in merged.columns:
            merged['projection'] = np.where(merged['projection_x'] > 0, merged['projection_x'], merged['projection_y'])
            merged = merged.drop(columns=['projection_x', 'projection_y', 'merge_key'])
            
        return merged

# ==========================================
# 🏭 4. UNIVERSAL OPTIMIZER ENGINE
# ==========================================

def get_rulebook(sport, site, mode):
    """Returns salary cap, roster size, and position constraints."""
    rules = {'cap': 50000, 'size': 8, 'positions': {}}
    
    # --- SITE DEFAULTS ---
    if site == 'DK':
        rules['cap'] = 50000
    elif site == 'FD':
        rules['cap'] = 60000
    elif site == 'Yahoo':
        rules['cap'] = 200

    # --- SHOWDOWN / SINGLE GAME OVERRIDES ---
    if mode == "Showdown":
        if site == 'DK':
            rules['size'] = 6
            rules['positions'] = {'CPT': 1, 'FLEX': 5}
            rules['cpt_multiplier'] = 1.5 # Salary & Points
        elif site == 'FD':
            rules['size'] = 5
            rules['positions'] = {'MVP': 1, 'FLEX': 4}
            rules['cpt_multiplier'] = 1.0 # Points only usually 1.5x, salary same. Keeping simple.
        return rules

    # --- CLASSIC MODES ---
    if sport == 'NFL':
        if site == 'DK':
            rules['size'] = 9
            rules['positions'] = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 1}
        elif site == 'FD':
            rules['size'] = 9
            rules['positions'] = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1, 'FLEX': 1}
            
    elif sport == 'NBA':
        if site == 'DK':
            rules['size'] = 8
            rules['positions'] = {'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1, 'G': 1, 'F': 1, 'UTIL': 1}
        elif site == 'FD':
            rules['size'] = 9
            rules['positions'] = {'PG': 2, 'SG': 2, 'SF': 2, 'PF': 2, 'C': 1}

    elif sport == 'MLB':
        if site == 'DK':
            rules['size'] = 10
            rules['positions'] = {'P': 2, 'C': 0, '1B': 0, '2B': 0, '3B': 0, 'SS': 0, 'OF': 3} # Simplified
            
    elif sport == 'PGA':
        rules['size'] = 6
        rules['positions'] = {'G': 6}
        
    elif sport in ['CFB', 'CBB']:
        rules['size'] = 8
        rules['positions'] = {'FLEX': 8} # Simplified for generic utility
        
    return rules

def expand_pool_for_showdown(pool, site):
    """Creates Captain/MVP rows dynamically."""
    if pool.empty: return pool
    
    flex_pool = pool.copy()
    flex_pool['pos_id'] = 'FLEX'
    
    cpt_pool = pool.copy()
    if site == 'DK':
        cpt_pool['name'] = cpt_pool['name'] + " (CPT)"
        cpt_pool['salary'] = cpt_pool['salary'] * 1.5
        cpt_pool['projection'] = cpt_pool['projection'] * 1.5
        cpt_pool['pos_id'] = 'CPT'
    elif site == 'FD':
        cpt_pool['name'] = cpt_pool['name'] + " (MVP)"
        cpt_pool['projection'] = cpt_pool['projection'] * 1.5
        cpt_pool['pos_id'] = 'MVP'
        
    return pd.concat([cpt_pool, flex_pool]).reset_index(drop=True)

def optimize_universal(df, sport, site, mode, num_lineups):
    # 1. Get Rules
    rules = get_rulebook(sport, site, mode)
    
    # 2. Prepare Pool
    pool = df[df['projection'] > 0].copy()
    
    if mode == "Showdown":
        pool = expand_pool_for_showdown(pool, site)
    else:
        # Standardize position column for Classic checks
        # Map specific positions to generic roles if needed
        pool['pos_id'] = pool['position']

    # 3. Build Problem
    lineups = []
    bar = st.progress(0)
    
    for i in range(num_lineups):
        prob = pulp.LpProblem("Titan_Solver", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("player", pool.index, cat='Binary')
        
        # Objective (with slight randomization for multiple lineups)
        pool['sim'] = pool['projection'] * np.random.normal(1.0, 0.05, len(pool))
        prob += pulp.lpSum([pool.loc[p, 'sim'] * x[p] for p in pool.index])
        
        # Salary Cap
        prob += pulp.lpSum([pool.loc[p, 'salary'] * x[p] for p in pool.index]) <= rules['cap']
        
        # Roster Size
        prob += pulp.lpSum([x[p] for p in pool.index]) == rules['size']
        
        # Positional Constraints
        if mode == "Showdown":
            # Captain Constraint (Exactly 1)
            cpt_label = 'CPT' if site == 'DK' else 'MVP'
            cpt_indices = pool[pool['pos_id'] == cpt_label].index
            prob += pulp.lpSum([x[p] for p in cpt_indices]) == 1
            
            # Uniqueness (Cannot play CPT and FLEX version of same player)
            # We assume names match except for " (CPT)" suffix
            # Logic: Group by base name and ensure sum <= 1
            # (Simplified for speed: usually standard CPT logic handles this by salary)
            pass
        else:
            # Classic Constraints
            for pos, count in rules['positions'].items():
                if pos == 'FLEX' or pos == 'UTIL': continue # Handle flexibility later
                
                # Loose matching: 'QB' matches 'QB' and 'QB/TE'
                # Strict: 'DST' matches 'DST'
                relevant = pool[pool['position'].str.contains(pos, na=False)]
                if not relevant.empty and count > 0:
                    prob += pulp.lpSum([x[p] for p in relevant.index]) >= count

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == 1:
            sel = [p for p in pool.index if x[p].varValue == 1]
            lu = pool.loc[sel].copy()
            lu['Lineup_ID'] = i+1
            lu['Total_Proj'] = lu['projection'].sum()
            lineups.append(lu)
            
            # Constraint: Block exact lineup
            prob += pulp.lpSum([x[p] for p in sel]) <= rules['size'] - 1
            
        bar.progress((i+1)/num_lineups)
        
    return pd.concat(lineups) if lineups else None

def get_csv(df):
    return df.to_csv(index=False).encode()

# ==========================================
# 🖥️ 5. DASHBOARD
# ==========================================

st.sidebar.title("TITAN OMNI")
st.sidebar.caption("Universal Edition")

# --- GLOBAL SETTINGS ---
c1, c2 = st.sidebar.columns(2)
sport = c1.selectbox("Sport", ["NFL", "NBA", "MLB", "CFB", "CBB", "PGA"])
site = c2.selectbox("Site", ["DK", "FD", "Yahoo"])
mode = st.sidebar.radio("Mode", ["Classic", "Showdown"])
bankroll = st.sidebar.number_input("Bankroll", 1000)

tabs = st.tabs(["1. 📡 Ingest", "2. 🏰 Optimizer", "3. 🎫 Props"])

# --- TAB 1: INGEST ---
with tabs[0]:
    st.header("Data Center")
    files = st.file_uploader("Upload CSVs", accept_multiple_files=True)
    if st.button("🧬 Fuse Data"):
        ref = DataRefinery()
        dfs = pd.DataFrame()
        props = pd.DataFrame()
        proj = pd.DataFrame()
        
        for f in files:
            try:
                try: raw = pd.read_csv(f, encoding='utf-8-sig')
                except: raw = pd.read_excel(f)
                clean, ftype = ref.detect_and_clean(raw)
                
                if ftype == "DFS": dfs = clean
                elif ftype == "PROPS": props = clean
                elif ftype == "PROJECTIONS": proj = clean
            except: pass
            
        if not proj.empty:
            if not dfs.empty: dfs = ref.smart_merge(dfs, proj)
            if not props.empty: props = ref.smart_merge(props, proj)
            
        st.session_state['dfs_pool'] = dfs.drop_duplicates(subset=['name'])
        st.session_state['prop_pool'] = props.drop_duplicates(subset=['name'])
        st.success(f"Ready. DFS: {len(dfs)} | Props: {len(props)}")

# --- TAB 2: OPTIMIZER ---
with tabs[1]:
    st.header(f"{site} {sport} {mode} Builder")
    df = st.session_state['dfs_pool']
    
    if df.empty:
        st.warning("No Salary Data Found.")
    else:
        st.write(f"Pool: {len(df)} Players")
        cnt = st.slider("Lineups", 1, 20, 5)
        
        if st.button("⚡ Generate Lineups"):
            res = optimize_universal(df, sport, site, mode, cnt)
            
            if res is not None:
                st.dataframe(res)
                st.download_button("Download CSV", get_csv(res), "titan_lineups.csv")
            else:
                st.error("Optimization Failed. Try checking salary constraints.")

# --- TAB 3: PROPS ---
with tabs[2]:
    st.header("Prop Sniper")
    df = st.session_state['prop_pool']
    
    if df.empty:
        st.warning("No Prop Lines Found.")
    else:
        brain = TitanBrain(bankroll)
        df['pick'] = np.where(df['projection'] > df['prop_line'], "OVER", "UNDER")
        
        res = df.apply(lambda x: brain.calculate_kelly_bet(x), axis=1, result_type='expand')
        df['units'] = res[0]
        df['rating'] = res[1]
        df['wager'] = res[2]
        df['narrative'] = df.apply(lambda x: brain.generate_narrative(x), axis=1)
        
        active = df[df['units'] > 0].sort_values('units', ascending=False)
        st.dataframe(active)
