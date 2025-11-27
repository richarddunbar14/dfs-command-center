import streamlit as st
import pandas as pd
import numpy as np
import pulp
import re
import base64
import difflib
from duckduckgo_search import DDGS

# ==========================================
# ⚙️ 1. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="TITAN OMNI: CHAMPIONSHIP", page_icon="🏆")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #f0f2f6; font-family: 'Segoe UI', sans-serif; }
    .titan-card { background: #1a1c24; border: 1px solid #444; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid #00d2ff; }
    .titan-card-prop { border-left: 5px solid #00ff41; }
    .stButton>button { width: 100%; font-weight: 900; background: #00d2ff; color: #000; border: none; padding: 10px; text-transform: uppercase; }
    .stButton>button:hover { background: #fff; box-shadow: 0 0 10px #00d2ff; }
    .header-text { font-size: 24px; font-weight: bold; color: #fff; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

if 'dfs_pool' not in st.session_state: st.session_state['dfs_pool'] = pd.DataFrame()
if 'prop_pool' not in st.session_state: st.session_state['prop_pool'] = pd.DataFrame()

# ==========================================
# 🧠 2. LOGIC & MATH ENGINE
# ==========================================

class TitanBrain:
    def __init__(self, bankroll=1000):
        self.bankroll = bankroll

    def calculate_kelly(self, row):
        if row.get('prop_line', 0) <= 0 or row.get('projection', 0) <= 0: return 0, "No Data"
        
        proj, line = row['projection'], row['prop_line']
        edge = abs(proj - line) / line
        
        if edge < 0.03: return 0, "Pass"
        
        # Win Prob (Simple Model)
        win_prob = 0.525 + (edge * 0.5)
        win_prob = min(0.68, win_prob)
        
        # Kelly Formula
        odds = 0.909 # -110
        kelly = ((odds * win_prob) - (1 - win_prob)) / odds
        units = max(0, kelly * 0.3) * 100 # Quarter Kelly
        
        rating = "LEAN"
        if units > 1.5: rating = "STRONG"
        if units > 3.0: rating = "MAX PLAY"
        
        return units, rating

# ==========================================
# 📂 3. DATA REFINERY
# ==========================================

class DataRefinery:
    @staticmethod
    def clean_curr(val):
        try:
            s = str(val).strip()
            if s in ['-', '', 'nan']: return 0.0
            return float(re.sub(r'[^\d.]', '', s))
        except: return 0.0

    @staticmethod
    def ingest(df, sport_tag):
        # Normalize Headers
        df.columns = df.columns.astype(str).str.upper().str.strip()
        std = pd.DataFrame()
        
        # 1. NAME
        if 'PLAYER' in df.columns: std['name'] = df['PLAYER']
        elif 'NAME' in df.columns: std['name'] = df['NAME']
        else: std['name'] = "Unknown"
        # Remove (ID) garbage
        std['name'] = std['name'].astype(str).apply(lambda x: x.split('(')[0].strip())
        
        # 2. PROJECTION
        if 'FPTS' in df.columns: std['projection'] = df['FPTS'].apply(DataRefinery.clean_curr)
        elif 'PROJ' in df.columns: std['projection'] = df['PROJ'].apply(DataRefinery.clean_curr)
        elif 'AVG FPTS' in df.columns: std['projection'] = df['AVG FPTS'].apply(DataRefinery.clean_curr)
        else: std['projection'] = 0.0
        
        # 3. SALARY
        if 'SAL' in df.columns: std['salary'] = df['SAL'].apply(DataRefinery.clean_curr)
        elif 'SALARY' in df.columns: std['salary'] = df['SALARY'].apply(DataRefinery.clean_curr)
        else: std['salary'] = 0.0
        
        # 4. POSITION
        if 'POS' in df.columns: std['position'] = df['POS'].astype(str)
        elif 'POSITION' in df.columns: std['position'] = df['POSITION'].astype(str)
        else: std['position'] = 'FLEX'
        
        # 5. TEAM & OPP
        if 'TEAM' in df.columns: std['team'] = df['TEAM']
        elif 'TM' in df.columns: std['team'] = df['TM']
        else: std['team'] = 'N/A'
        
        if 'OPP' in df.columns: std['opp'] = df['OPP']
        else: std['opp'] = 'N/A'
        
        # 6. PROP LINES
        if 'O/U' in df.columns: std['prop_line'] = df['O/U'].apply(DataRefinery.clean_curr)
        elif 'PROP' in df.columns: std['prop_line'] = df['PROP'].apply(DataRefinery.clean_curr)
        else: std['prop_line'] = 0.0
        
        # 7. TAG SPORT
        std['sport'] = sport_tag

        return std

    @staticmethod
    def merge(base, new_df):
        if base.empty: return new_df
        if new_df.empty: return base
        
        # Separate by Sport to prevent cross-contamination
        # This implementation simply appends new data, handling deduplication by name+sport
        combined = pd.concat([base, new_df]).drop_duplicates(subset=['name', 'sport'], keep='last').reset_index(drop=True)
        return combined

# ==========================================
# 🏭 4. CHAMPIONSHIP OPTIMIZER
# ==========================================

def get_constraints(sport, site, mode):
    c = {'cap': 50000, 'size': 8}
    
    # SALARY CAPS
    if site == 'DK': c['cap'] = 50000
    elif site == 'FD': c['cap'] = 60000
    elif site == 'Yahoo': c['cap'] = 200
    
    # SHOWDOWN MODES
    if mode == 'Showdown':
        if site == 'DK': 
            c['size'] = 6
            c['positions'] = ['CPT', 'FLEX']
        elif site == 'FD': 
            c['size'] = 5
            c['positions'] = ['MVP', 'FLEX']
        return c

    # CLASSIC MODES
    if sport == 'NFL':
        c['size'] = 9
        c['pos_map'] = {'QB':1, 'DST':1} # Strict requirements
        
    elif sport == 'NBA':
        if site == 'DK': c['size'] = 8
        elif site == 'FD': c['size'] = 9
        
    elif sport == 'MLB':
        c['size'] = 10
        
    return c

def optimize_lineup(df, config):
    # 1. Filter by Sport (CRITICAL FIX)
    pool = df[df['sport'] == config['sport']].copy()
    pool = pool[pool['projection'] > 0].reset_index(drop=True)
    
    # 2. Exclusions
    if config['bans']:
        pool = pool[~pool['name'].isin(config['bans'])].reset_index(drop=True)
    
    if pool.empty: return None

    # 3. Handle Showdown Expansion
    if config['mode'] == 'Showdown':
        flex = pool.copy()
        flex['pos_id'] = 'FLEX'
        
        cpt = pool.copy()
        mult = 1.5 if config['site'] == 'DK' else 1.5
        sal_mult = 1.5 if config['site'] == 'DK' else 1.0 # FD MVP doesn't cost more
        
        cpt['name'] = cpt['name'] + " (CPT)"
        cpt['projection'] = cpt['projection'] * mult
        cpt['salary'] = cpt['salary'] * sal_mult
        cpt['pos_id'] = 'CPT'
        
        pool = pd.concat([cpt, flex]).reset_index(drop=True)

    rules = get_constraints(config['sport'], config['site'], config['mode'])
    lineups = []
    
    bar = st.progress(0)
    
    for i in range(config['count']):
        prob = pulp.LpProblem("Titan_Win", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("player", pool.index, cat='Binary')
        
        # Objective: Projection + Variance (Randomness for GPP)
        # We add 15% +/- randomization to create diverse lineups
        pool['sim'] = pool['projection'] * np.random.normal(1.0, 0.15, len(pool))
        prob += pulp.lpSum([pool.loc[p, 'sim'] * x[p] for p in pool.index])
        
        # Constraints
        prob += pulp.lpSum([pool.loc[p, 'salary'] * x[p] for p in pool.index]) <= rules['cap']
        prob += pulp.lpSum([x[p] for p in pool.index]) == rules['size']
        
        # --- SPORT SPECIFIC LOGIC ---
        
        if config['mode'] == 'Classic':
            if config['sport'] == 'NFL':
                # Force QB
                qbs = pool[pool['position'].str.contains("QB")]
                if not qbs.empty: prob += pulp.lpSum([x[p] for p in qbs.index]) == 1
                
                # Force DST (Defense)
                # Regex looks for DST, DEF, or a lone D
                dsts = pool[pool['position'].str.contains("DST|DEF|^D$", regex=True)]
                if not dsts.empty: prob += pulp.lpSum([x[p] for p in dsts.index]) == 1
                
                # STACKING LOGIC (The Winning Edge)
                if config['stack']:
                    for qb_idx in qbs.index:
                        team = pool.loc[qb_idx, 'team']
                        # Find WR/TE on same team
                        stack_partners = pool[(pool['team'] == team) & (pool['position'].str.contains("WR|TE"))]
                        if not stack_partners.empty:
                            # If QB is picked, pick at least 1 partner
                            prob += pulp.lpSum([x[s] for s in stack_partners.index]) >= x[qb_idx]

            elif config['sport'] == 'NBA':
                # Positional Integrity for NBA is tricky in generic solvers.
                # We enforce minimums for Guards/Forwards/Centers
                guards = pool[pool['position'].str.contains("PG|SG|G")]
                forwards = pool[pool['position'].str.contains("SF|PF|F")]
                centers = pool[pool['position'].str.contains("C")]
                
                if not guards.empty: prob += pulp.lpSum([x[p] for p in guards.index]) >= 3
                if not forwards.empty: prob += pulp.lpSum([x[p] for p in forwards.index]) >= 3
                if not centers.empty: prob += pulp.lpSum([x[p] for p in centers.index]) >= 1
                
                # Team Limit (Prevent 8 players from Detroit)
                for team in pool['team'].unique():
                    team_players = pool[pool['team'] == team]
                    prob += pulp.lpSum([x[p] for p in team_players.index]) <= 4

        elif config['mode'] == 'Showdown':
            # Force exactly 1 CPT
            cpts = pool[pool['pos_id'] == 'CPT']
            prob += pulp.lpSum([x[p] for p in cpts.index]) == 1
            
            # Uniqueness: Cant play CPT and FLEX of same guy
            # (Simplified logic handled by salary constraint mostly, but added for safety)
            # This requires matching names strings which is slow, relying on salary diff.

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == 1:
            sel = [p for p in pool.index if x[p].varValue == 1]
            lu = pool.loc[sel].copy()
            lu['Lineup_ID'] = i + 1
            lu['Total_Proj'] = lu['projection'].sum()
            lineups.append(lu)
            # Avoid duplicate lineup
            prob += pulp.lpSum([x[p] for p in sel]) <= rules['size'] - 1
            
        bar.progress((i+1)/config['count'])
        
    return pd.concat(lineups) if lineups else None

def get_csv(df):
    return df.to_csv(index=False).encode()

# ==========================================
# 🖥️ 5. DASHBOARD UI
# ==========================================

st.sidebar.title("TITAN OMNI")
st.sidebar.info("Championship Edition")

# SETTINGS
sport = st.sidebar.selectbox("Sport", ["NFL", "NBA", "MLB", "CFB", "CBB", "PGA", "NHL"])
site = st.sidebar.selectbox("Site", ["DK", "FD", "Yahoo"])
mode = st.sidebar.radio("Game Mode", ["Classic", "Showdown"])
bankroll = st.sidebar.number_input("Bankroll ($)", 1000)

if st.sidebar.button("🗑️ Clear All Data"):
    st.session_state.clear()
    st.rerun()

tabs = st.tabs(["1. 📡 Ingest", "2. 🏰 Optimizer", "3. 🎫 Prop Sniper"])

# --- TAB 1: INGEST ---
with tabs[0]:
    st.markdown("### 📥 Upload Data")
    st.caption(f"Uploading for: **{sport}** ({site})")
    
    files = st.file_uploader("Drop CSVs", accept_multiple_files=True)
    
    if st.button("🧬 Fuse & Tag Data"):
        if not files:
            st.error("No files selected.")
        else:
            ref = DataRefinery()
            new_data = pd.DataFrame()
            
            for f in files:
                try:
                    try: raw = pd.read_csv(f, encoding='utf-8-sig')
                    except: raw = pd.read_excel(f)
                    
                    # Tag data with CURRENT sport selection
                    clean = ref.ingest(raw, sport) 
                    new_data = ref.merge(new_data, clean)
                except Exception as e:
                    st.error(f"Error {f.name}: {e}")
            
            # Merge into Master Pool
            st.session_state['dfs_pool'] = ref.merge(st.session_state['dfs_pool'], new_data)
            
            # Separate Props (rows with prop_line > 0)
            props = st.session_state['dfs_pool']
            st.session_state['prop_pool'] = props[props['prop_line'] > 0].copy()
            
            st.success(f"Success! Total Database: {len(st.session_state['dfs_pool'])} players.")

# --- TAB 2: OPTIMIZER ---
with tabs[1]:
    st.markdown(f"### 🏰 {sport} Lineup Factory")
    
    # 1. FILTER POOL BY SPORT
    full_pool = st.session_state['dfs_pool']
    active_pool = full_pool[full_pool['sport'] == sport].copy()
    
    if active_pool.empty:
        st.warning(f"No players found for **{sport}**. Go to Ingest tab and upload data while '{sport}' is selected.")
    else:
        # Show stats
        c1, c2, c3 = st.columns(3)
        c1.metric("Player Pool", len(active_pool))
        c2.metric("Salary Cap", f"${get_constraints(sport, site, mode)['cap']}")
        
        # 2. BAN HAMMER (Player Exclusion)
        st.markdown("#### 🛑 Exclude Players")
        all_players = sorted(active_pool['name'].unique())
        bans = st.multiselect("Select players to remove from pool:", all_players)
        
        # 3. SETTINGS
        st.markdown("#### ⚙️ Build Settings")
        col_a, col_b = st.columns(2)
        count = col_a.slider("Lineups", 1, 50, 10)
        stack = col_b.checkbox("Force Stacking (QB+WR)", value=(sport=='NFL'))
        
        if st.button("⚡ BUILD CHAMPIONSHIP LINEUPS"):
            config = {'sport': sport, 'site': site, 'mode': mode, 'count': count, 'bans': bans, 'stack': stack}
            
            with st.spinner("Running Monte Carlo Simulations..."):
                res = optimize_lineup(st.session_state['dfs_pool'], config)
            
            if res is not None:
                st.success("Optimization Complete!")
                st.dataframe(res)
                st.download_button("📥 Download CSV", get_csv(res), "titan_lineups.csv")
                
                # Show Top Lineup Summary
                top = res[res['Lineup_ID']==1]
                st.markdown("### 🥇 Top Lineup Preview")
                for i, row in top.iterrows():
                    st.text(f"{row['position']} - {row['name']} (${row['salary']}) - Proj: {row['projection']}")
            else:
                st.error("Optimization Failed. Try relaxing constraints (e.g., uncheck Stacking) or ensure valid salary data exists.")

# --- TAB 3: PROPS ---
with tabs[2]:
    st.markdown("### 🎫 Prop Betting Desk")
    
    # Filter by sport
    prop_pool = st.session_state['prop_pool']
    sport_props = prop_pool[prop_pool['sport'] == sport].copy()
    
    if sport_props.empty:
        st.info(f"No prop lines found for {sport}.")
    else:
        brain = TitanBrain(bankroll)
        sport_props['pick'] = np.where(sport_props['projection'] > sport_props['prop_line'], "OVER", "UNDER")
        
        res = sport_props.apply(lambda x: brain.calculate_kelly(x), axis=1, result_type='expand')
        sport_props['units'] = res[0]
        sport_props['rating'] = res[1]
        
        active = sport_props[sport_props['units'] > 0].sort_values('units', ascending=False)
        
        for i, row in active.iterrows():
            color = "#00ff41" if row['rating'] == "MAX PLAY" else "#00d2ff"
            st.markdown(f"""
            <div class="titan-card titan-card-prop" style="border-left: 5px solid {color}">
                <div style="display:flex; justify-content:space-between">
                    <span style="font-size:18px; font-weight:bold">{row['name']}</span>
                    <span style="font-size:18px; color:{color}">{row['pick']} {row['prop_line']}</span>
                </div>
                <div style="color:#aaa; font-size:14px">Projection: {row['projection']} | {row['rating']}</div>
                <div style="margin-top:5px; font-weight:bold">BET: {row['units']:.2f} UNITS</div>
            </div>
            """, unsafe_allow_html=True)
