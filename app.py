import streamlit as st
import pandas as pd
import numpy as np
import pulp
import re
from datetime import datetime

# --- CONFIG ---
st.set_page_config(layout="wide", page_title="TITAN OMNI: V40.0", page_icon="⚡")
st.markdown("""<style>
    .stApp { background-color: #0e1117; color: #fafafa; font-family: monospace; }
    .titan-card { background: #1f2937; border: 1px solid #374151; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; background: #3b82f6; color: white; border: none; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #60a5fa; }
</style>""", unsafe_allow_html=True)

if 'master_pool' not in st.session_state: st.session_state['master_pool'] = pd.DataFrame()

# --- DATA ENGINE ---
class DataEngine:
    @staticmethod
    def clean_col(col): return re.sub(r'[^a-zA-Z0-9]', '', str(col).lower())

    @staticmethod
    def normalize(df):
        # standardize column names
        col_map = {}
        # KEYWORD MAPPING
        keywords = {
            'name': ['player', 'name', 'who', 'athlete'],
            'projection': ['proj', 'fpts', 'points', 'fp', 'prediction'],
            'salary': ['sal', 'cost', 'price', 'cap'],
            'position': ['pos', 'slot', 'roster'],
            'team': ['team', 'tm', 'squad'],
            'opp': ['opp', 'vs', 'opponent'],
            'prop_line': ['line', 'prop', 'total', 'strike'],
            'game_info': ['game', 'matchup', 'schedule'],
            'status': ['status', 'inj', 'injury']
        }

        for c in df.columns:
            clean = DataEngine.clean_col(c)
            for key, variants in keywords.items():
                if any(v in clean for v in variants):
                    if key not in col_map.values(): 
                        col_map[c] = key
        
        df = df.rename(columns=col_map)
        
        # clean numbers
        for col in ['projection', 'salary', 'prop_line']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # ensure essential cols exist
        if 'name' not in df.columns: return pd.DataFrame() # Invalid file
        if 'projection' not in df.columns: df['projection'] = 0.0
        if 'salary' not in df.columns: df['salary'] = 0.0
        if 'position' not in df.columns: df['position'] = 'FLEX'
        if 'team' not in df.columns: df['team'] = 'N/A'
        
        return df

    @staticmethod
    def get_roster_rules(sport, site):
        """Returns simplified constraints to prevent solver failure"""
        rules = {'size': 6, 'cap': 50000, 'pos_limits': []}
        
        if site == 'DK': rules['cap'] = 50000
        elif site == 'FD': rules['cap'] = 60000
        elif site == 'Yahoo': rules['cap'] = 200
        else: rules['cap'] = 999999 # Pickem
        
        # Simplified Sport Rules
        if sport == 'NFL':
            rules['size'] = 9
            if site == 'DK': rules['pos_limits'] = [('QB', 1, 1), ('RB', 2, 3), ('WR', 3, 4), ('TE', 1, 2), ('DST', 1, 1)]
        elif sport == 'NBA':
            rules['size'] = 8
            if site == 'DK': rules['pos_limits'] = [('PG', 1, 3), ('SG', 1, 3), ('SF', 1, 3), ('PF', 1, 3), ('C', 1, 2)]
        
        return rules

# --- OPTIMIZER ENGINE ---
def run_optimizer(df, config):
    # Filter
    pool = df.copy()
    pool = pool[pool['projection'] > 0]
    if config.get('require_salary'):
        pool = pool[pool['salary'] > 0]
        
    if pool.empty: return None
    
    # Setup Problem
    prob = pulp.LpProblem("Titan_Omni", pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("p", pool.index, cat='Binary')
    
    # Objective
    prob += pulp.lpSum([pool.loc[i, 'projection'] * player_vars[i] for i in pool.index])
    
    # Salary Constraint
    if config.get('salary_cap'):
        prob += pulp.lpSum([pool.loc[i, 'salary'] * player_vars[i] for i in pool.index]) <= config['salary_cap']
    
    # Size Constraint
    prob += pulp.lpSum([player_vars[i] for i in pool.index]) == config['roster_size']
    
    # Position Constraints (Flexible Match)
    for pos, min_q, max_q in config.get('pos_limits', []):
        eligible = [i for i in pool.index if pos in str(pool.loc[i, 'position'])]
        if eligible:
            prob += pulp.lpSum([player_vars[i] for i in eligible]) >= min_q
            prob += pulp.lpSum([player_vars[i] for i in eligible]) <= max_q
            
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if prob.status == 1:
        selected_indices = [i for i in pool.index if player_vars[i].varValue == 1]
        return pool.loc[selected_indices]
    return None

# --- UI ---
st.title("TITAN OMNI V40")

# 1. UPLOAD
st.subheader("1. Upload Data")
uploaded_files = st.file_uploader("Upload CSVs (DFS or Props)", accept_multiple_files=True)

if uploaded_files:
    all_data = []
    for f in uploaded_files:
        try:
            df = pd.read_csv(f)
            clean_df = DataEngine.normalize(df)
            if not clean_df.empty:
                all_data.append(clean_df)
        except: pass
    
    if all_data:
        # merge all
        st.session_state['master_pool'] = pd.concat(all_data).groupby('name', as_index=False).max()
        st.success(f"Loaded {len(st.session_state['master_pool'])} players")
    else:
        st.error("Could not read files. Ensure they are CSVs with Player Names.")

# 2. VIEW & CONFIGURE
if not st.session_state['master_pool'].empty:
    st.subheader("2. Settings")
    c1, c2 = st.columns(2)
    sport = c1.selectbox("Sport", ["NFL", "NBA", "MLB", "NHL", "CBB"])
    site = c2.selectbox("Site", ["DK", "FD", "PrizePicks"])
    
    # Dynamic Config
    rules = DataEngine.get_roster_rules(sport, site)
    
    # 3. OPTIMIZE
    st.subheader("3. Generate Lineup")
    
    if st.button("Run Optimizer"):
        cfg = {
            'salary_cap': rules['cap'], 
            'roster_size': rules['size'],
            'pos_limits': rules['pos_limits'],
            'require_salary': (site in ['DK', 'FD'])
        }
        res = run_optimizer(st.session_state['master_pool'], cfg)
        
        if res is not None:
            st.success(f"Optimal Lineup Found! Proj: {res['projection'].sum():.2f}")
            st.dataframe(res)
        else:
            st.error("No feasible lineup found. Check constraints or pool size.")
            
    # 4. PROPS
    st.subheader("4. Top Props")
    df = st.session_state['master_pool']
    if 'prop_line' in df.columns and 'projection' in df.columns:
        df['edge'] = (df['projection'] - df['prop_line']) / df['prop_line']
        props = df[df['prop_line'] > 0].sort_values('edge', ascending=False).head(10)
        st.dataframe(props[['name', 'prop_line', 'projection', 'edge']])
