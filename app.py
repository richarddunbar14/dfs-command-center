import streamlit as st
import pandas as pd
import numpy as np
import pulp
import base64
import io

# ==========================================
# ⚙️ 1. PAGE CONFIG & DARK MODE STYLING
# ==========================================
st.set_page_config(layout="wide", page_title="DFS Command Center: TITAN", page_icon="⚡")

st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    
    /* Heads Up Display (HUD) Metrics */
    div[data-testid="stMetricValue"] { color: #38bdf8; font-size: 26px; font-weight: 800; }
    div[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
    
    /* Data Grids */
    [data-testid="stDataFrame"] { border: 1px solid #1e293b; border-radius: 8px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
    
    /* Buttons */
    .stButton>button { background-color: #3b82f6; color: white; border: none; font-weight: 700; padding: 0.6rem 1.2rem; border-radius: 6px; }
    .stButton>button:hover { background-color: #2563eb; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2); }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #1e293b; border-radius: 4px; color: #94a3b8; }
    .stTabs [aria-selected="true"] { background-color: #3b82f6; color: white; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🛠️ 2. THE UNIVERSAL TRANSLATOR (INGEST ENGINE)
# ==========================================

def get_csv_download(df, filename="export.csv", label="Download CSV"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="stButton" style="text-decoration:none; color:#38bdf8; font-weight:bold;">📥 {label}</a>'

def detect_platform(df, filename):
    cols = [c.lower() for c in df.columns]
    fname = filename.lower()
    if 'pick6' in fname or 'pick6' in cols: return "Pick6"
    if 'prizepicks' in fname or 'payout' in cols: return "PrizePicks"
    if 'sleeper' in fname or 'stat_type' in cols: return "Sleeper"
    if 'underdog' in fname or 'higher_lower' in cols: return "Underdog"
    return "Projection Source"

def standardize_columns(df):
    """
    Maps 100+ variations of column names to a standard set.
    """
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('-', '_').str.replace('%', '')
    
    column_map = {
        'name': ['player', 'athlete', 'full_name', 'name_id', 'player_name', 'who'],
        'team': ['squad', 'tm', 'team_id', 'org'],
        'position': ['pos', 'position_id', 'roster_position', 'slot'],
        'proj_pts': ['projection', 'proj', 'fpts', 'fantasy_points', 'pts_proj', 'ppg', 'median', 'base_proj'],
        'ceiling': ['ceil', 'max_pts', 'upside', 'boom', '95th'],
        'salary': ['cost', 'sal', 'price', 'salary_cap'],
        'prop_line': ['line', 'prop', 'ou', 'total', 'over_under', 'strike', 'stat_value'],
        'opp_rank': ['dvp', 'opp_rank', 'defense_rank', 'vs_pos', 'rank'],
        'rest': ['days_rest', 'rest', 'b2b', 'days_off'],
        'injury': ['inj', 'status', 'injury_status', 'health'],
        'usage': ['usg', 'usage_rate', 'usage_pct'],
        'minutes': ['min', 'minutes', 'proj_min', 'avg_min'],
        'air_yards': ['air_yards', 'ay', 'intended_air_yards']
    }
    
    renamed_cols = {}
    for standard, variations in column_map.items():
        if standard not in df.columns:
            for v in variations:
                match = next((c for c in df.columns if v in c), None)
                if match:
                    renamed_cols[match] = standard
                    break
    if renamed_cols:
        df = df.rename(columns=renamed_cols)
        
    # Numeric Cleanup
    cols_to_numeric = ['proj_pts', 'salary', 'prop_line', 'ceiling', 'opp_rank', 'usage', 'minutes', 'air_yards']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    if 'position' in df.columns:
        df['position'] = df['position'].astype(str).str.upper().str.replace('/', ',')
        
    return df

def process_files(uploaded_files):
    master_df = pd.DataFrame()
    prop_dfs = []
    
    for file in uploaded_files:
        try:
            if file.name.endswith('.csv'): df = pd.read_csv(file)
            else: df = pd.read_excel(file)
            
            df = standardize_columns(df)
            platform = detect_platform(df, file.name)
            df['platform'] = platform
            
            # Logic: If it has Salary, it's a Base Projection. If it has Lines but no salary, it's Props.
            if 'salary' in df.columns:
                if master_df.empty: master_df = df
                else: 
                    # Merge fields
                    cols = [c for c in df.columns if c not in master_df.columns or c == 'name']
                    if 'name' in df.columns: master_df = master_df.merge(df[cols], on='name', how='left')
            elif 'prop_line' in df.columns:
                prop_dfs.append(df)
            else:
                # Fallback merge
                if master_df.empty: master_df = df
                
        except Exception as e:
            st.error(f"Error in {file.name}: {e}")
            
    # Merge Props into Master for single view
    if not master_df.empty:
        for p_df in prop_dfs:
            if 'name' in p_df.columns:
                plat = p_df['platform'].iloc[0]
                # Keep specific line for that platform
                rename_map = {'prop_line': f'line_{plat}'}
                temp = p_df.rename(columns=rename_map)
                if f'line_{plat}' in temp.columns:
                    master_df = master_df.merge(temp[['name', f'line_{plat}']], on='name', how='left')
                    
    return master_df, prop_dfs

# ==========================================
# 🧠 3. THE SHARK ENGINE (METRICS & GOD MODE)
# ==========================================

def run_shark_engine(df, sport, config):
    df['notes'] = ""
    df['shark_score'] = 50.0 # Base
    
    # Value Boost
    if 'salary' in df.columns:
        df['value'] = np.where(df['salary']>0, (df['proj_pts']/df['salary'])*1000, 0)
        df['shark_score'] += (df['value'] - 4.5) * 8
    
    # DvP Boost
    if 'opp_rank' in df.columns:
        df['shark_score'] += (df['opp_rank'] - 16) * 0.5
        df.loc[df['opp_rank'] > 25, 'notes'] += "🟢 Soft D. "
        
    # Weather / Narrative (God Mode)
    if config.get('narrative_active'):
        # Placeholder: In a real app, you'd have a list of 'Revenge Players'
        pass
    if config.get('high_wind') and sport == "NFL":
        if 'position' in df.columns:
            mask = df['position'].str.contains('QB|WR', na=False)
            df.loc[mask, 'shark_score'] -= 10
            df.loc[mask, 'notes'] += "🌪️ Wind. "
            
    # Prop Validation
    # Boost if ANY prop line has a huge edge
    prop_cols = [c for c in df.columns if 'line_' in c]
    for p_col in prop_cols:
        edge = ((df['proj_pts'] - df[p_col]) / df[p_col]) * 100
        mask = edge > 15
        df.loc[mask, 'shark_score'] += 10
        df.loc[mask, 'notes'] += f"💰 {p_col.replace('line_','')} Edge. "
        
    return df

def calculate_ceiling(df, sport, volatility):
    # Milly Maker Logic
    vol_factor = 1.0
    if sport == "NFL":
        conds = [df['position'].str.contains('WR|QB', na=False), df['position'].str.contains('RB', na=False)]
        df['vol'] = np.select(conds, [1.3, 1.1], default=1.0)
    elif sport == "NBA":
        df['vol'] = np.where(df['salary'] < 5000, 1.4, 1.1)
    else:
        df['vol'] = 1.0
        
    df['ceiling_proj'] = df['proj_pts'] * (1 + (df['vol'] * (volatility/10)))
    return df

# ==========================================
# 🏗️ 4. THE OPTIMIZER (SLATE BREAKER)
# ==========================================

def optimize_lineup(df, salary_cap, sport, site, num_lineups, target_col, stack_team=None):
    valid_lineups = []
    df = df[(df['proj_pts'] > 0) & (df['salary'] > 0)].reset_index(drop=True)
    
    for _ in range(num_lineups):
        prob = pulp.LpProblem("DFS", pulp.LpMaximize)
        player_vars = pulp.LpVariable.dicts("P", df.index, cat='Binary')
        
        prob += pulp.lpSum([df.loc[i, target_col] * player_vars[i] for i in df.index])
        prob += pulp.lpSum([df.loc[i, 'salary'] * player_vars[i] for i in df.index]) <= salary_cap
        
        # Sport Logic
        if sport == "NFL":
            prob += pulp.lpSum([player_vars[i] for i in df.index]) == 9
            # Positional Constraints
            qbs = df[df['position'].str.contains('QB', na=False)]
            dsts = df[df['position'].str.contains('DST|DEF', na=False)]
            rbs = df[df['position'].str.contains('RB', na=False)]
            wrs = df[df['position'].str.contains('WR', na=False)]
            tes = df[df['position'].str.contains('TE', na=False)]
            
            prob += pulp.lpSum([player_vars[i] for i in qbs.index]) == 1
            prob += pulp.lpSum([player_vars[i] for i in dsts.index]) == 1
            prob += pulp.lpSum([player_vars[i] for i in rbs.index]) >= 2
            prob += pulp.lpSum([player_vars[i] for i in wrs.index]) >= 3
            prob += pulp.lpSum([player_vars[i] for i in tes.index]) >= 1
            
            if stack_team:
                stackers = df[(df['team'] == stack_team) & (df['position'].isin(['WR','TE']))]
                qb_stack = df[(df['team'] == stack_team) & (df['position'].str.contains('QB'))]
                if not qb_stack.empty:
                    # If picking this QB, must pick WR
                    # Simplified constraint: Just force at least 1 stacker
                     prob += pulp.lpSum([player_vars[i] for i in stackers.index]) >= 1

        elif sport == "NBA":
            prob += pulp.lpSum([player_vars[i] for i in df.index]) == 8
            pgs = df[df['position'].str.contains('PG', na=False)]
            sgs = df[df['position'].str.contains('SG', na=False)]
            sfs = df[df['position'].str.contains('SF', na=False)]
            pfs = df[df['position'].str.contains('PF', na=False)]
            cs = df[df['position'].str.contains('C', na=False)]
            
            prob += pulp.lpSum([player_vars[i] for i in pgs.index]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in sgs.index]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in sfs.index]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in pfs.index]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in cs.index]) >= 1
            
        elif sport == "MLB":
            prob += pulp.lpSum([player_vars[i] for i in df.index]) == 10
            ps = df[df['position'].str.contains('P|SP', na=False)]
            prob += pulp.lpSum([player_vars[i] for i in ps.index]) == 2
            
        elif sport == "NHL":
            prob += pulp.lpSum([player_vars[i] for i in df.index]) == 9
            gs = df[df['position'].str.contains('G', na=False)]
            prob += pulp.lpSum([player_vars[i] for i in gs.index]) == 1

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            selected = [i for i in df.index if player_vars[i].varValue == 1]
            lineup = df.loc[selected].copy()
            lineup['Lineup_ID'] = _ + 1
            valid_lineups.append(lineup)
            # Constraints for next lineup to be unique
            prob += pulp.lpSum([player_vars[i] for i in selected]) <= len(selected) - 1
        else:
            break
    return pd.concat(valid_lineups) if valid_lineups else None

def format_export(df, site):
    rows = []
    for lid, group in df.groupby('Lineup_ID'):
        row = {}
        names = group['name'].tolist()
        for idx, name in enumerate(names):
            row[f'Player {idx+1}'] = name
        rows.append(row)
    return pd.DataFrame(rows)

# ==========================================
# 🖥️ 5. MAIN UI LAYOUT
# ==========================================

st.sidebar.title("🧬 TITAN COMMAND")
sport = st.sidebar.selectbox("Sport", ["NBA", "NFL", "MLB", "NHL", "PGA"])
site = st.sidebar.selectbox("Platform", ["DraftKings", "FanDuel"])
st.sidebar.markdown("---")

# Session States
if 'master' not in st.session_state: st.session_state['master'] = pd.DataFrame()
if 'props' not in st.session_state: st.session_state['props'] = []

# TABS FOR NAVIGATION
tab_load, tab_war_room, tab_optimizer, tab_betting = st.tabs([
    "1. 📂 Omni-Ingest", 
    "2. ⚔️ War Room (God Mode)", 
    "3. 🏗️ Slate Breaker", 
    "4. 💸 Prop Sniper"
])

# --- TAB 1: OMNI-INGEST ---
with tab_load:
    st.markdown("### 🧬 The Universal Data Hub")
    st.info("Drag & Drop UNLIMITED files (Projections, Props.cash, Underdog CSVs, Pick6). We sort it all.")
    
    files = st.file_uploader("Upload Files", accept_multiple_files=True, type=['csv','xlsx'])
    
    if files:
        if st.button("🚀 Process & Merge All"):
            with st.spinner("Running Universal Translator..."):
                m, p = process_files(files)
                st.session_state['master'] = m
                st.session_state['props'] = p
                st.success(f"✅ Merged {len(m)} Players from {len(files)} Sources!")

# --- TAB 2: WAR ROOM (ANALYSIS) ---
with tab_war_room:
    df = st.session_state['master']
    if df.empty:
        st.warning("Upload data first.")
    else:
        st.markdown("### ⚔️ Tactical Adjustments (God Mode)")
        
        c1, c2, c3 = st.columns(3)
        narrative = c1.checkbox("😤 Narrative/Revenge Boosts", value=True)
        high_wind = c2.checkbox("🌪️ High Wind (>15mph)", value=False)
        volatility = c3.slider("⚡ Slate Volatility (1-10)", 1, 10, 5)
        
        config = {'narrative_active': narrative, 'high_wind': high_wind}
        
        # RUN ENGINES
        df = run_shark_engine(df, sport, config)
        df = calculate_ceiling(df, sport, volatility)
        st.session_state['master'] = df # Update state
        
        # DISPLAY SHARK BOARD
        st.markdown("### 🦈 The Shark Board")
        
        # Dynamic Columns
        cols = ['name', 'team', 'position', 'salary', 'proj_pts', 'shark_score', 'ceiling_proj', 'notes']
        line_cols = [c for c in df.columns if 'line_' in c]
        disp_cols = cols + line_cols
        disp_cols = [c for c in disp_cols if c in df.columns]
        
        st.dataframe(
            df.sort_values(by='shark_score', ascending=False)[disp_cols]
            .style.background_gradient(subset=['shark_score'], cmap='coolwarm')
            .format({'shark_score': '{:.1f}', 'ceiling_proj': '{:.1f}'}),
            use_container_width=True
        )

# --- TAB 3: OPTIMIZER ---
with tab_optimizer:
    df = st.session_state['master']
    if df.empty: st.warning("Upload data first.")
    else:
        st.markdown(f"### 🏗️ {site} {sport} Optimizer")
        
        c1, c2, c3 = st.columns(3)
        num_lineups = c1.number_input("Count", 1, 50, 5)
        cap = c2.number_input("Cap", 50000 if site=="DraftKings" else 60000)
        target = c3.selectbox("Goal", ["Shark Score (Smart)", "Ceiling (Milly Maker)", "Base Projection"])
        
        stack = None
        if sport in ["NFL", "MLB"]:
            teams = sorted(df['team'].astype(str).unique())
            stack = st.selectbox("Force Team Stack", [None] + teams)
            
        if st.button("⚡ GENERATE LINEUPS"):
            with st.spinner("Optimizing..."):
                target_col = 'shark_score' if target == "Shark Score (Smart)" else 'ceiling_proj' if target == "Ceiling (Milly Maker)" else 'proj_pts'
                res = optimize_lineup(df, cap, sport, site, num_lineups, target_col, stack)
            
            if res is not None:
                st.success("✅ Lineups Built!")
                
                # Stats
                avg_ceil = res.groupby('Lineup_ID')['ceiling_proj'].sum().mean()
                st.metric("Avg Ceiling", f"{avg_ceil:.1f}")
                
                # Export
                exp = format_export(res, site)
                st.markdown(get_csv_download(exp, f"{site}_{sport}_Lineups.csv"), unsafe_allow_html=True)
                
                # View
                for lid, group in res.groupby('Lineup_ID'):
                    with st.expander(f"Lineup #{lid} ({group['proj_pts'].sum():.1f} Pts)"):
                        st.dataframe(group[['name', 'position', 'salary', 'proj_pts', 'shark_score', 'notes']])
            else:
                st.error("Infeasible. Check constraints.")

# --- TAB 4: PROP SNIPER ---
with tab_betting:
    st.markdown("### 💸 Platform-Specific Prop Sniper")
    props = st.session_state['props']
    master = st.session_state['master']
    
    if not props:
        st.warning("No Prop files uploaded.")
    else:
        # Create Tabs for each platform
        platforms = [d['platform'].iloc[0] for d in props]
        ptabs = st.tabs(platforms)
        
        for i, tab in enumerate(ptabs):
            with tab:
                p_df = props[i]
                plat = platforms[i]
                
                # Ensure projections are attached
                if 'proj_pts' not in p_df.columns and 'name' in p_df.columns:
                     p_df = p_df.merge(master[['name', 'proj_pts', 'shark_score', 'notes']], on='name', how='left')
                
                if 'proj_pts' in p_df.columns and 'prop_line' in p_df.columns:
                    p_df['edge'] = ((p_df['proj_pts'] - p_df['prop_line']) / p_df['prop_line']) * 100
                    p_df['pick'] = np.where(p_df['edge'] > 0, 'OVER', 'UNDER')
                    
                    st.dataframe(
                        p_df.sort_values(by='edge', ascending=False)
                        [['name', 'prop_line', 'proj_pts', 'pick', 'edge', 'notes']]
                        .style.background_gradient(subset=['edge'], cmap='RdYlGn')
                        .format({'edge': '{:.1f}%'}),
                        use_container_width=True
                    )
