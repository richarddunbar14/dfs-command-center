import streamlit as st
import pandas as pd
import numpy as np
import pulp
import base64

# ==========================================
# ⚙️ PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(layout="wide", page_title="DFS Command Center: ULTIMATE", page_icon="👑")

st.markdown("""
<style>
    /* Dark Theme & Professional Typography */
    .stApp { background-color: #0b0f19; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { color: #38bdf8; font-size: 24px; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 14px; }
    
    /* Tables */
    [data-testid="stDataFrame"] { border: 1px solid #1e293b; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    
    /* Buttons */
    .stButton>button { background-color: #3b82f6; color: white; border: none; font-weight: 600; padding: 0.5rem 1rem; border-radius: 6px; transition: all 0.2s; }
    .stButton>button:hover { background-color: #2563eb; transform: translateY(-1px); }
    
    /* Custom Badges */
    .badge-green { background-color: #059669; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
    .badge-red { background-color: #dc2626; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
    .badge-gold { background-color: #d97706; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
    
    /* Headers */
    h1, h2, h3 { color: #f8fafc; letter-spacing: -0.025em; }
    .highlight { color: #38bdf8; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🛠️ HELPER FUNCTIONS (DATA & EXPORT)
# ==========================================
def get_csv_download(df, filename="export.csv", label="Download CSV"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="stButton" style="text-decoration:none; color:#38bdf8; font-weight:bold;">📥 {label}</a>'

def standardize_columns(df):
    """
    The Universal Translator: Converts any CSV format into our internal standard.
    """
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('-', '_').str.replace('%', '')
    
    column_map = {
        'name': ['player', 'athlete', 'full_name', 'name_id', 'player_name'],
        'id': ['player_id', 'id', 'xid'],
        'proj_pts': ['projection', 'proj', 'fpts', 'fantasy_points', 'pts_proj', 'ppg', 'median'],
        'ceiling': ['ceil', 'max_pts', 'upside', 'boom'],
        'salary': ['cost', 'sal', 'price', 'salary_cap'],
        'team': ['squad', 'tm', 'team_id'],
        'position': ['pos', 'position_id', 'roster_position'],
        'opp_rank': ['dvp', 'opp_rank', 'defense_rank', 'vs_pos', 'opp_rank_pts'],
        'prop_line': ['line', 'prop', 'ou', 'total', 'over_under', 'strike'],
        'rest': ['days_rest', 'rest', 'b2b'],
        'injury': ['inj', 'status', 'injury_status'],
        'proj_mins': ['min', 'minutes', 'proj_min'],
        'usage': ['usg', 'usage_rate'],
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
    cols_to_numeric = ['proj_pts', 'salary', 'proj_mins', 'prop_line', 'ceiling', 'opp_rank', 'usage', 'air_yards']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # Position Standardization
    if 'position' in df.columns:
        df['position'] = df['position'].astype(str).str.upper().str.replace('/', ',')
        
    return df

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return standardize_columns(df)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# ==========================================
# 🧠 THE LOGIC ENGINES
# ==========================================

def run_shark_engine(df, sport, config):
    """
    The Context Engine: Calculates Smash Score based on DvP, Rest, and Narrative.
    """
    df['notes'] = ""
    
    # 1. Base Value
    if 'salary' in df.columns and df['salary'].sum() > 0:
        df['value_score'] = (df['proj_pts'] / df['salary']) * 1000
    else:
        df['value_score'] = 0
        
    # 2. Shark Score Initialization (0-100 scale)
    df['shark_score'] = (df['value_score'] / 5) * 50 
    
    # 3. DvP Adjustments (Defense vs Position)
    if 'opp_rank' in df.columns:
        # Rank 32 (Bad D) = Boost, Rank 1 (Good D) = Penalty
        df['shark_score'] += (df['opp_rank'] - 16) * 0.5
        df.loc[df['opp_rank'] > 25, 'notes'] += "🟢 Soft Matchup. "
        df.loc[df['opp_rank'] < 5, 'notes'] += "🔴 Tough Matchup. "
        
    # 4. Narrative & Psychology (God Mode Config)
    if config.get('narrative_active'):
        # Example: Revenge Game Logic (Placeholder for manual tags)
        # In a real app, you'd match player names to a list
        pass 
        
    # 5. Weather / Game Script
    if config.get('high_wind') and sport == "NFL":
        if 'position' in df.columns:
            # Downgrade QB/WR
            mask = df['position'].str.contains('QB|WR', na=False)
            df.loc[mask, 'shark_score'] -= 10
            df.loc[mask, 'notes'] += "🌪️ Wind Fade. "
            
    # 6. Usage / Air Yards Boost
    if 'usage' in df.columns and sport == "NBA":
        df.loc[df['usage'] > 30, 'shark_score'] += 8
        df.loc[df['usage'] > 30, 'notes'] += "⭐ High Usage. "
        
    if 'air_yards' in df.columns and sport == "NFL":
        # Buy Low Candidate
        mask = (df['air_yards'] > 100) & (df['salary'] < 5500)
        df.loc[mask, 'shark_score'] += 15
        df.loc[mask, 'notes'] += "🚀 AirYard BuyLow. "

    return df

def generate_ceiling(df, sport, volatility):
    """
    The Million Maker Engine: Creates upside projections based on volatility settings.
    """
    if 'ceiling' not in df.columns or df['ceiling'].sum() == 0:
        # Default Volatility Factors
        vol_map = 1.0
        if sport == "NFL":
            # WR/QB higher variance than RB
            conditions = [
                df['position'].str.contains('WR|QB', na=False),
                df['position'].str.contains('RB', na=False)
            ]
            choices = [1.3, 1.1]
            df['vol_factor'] = np.select(conditions, choices, default=1.0)
        elif sport == "NBA":
            # Cheap players have higher variance relative to projection
            df['vol_factor'] = np.where(df['salary'] < 5000, 1.4, 1.1)
        else:
            df['vol_factor'] = 1.0
            
        # Calculation: Proj * (1 + (Vol_Factor * User_Slider_Scale))
        scale = volatility / 10.0 # 0.1 to 1.0
        df['ceiling_proj'] = df['proj_pts'] * (1 + (df['vol_factor'] * scale))
    else:
        df['ceiling_proj'] = df['ceiling']
        
    return df

def optimize_lineup(df, salary_cap, sport, site, num_lineups, optimization_target, stack_team=None):
    valid_lineups = []
    
    # Filter Dead Players
    df = df[(df['proj_pts'] > 0) & (df['salary'] > 0)].copy()
    
    # Choose Source of Truth
    if optimization_target == "Base Projection":
        target_col = 'proj_pts'
    elif optimization_target == "Shark Score (Smart)":
        target_col = 'shark_score'
    elif optimization_target == "Ceiling (GPP/Milly Maker)":
        target_col = 'ceiling_proj'
        # Milly Maker Rule: Filter out low ceiling punts
        df = df[~((df['salary'] < 4500) & (df['ceiling_proj'] < 25))]
        
    # Re-index
    df = df.reset_index(drop=True)

    for _ in range(num_lineups):
        prob = pulp.LpProblem("DFS_Optimizer", pulp.LpMaximize)
        player_vars = pulp.LpVariable.dicts("P", df.index, cat='Binary')
        
        # Objective
        prob += pulp.lpSum([df.loc[i, target_col] * player_vars[i] for i in df.index])
        
        # Cap
        prob += pulp.lpSum([df.loc[i, 'salary'] * player_vars[i] for i in df.index]) <= salary_cap
        
        # --- SPORT LOGIC ---
        if sport == "NFL":
            # 9 Players
            prob += pulp.lpSum([player_vars[i] for i in df.index]) == 9
            
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
            
            # STACKING
            if stack_team:
                stackers = df[(df['team'] == stack_team) & (df['position'].isin(['WR', 'TE']))]
                qb_stack = df[(df['team'] == stack_team) & (df['position'].str.contains('QB'))]
                if not qb_stack.empty:
                    # If forcing a team stack, ensure we pick the QB
                    # Note: This is a simplified force. In advanced mode we'd tie variables.
                    prob += pulp.lpSum([player_vars[i] for i in qb_stack.index]) == 1
                    prob += pulp.lpSum([player_vars[i] for i in stackers.index]) >= 1
            
        elif sport == "NBA":
            # 8 Players
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

        # SOLVE
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            selected = [i for i in df.index if player_vars[i].varValue == 1]
            lineup = df.loc[selected].copy()
            lineup['Lineup_ID'] = _ + 1
            lineup['Total_Proj'] = lineup[target_col].sum()
            valid_lineups.append(lineup)
            
            # Diversity Constraint for next loop
            prob += pulp.lpSum([player_vars[i] for i in selected]) <= len(selected) - 1
        else:
            break
            
    return pd.concat(valid_lineups) if valid_lineups else None

def format_dk_export(df):
    rows = []
    for lid, group in df.groupby('Lineup_ID'):
        row = {}
        names = group['name'].tolist()
        for idx, name in enumerate(names):
            row[f'Player {idx+1}'] = name
        rows.append(row)
    return pd.DataFrame(rows)

# ==========================================
# 🖥️ THE MAIN APPLICATION UI
# ==========================================

# --- SIDEBAR CONTROLS ---
st.sidebar.title("👑 DFS Command Center")
sport = st.sidebar.selectbox("Sport", ["NBA", "NFL", "MLB", "NHL"], index=1)
site = st.sidebar.selectbox("Platform", ["DraftKings", "FanDuel"], index=0)
st.sidebar.markdown("---")

app_mode = st.sidebar.radio("COMMAND MODULES", [
    "1. ⚔️ The War Room (Data)",
    "2. 🔬 The Lab (Analysis)", 
    "3. 🏗️ Slate Breaker (Optimizer)",
    "4. 💸 Prop Sniper (Betting)"
])

if 'data_store' not in st.session_state: st.session_state['data_store'] = {}

# --- MODULE 1: THE WAR ROOM (DATA INGEST & EDIT) ---
if app_mode == "1. ⚔️ The War Room (Data)":
    st.title(f"⚔️ {sport} War Room")
    st.markdown("### Step 1: Intelligence Ingest")
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("Upload Projections (CSV)")
        f1 = st.file_uploader("Projections / DFS Export", type='csv')
    with c2:
        st.info("Upload Props (Optional)")
        f2 = st.file_uploader("Props.cash / Underdog CSV", type='csv')
        
    if f1:
        df = load_data(f1)
        if f2:
            props = load_data(f2)
            if 'prop_line' in props.columns and 'name' in props.columns:
                # Merge logic
                df = df.merge(props[['name', 'prop_line']], on='name', how='left')
                st.success("✅ Props Merged Successfully!")
        
        # Save to Session
        st.session_state['data_store'][sport] = df
        st.success(f"✅ Active Roster: {len(df)} Players")
        
    # --- GOD MODE EDITOR ---
    if sport in st.session_state['data_store']:
        st.markdown("---")
        st.markdown("### Step 2: God Mode Adjustments (Manual Overrides)")
        
        # Config Panel
        with st.expander("🌍 Environmental & Script Factors"):
            c1, c2, c3 = st.columns(3)
            high_wind = c1.checkbox("🌪️ High Wind (>15mph)", value=False)
            narrative = c2.checkbox("😤 Narrative/Revenge Boosts", value=True)
            volatility = c3.slider("⚡ Slate Volatility (1-10)", 1, 10, 5)
            
            # Save config to state
            config = {'high_wind': high_wind, 'narrative_active': narrative, 'volatility': volatility}
            
        # Run Engines
        active_df = st.session_state['data_store'][sport].copy()
        
        # 1. Run Shark Engine
        active_df = run_shark_engine(active_df, sport, config)
        # 2. Run Ceiling Engine
        active_df = generate_ceiling(active_df, sport, volatility)
        
        # Save processed data back
        st.session_state['data_store'][f"{sport}_PROCESSED"] = active_df
        
        st.dataframe(active_df.head(8))
        st.caption("Data processed with Shark Score & Ceiling Algorithms.")

# --- MODULE 2: THE LAB (ANALYSIS) ---
elif app_mode == "2. 🔬 The Lab (Analysis)":
    st.title(f"🔬 {sport} Strategy Lab")
    
    if f"{sport}_PROCESSED" not in st.session_state['data_store']:
        st.warning("⚠️ Go to War Room and Upload Data first.")
    else:
        df = st.session_state['data_store'][f"{sport}_PROCESSED"]
        
        tab1, tab2, tab3 = st.tabs(["🦈 Shark Board", "💎 Value Nukes", "📈 Advanced Metrics"])
        
        with tab1:
            st.markdown("### 🦈 Top Rated Plays (Shark Score)")
            st.markdown("Combines **Value**, **Matchup (DvP)**, and **Prop Edges**.")
            st.dataframe(
                df.sort_values(by='shark_score', ascending=False).head(20)
                [['name', 'team', 'position', 'salary', 'proj_pts', 'shark_score', 'notes']]
                .style.background_gradient(subset=['shark_score'], cmap='coolwarm')
            )
            
        with tab2:
            st.markdown("### 💎 GPP Ceiling Plays (Milly Maker)")
            st.markdown("Players with massive upside relative to salary.")
            nukes = df[(df['salary'] < 6000) & (df['ceiling_proj'] > 25)].sort_values(by='ceiling_proj', ascending=False)
            st.dataframe(
                nukes[['name', 'salary', 'proj_pts', 'ceiling_proj', 'notes']]
                .style.background_gradient(subset=['ceiling_proj'], cmap='Greens')
            )
            
        with tab3:
            st.markdown("### 📈 Trend Spotter")
            c1, c2 = st.columns(2)
            with c1:
                if 'usage' in df.columns:
                    st.write("**Usage Monsters (>30%)**")
                    st.dataframe(df[df['usage'] > 30][['name', 'usage']])
            with c2:
                if 'opp_rank' in df.columns:
                    st.write("**Softest Matchups (Rank > 28)**")
                    st.dataframe(df[df['opp_rank'] > 28][['name', 'opp_rank', 'position']])

# --- MODULE 3: SLATE BREAKER (OPTIMIZER) ---
elif app_mode == "3. 🏗️ Slate Breaker (Optimizer)":
    st.title(f"🏗️ {site} {sport} Optimizer")
    
    if f"{sport}_PROCESSED" not in st.session_state['data_store']:
        st.warning("⚠️ Data missing.")
    else:
        df = st.session_state['data_store'][f"{sport}_PROCESSED"]
        
        c1, c2, c3 = st.columns(3)
        num_lineups = c1.number_input("Lineups", 1, 50, 5)
        salary_cap = c2.number_input("Salary Cap", value=(60000 if site=="FanDuel" else 50000))
        target = c3.selectbox("Optimization Source", ["Shark Score (Smart)", "Ceiling (GPP/Milly Maker)", "Base Projection"])
        
        stack_team = None
        if sport in ["NFL", "MLB"]:
            stack_team = st.selectbox("Force Team Stack", [None] + sorted(df['team'].astype(str).unique()))
            
        if st.button("🚀 BUILD LINEUPS"):
            with st.spinner(f"Optimizing using {target} logic..."):
                res = optimize_lineup(df, salary_cap, sport, site, num_lineups, target, stack_team)
            
            if res is not None:
                st.success(f"✅ Built {num_lineups} Lineups!")
                
                # Summary Stats
                avg_score = res.groupby('Lineup_ID')['proj_pts'].sum().mean()
                avg_cost = res.groupby('Lineup_ID')['salary'].sum().mean()
                st.metric("Avg Projected Score", f"{avg_score:.1f}", f"${avg_cost:,.0f} Cost")
                
                # Export
                st.markdown("### 📤 Export to CSV")
                export_df = format_dk_export(res)
                st.markdown(get_csv_download(export_df, f"{site}_{sport}_lineups.csv"), unsafe_allow_html=True)
                
                # Detailed View
                st.markdown("### 🧐 Lineup Inspector")
                for lid, group in res.groupby('Lineup_ID'):
                    with st.expander(f"Lineup #{lid} ({group['proj_pts'].sum():.1f} Pts)"):
                        st.dataframe(group[['name', 'position', 'team', 'salary', 'proj_pts', 'shark_score', 'notes']])
            else:
                st.error("Infeasible. Check constraints.")

# --- MODULE 4: PROP SNIPER ---
elif app_mode == "4. 💸 Prop Sniper (Betting)":
    st.title("💸 Unlimited Edge Calculator")
    
    if f"{sport}_PROCESSED" not in st.session_state['data_store']:
        st.warning("⚠️ Upload Data with Props first.")
    else:
        df = st.session_state['data_store'][f"{sport}_PROCESSED"]
        
        if 'prop_line' not in df.columns or df['prop_line'].sum() == 0:
            st.error("❌ No Prop Lines found in your data.")
        else:
            # Calc Edge
            df['edge_pct'] = ((df['proj_pts'] - df['prop_line']) / df['prop_line']) * 100
            df['pick'] = np.where(df['edge_pct'] > 0, 'OVER', 'UNDER')
            
            # Master Pools
            ud_pool = df[df['edge_pct'] > -0.5].sort_values(by='edge_pct', ascending=False)
            
            t1, t2 = st.tabs(["🐶 Underdog / PrizePicks", "🎯 Pick6 / Parlay"])
            
            with t1:
                st.subheader("Volume Pool (Edge > -0.5%)")
                st.dataframe(
                    ud_pool[['name', 'prop_line', 'proj_pts', 'pick', 'edge_pct', 'notes']]
                    .style.background_gradient(subset=['edge_pct'], cmap='RdYlGn')
                )
                
            with t2:
                st.subheader("High Confidence (Edge > 5%)")
                safe_pool = df[df['edge_pct'].abs() > 5].sort_values(by='edge_pct', ascending=False)
                st.dataframe(
                    safe_pool[['name', 'prop_line', 'proj_pts', 'pick', 'edge_pct']]
                    .style.background_gradient(subset=['edge_pct'], cmap='Blues')
                )
