import streamlit as st
import pandas as pd
import numpy as np
import pulp

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="DFS Command Center", page_icon="🧠")

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    div[data-testid="stMetricValue"] { color: #4ade80; font-size: 26px; font-weight: bold; }
    div[data-testid="stExpander"] { background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; }
    .big-font { font-size: 18px !important; color: #94a3b8; }
    .success-text { color: #4ade80; font-weight: bold; }
    .danger-text { color: #f87171; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        # Normalize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def optimize_lineup(df, salary_cap, roster_size, strategy="Optimal"):
    problem = pulp.LpProblem("DFS_Optimization", pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("Player", df.index, cat='Binary')
    
    # Constraints
    problem += pulp.lpSum([df.loc[i, 'proj_pts'] * player_vars[i] for i in df.index]) # Objective
    problem += pulp.lpSum([df.loc[i, 'salary'] * player_vars[i] for i in df.index]) <= salary_cap
    problem += pulp.lpSum([player_vars[i] for i in df.index]) == roster_size
    
    if strategy == "Stars & Scrubs (NBA)":
        problem += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'salary'] >= 10000]) >= 2
        
    solve_status = problem.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if solve_status == pulp.LpStatusOptimal:
        selected = [i for i in df.index if player_vars[i].varValue == 1]
        return df.loc[selected]
    return None

# --- SIDEBAR ---
st.sidebar.title("🧠 Sports Brain")
app_mode = st.sidebar.radio("Navigation", 
    ["📂 Data Upload", "🔬 Strategy Lab", "🏗️ Lineup Builder", "💸 Betting Edge Calculator"])

# --- SESSION STATE ---
if 'master_df' not in st.session_state:
    # Mock Data with 'edge' implied
    mock_data = pd.DataFrame([
        {"name": "Tyus Jones", "team": "WAS", "position": "PG", "salary": 4300, "proj_mins": 32.0, "avg_mins": 18.5, "proj_pts": 30.4, "prop_line": 28.5, "confidence": 0.75},
        {"name": "Naz Reid", "team": "MIN", "position": "C", "salary": 3800, "proj_mins": 28.0, "avg_mins": 14.0, "proj_pts": 32.2, "prop_line": 25.5, "confidence": 0.82},
        {"name": "LeBron James", "team": "LAL", "position": "SF", "salary": 10500, "proj_mins": 36.0, "avg_mins": 35.0, "proj_pts": 52.2, "prop_line": 50.5, "confidence": 0.55},
        {"name": "Luka Doncic", "team": "DAL", "position": "PG", "salary": 11800, "proj_mins": 38.0, "avg_mins": 37.0, "proj_pts": 61.0, "prop_line": 62.5, "confidence": 0.60},
        {"name": "Christian Braun", "team": "DEN", "position": "SG", "salary": 3500, "proj_mins": 18.0, "avg_mins": 15.0, "proj_pts": 13.5, "prop_line": 15.5, "confidence": 0.40},
    ])
    st.session_state['master_df'] = mock_data

# =========================================================
# 📂 TAB 1: DATA UPLOAD
# =========================================================
if app_mode == "📂 Data Upload":
    st.title("📂 Upload Your Data")
    st.markdown("Upload your CSV from **Props.cash**, **Underdog**, or **DFS Sites**.")
    uploaded_file = st.file_uploader("Drag and drop CSV here", type=['csv', 'xlsx'])
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state['master_df'] = df
            st.success(f"✅ Loaded {len(df)} players!")
            st.dataframe(df.head())

# =========================================================
# 🔬 TAB 2: STRATEGY LAB (DFS)
# =========================================================
elif app_mode == "🔬 Strategy Lab":
    st.title("🔬 Strategy Lab (DFS)")
    df = st.session_state['master_df']
    strategy = st.radio("Select Strategy:", ["⏰ Minutes > Talent", "🏈 NFL Stacking"])
    
    if strategy == "⏰ Minutes > Talent":
        if 'proj_mins' in df.columns and 'avg_mins' in df.columns:
            df['minute_diff'] = df['proj_mins'] - df['avg_mins']
            df['value'] = (df['proj_pts'] / df['salary']) * 1000
            opportunities = df[(df['minute_diff'] >= 5.0) & (df['salary'] <= 6000)].sort_values(by='value', ascending=False)
            if not opportunities.empty:
                st.dataframe(opportunities[['name', 'salary', 'minute_diff', 'value']])
            else:
                st.warning("No minute surges found.")
    elif strategy == "🏈 NFL Stacking":
        if 'team' in df.columns:
            st.write("NFL Stacking logic active...")

# =========================================================
# 🏗️ TAB 3: LINEUP BUILDER
# =========================================================
elif app_mode == "🏗️ Lineup Builder":
    st.title("🏗️ Automated Lineup Builder")
    df = st.session_state['master_df']
    salary_cap = st.number_input("Salary Cap", 50000)
    roster_size = st.number_input("Roster Size", 8)
    if st.button("Generate Lineup"):
        if 'proj_pts' in df.columns and 'salary' in df.columns:
            res = optimize_lineup(df, salary_cap, roster_size)
            if res is not None:
                st.success(f"Optimal Lineup ({res['proj_pts'].sum():.1f} Proj Pts)")
                st.dataframe(res)
            else:
                st.error("No valid lineup found.")

# =========================================================
# 💸 TAB 4: BETTING EDGE CALCULATOR (NEW!)
# =========================================================
elif app_mode == "💸 Betting Edge Calculator":
    st.title("💸 Unlimited Betting Edge Calculator")
    st.markdown("Generates **Master Pools** of all qualifying props based on your new logic.")
    
    df = st.session_state['master_df'].copy()
    
    if 'prop_line' in df.columns and 'proj_pts' in df.columns:
        # Calculate Edge
        # Edge formula: (Projection - Line) / Line
        df['edge_pct'] = ((df['proj_pts'] - df['prop_line']) / df['prop_line']) * 100
        df['pick_type'] = np.where(df['edge_pct'] > 0, 'OVER', 'UNDER')
        
        # Ensure confidence column exists (default to 0.5 if missing)
        if 'confidence' not in df.columns:
            df['confidence'] = 0.5
            
        # --- LOGIC ENGINE ---
        
        # 1. UNDERDOG GENERATION (Edge > -0.5%)
        # "Finds ALL unique players with qualifying edges > -0.5"
        ud_pool = df[df['edge_pct'] > -0.5].sort_values(by='edge_pct', ascending=False)
        
        # 2. PICK6 GENERATION (Deterministic + Diversity)
        # Deterministic: Edge > -0.2
        # Diversity: |Edge| > 0.3 AND Confidence > 0.6
        p6_deterministic = df[df['edge_pct'] > -0.2]
        p6_diversity = df[(df['edge_pct'].abs() > 0.3) & (df['confidence'] > 0.6)]
        p6_pool = pd.concat([p6_deterministic, p6_diversity]).drop_duplicates().sort_values(by='edge_pct', ascending=False)
        
        # 3. CONFIDENCE SLIPS (Edge > -0.3%)
        conf_pool = df[df['edge_pct'] > -0.3].sort_values(by='confidence', ascending=False)
        
        # --- DISPLAY RESULTS ---
        
        tab1, tab2, tab3 = st.tabs(["🐶 Underdog Master Pool", "🎯 Pick6 Master Pool", "🔒 Confidence Pool"])
        
        with tab1:
            st.subheader(f"Underdog Pool ({len(ud_pool)} Plays)")
            st.write("Criteria: Edge > -0.5% (Max Volume)")
            if not ud_pool.empty:
                st.dataframe(ud_pool[['name', 'team', 'prop_line', 'proj_pts', 'pick_type', 'edge_pct']].style.background_gradient(subset=['edge_pct'], cmap='Greens'), use_container_width=True)
            else:
                st.warning("No plays qualify for Underdog criteria.")
                
        with tab2:
            st.subheader(f"Pick6 Pool ({len(p6_pool)} Plays)")
            st.write("Criteria: Edge > -0.2% OR (Edge > 0.3% & Conf > 60%)")
            if not p6_pool.empty:
                st.dataframe(p6_pool[['name', 'team', 'prop_line', 'proj_pts', 'pick_type', 'edge_pct', 'confidence']].style.background_gradient(subset=['confidence'], cmap='Blues'), use_container_width=True)
            else:
                st.warning("No plays qualify for Pick6 criteria.")
                
        with tab3:
            st.subheader(f"Confidence Pool ({len(conf_pool)} Plays)")
            st.write("Criteria: Edge > -0.3% (Sorted by Confidence)")
            if not conf_pool.empty:
                st.dataframe(conf_pool[['name', 'team', 'pick_type', 'edge_pct', 'confidence']].style.format({'confidence': '{:.0%}'}), use_container_width=True)
            else:
                st.warning("No plays qualify for Confidence criteria.")

    else:
        st.error("⚠️ Data Missing: Your CSV must have `prop_line` and `proj_pts` columns to calculate edges.")
        with st.expander("See Example CSV Structure"):
            st.code("name,team,proj_pts,prop_line,confidence\nLeBron James,LAL,28.5,26.5,0.75", language="csv")
