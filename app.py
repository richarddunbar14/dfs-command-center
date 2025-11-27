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
    [data-testid="stDataFrame"] { border: 1px solid #334155; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- SMART DATA LOADER ---
def standardize_columns(df):
    """
    Renames weird CSV columns to the standard names the app expects.
    """
    # Clean up column names (lowercase, no spaces)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('-', '_')
    
    # Mapping Dictionary: { Standard_Name : [Possible Variations] }
    column_map = {
        'name': ['player', 'athlete', 'full_name', 'name_id'],
        'proj_pts': ['projection', 'proj', 'fpts', 'fantasy_points', 'pts_proj', 'ppg', 'avg_fpts'],
        'prop_line': ['line', 'prop', 'ou', 'total', 'over_under', 'strike', 'pick_line'],
        'salary': ['cost', 'sal', 'price', 'salary_cap'],
        'team': ['squad', 'tm', 'team_id'],
        'position': ['pos', 'position_id'],
        'proj_mins': ['min', 'minutes', 'proj_min'],
        'avg_mins': ['avg_min', 'season_min']
    }
    
    renamed_cols = {}
    for standard, variations in column_map.items():
        if standard not in df.columns:
            for v in variations:
                # Look for columns that contain the variation keyword
                match = next((c for c in df.columns if v in c), None)
                if match:
                    renamed_cols[match] = standard
                    break
    
    if renamed_cols:
        df = df.rename(columns=renamed_cols)
        
    # Force numbers to be numbers (not text)
    cols_to_numeric = ['proj_pts', 'prop_line', 'salary', 'proj_mins', 'avg_mins']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    return df

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Apply the Smart Fix
        df = standardize_columns(df)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def optimize_lineup(df, salary_cap, roster_size):
    # Filter out players with 0 projection or 0 salary to prevent errors
    df = df[(df['proj_pts'] > 0) & (df['salary'] > 0)]
    
    problem = pulp.LpProblem("DFS_Optimization", pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("Player", df.index, cat='Binary')
    
    problem += pulp.lpSum([df.loc[i, 'proj_pts'] * player_vars[i] for i in df.index])
    problem += pulp.lpSum([df.loc[i, 'salary'] * player_vars[i] for i in df.index]) <= salary_cap
    problem += pulp.lpSum([player_vars[i] for i in df.index]) == roster_size
    
    solve_status = problem.solve(pulp.PULP_CBC_CMD(msg=0))
    if solve_status == pulp.LpStatusOptimal:
        selected = [i for i in df.index if player_vars[i].varValue == 1]
        return df.loc[selected]
    return None

# --- SIDEBAR ---
st.sidebar.title("🧠 Sports Brain")
app_mode = st.sidebar.radio("Navigation", 
    ["📂 Data Upload", "🔬 Strategy Lab", "🏗️ Lineup Builder", "💸 Betting Edge Calculator"])

if 'master_df' not in st.session_state:
    st.session_state['master_df'] = pd.DataFrame()

# =========================================================
# 📂 TAB 1: DATA UPLOAD
# =========================================================
if app_mode == "📂 Data Upload":
    st.title("📂 Upload Your Data")
    uploaded_file = st.file_uploader("Drag and drop CSV here", type=['csv', 'xlsx'])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state['master_df'] = df
            st.success(f"✅ Successfully ingested {len(df)} players!")
            
            # Diagnostic Panel
            st.write("### 🔍 Smart Column Detection:")
            cols = list(df.columns)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Name", "✅" if 'name' in cols else "❌")
            c2.metric("Projection", "✅" if 'proj_pts' in cols else "❌")
            c3.metric("Prop Line", "✅" if 'prop_line' in cols else "❌")
            c4.metric("Salary", "✅" if 'salary' in cols else "❌")
            
            with st.expander("See Raw Data"):
                st.dataframe(df.head())

# =========================================================
# 🔬 TAB 2: STRATEGY LAB
# =========================================================
elif app_mode == "🔬 Strategy Lab":
    st.title("🔬 Strategy Lab")
    df = st.session_state['master_df']
    
    if df.empty:
        st.warning("⚠️ Upload data first!")
    else:
        strategy = st.radio("Select Strategy:", ["⏰ Minutes > Talent", "🏈 NFL Stacking"])
        
        if strategy == "⏰ Minutes > Talent":
            if 'proj_mins' in df.columns and 'avg_mins' in df.columns:
                df['minute_diff'] = df['proj_mins'] - df['avg_mins']
                df['value'] = (df['proj_pts'] / df['salary']) * 1000
                opps = df[(df['minute_diff'] >= 5.0) & (df['salary'] <= 6000)].sort_values(by='value', ascending=False)
                st.dataframe(opps[['name', 'team', 'salary', 'minute_diff', 'value']])
            else:
                st.error("Missing 'minutes' data. Check your CSV columns.")

        elif strategy == "🏈 NFL Stacking":
            if 'team' in df.columns and 'proj_pts' in df.columns:
                st.write("### Top Team Stacks")
                # Group by team and sum points
                stacks = df.groupby('team')['proj_pts'].sum().reset_index().sort_values(by='proj_pts', ascending=False)
                st.dataframe(stacks.head(10))
            else:
                st.error("Missing 'team' or 'projection' data.")

# =========================================================
# 🏗️ TAB 3: LINEUP BUILDER
# =========================================================
elif app_mode == "🏗️ Lineup Builder":
    st.title("🏗️ Automated Lineup Builder")
    df = st.session_state['master_df']
    
    if df.empty or 'salary' not in df.columns:
        st.warning("⚠️ Need data with 'Salary' column.")
    else:
        c1, c2 = st.columns(2)
        salary_cap = c1.number_input("Salary Cap", 50000)
        roster_size = c2.number_input("Roster Size", 8)
        
        if st.button("Generate Optimal Lineup"):
            res = optimize_lineup(df, salary_cap, roster_size)
            if res is not None:
                st.balloons()
                st.success(f"🏆 Proj Score: {res['proj_pts'].sum():.1f}")
                st.dataframe(res[['name', 'team', 'salary', 'proj_pts']])
            else:
                st.error("Infeasible. Try increasing cap or changing roster size.")

# =========================================================
# 💸 TAB 4: BETTING EDGE CALCULATOR
# =========================================================
elif app_mode == "💸 Betting Edge Calculator":
    st.title("💸 Unlimited Betting Edge Calculator")
    df = st.session_state['master_df']
    
    if df.empty: 
        st.warning("Upload data first.")
    elif 'prop_line' in df.columns and 'proj_pts' in df.columns:
        
        # Calc Edge
        df['edge_pct'] = ((df['proj_pts'] - df['prop_line']) / df['prop_line']) * 100
        df['pick_type'] = np.where(df['edge_pct'] > 0, 'OVER', 'UNDER')
        
        # Filter Logic
        ud_pool = df[df['edge_pct'] > -0.5].sort_values(by='edge_pct', ascending=False)
        
        st.subheader(f"✅ Found {len(ud_pool)} Qualifying Props")
        
        # Use simple dataframe first to ensure no styling errors
        st.dataframe(
            ud_pool[['name', 'team', 'prop_line', 'proj_pts', 'pick_type', 'edge_pct']]
            .style.background_gradient(subset=['edge_pct'], cmap='Greens'),
            use_container_width=True
        )
    else:
        st.error("⚠️ Data Missing Projections or Lines")
        st.write("Columns found:", list(df.columns))
