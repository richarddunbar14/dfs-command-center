import streamlit as st
import pandas as pd
import numpy as np
import pulp
import base64
from duckduckgo_search import DDGS
from sklearn.preprocessing import MinMaxScaler
import math

# ==========================================
# ⚙️ 1. TITAN CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="TITAN COMMAND: FINAL FUSION", page_icon="💥")

# --- UI STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #e2e8f0; font-family: 'Roboto', sans-serif; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { color: #38bdf8; font-size: 24px; font-weight: 800; }
    
    /* Tables */
    [data-testid="stDataFrame"] { border: 1px solid #334155; border-radius: 5px; }
    
    /* Buttons */
    .stButton>button { width: 100%; border-radius: 6px; font-weight: bold; background-color: #3b82f6; border: none; }
    .stButton>button:hover { background-color: #2563eb; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 2. THE TITAN BRAIN (LOGIC CLASS)
# ==========================================

class TitanBrain:
    def __init__(self, sport, spread=0, total=0):
        self.sport = sport
        self.spread = spread
        self.total = total
        
    def evaluate_player(self, row):
        reasons = []
        score = 50.0 
        
        # 1. LEVERAGE LOGIC
        if 'rank_proj' in row and 'rank_own' in row and row['rank_proj'] > 0 and row['rank_own'] > 0:
            leverage_diff = row['rank_own'] - row['rank_proj']
            if leverage_diff > 15:
                score += 15
                reasons.append(f"💎 High Leverage (Underrated by {leverage_diff} spots)")
            elif leverage_diff < -10:
                score -= 10
                reasons.append("⚠️ Chalk Trap (Overowned)")
                
        # 2. GAME SCRIPT LOGIC (NFL Focused)
        if self.sport == "NFL" and abs(self.spread) > 7:
            if 'WR' in str(row.get('position', '')):
                score += 7
                reasons.append("📜 Blowout Script: Garbage Time potential")

        # 3. PROP VALUE LOGIC
        if 'prop_line' in row and 'proj_pts' in row and row['prop_line'] > 0:
            edge = ((row['proj_pts'] - row['prop_line']) / row['prop_line']) * 100
            if edge > 15:
                score += 15
                reasons.append(f"💰 Prop Smash ({edge:.1f}% Edge)")

        # 4. UPSIDE/USAGE METRICS
        if 'ceiling' in row and row['ceiling'] > row['proj_pts'] * 1.5:
            score += 10
            reasons.append("🔥 Massive Ceiling Differential")
            
        final_score = max(0, min(100, score))
        verdict_text = "✅ PLAY: " + " | ".join(reasons) if reasons else "ℹ️ NEUTRAL: Balanced profile."
        return final_score, verdict_text

# ==========================================
# 🛠️ 4. DATA PROCESSING & UTILITIES
# ==========================================

def standardize_columns(df):
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('-', '_').str.replace('$', '').str.replace(',', '').str.replace('%', '')
    column_map = {
        'name': ['player', 'athlete', 'full_name'], 'proj_pts': ['projection', 'proj', 'fpts', 'median'],
        'ownership': ['own', 'own', 'projected_ownership'], 'salary': ['cost', 'sal', 'price'],
        'prop_line': ['line', 'prop', 'ou', 'total', 'strike'], 'position': ['pos', 'position_id'],
        'team': ['squad', 'tm'], 'air_yards': ['air_yards', 'ay']
    }
    renamed = {}
    for std, alts in column_map.items():
        for col in df.columns:
            if col not in renamed.values():
                if any(a in col for a in alts): renamed[col] = std
    df = df.rename(columns=renamed)
    
    # Numeric Cleanup
    for c in ['proj_pts', 'salary', 'ownership', 'prop_line', 'ceiling']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    if 'position' in df.columns:
        df['position'] = df['position'].astype(str).str.upper().str.replace('/', ',')
        
    return df

def process_and_analyze(files, sport, spread, total):
    df = pd.DataFrame()
    if files:
        df = pd.DataFrame()
        for f in files:
            try:
                if f.name.endswith('.csv'): temp_df = pd.read_csv(f)
                else: temp_df = pd.read_excel(f)
                temp_df = standardize_columns(temp_df)
                if df.empty: df = temp_df
                else:
                    cols_to_merge = [c for c in temp_df.columns if c not in df.columns or c == 'name']
                    if 'name' in temp_df.columns: 
                        df = df.merge(temp_df[cols_to_merge], on='name', how='left', suffixes=('', '_new'))
                        for col in [c for c in df.columns if col.endswith('_new')]:
                            df[col.replace('_new', '')] = df[col].fillna(df[col.replace('_new', '')])
                            df = df.drop(columns=[col])
            except Exception as e: st.error(f"Error loading file: {e}")
                
        # --- TITAN BRAIN EXECUTION ---
        if 'proj_pts' in df.columns and 'ownership' in df.columns:
            df['rank_proj'] = df['proj_pts'].rank(ascending=False)
            df['rank_own'] = df['ownership'].rank(ascending=False)
            
            brain = TitanBrain(sport, spread, total)
            results = df.apply(lambda row: brain.evaluate_player(row), axis=1, result_type='expand')
            df['shark_score'] = results[0]
            df['reasoning'] = results[1]
        
    return df

def get_player_pool(df, top_n_shark=25, top_n_value=15):
    """Generates the final player pool."""
    if df.empty or 'shark_score' not in df.columns: return pd.DataFrame()

    # 1. Primary Filter: Best Overall Players
    pool_shark = df.nlargest(top_n_shark, 'shark_score')
    
    # 2. High Value Punts (High Risk/Reward)
    if 'value' in df.columns:
        pool_value = df[df['salary'] < 5500].nlargest(top_n_value, 'value')
    else:
        pool_value = pd.DataFrame()

    final_pool = pd.concat([pool_shark, pool_value]).drop_duplicates(subset=['name'])
    return final_pool.sort_values(by='shark_score', ascending=False)

# ==========================================
# 🏗️ 5. OPTIMIZER & EXPORT
# ==========================================

def optimize_lineup(df, config):
    # This function remains the core solver with explicit constraints
    site, sport, cap, num_lineups, target_col, use_correlation = (
        config['site'], config['sport'], config['cap'], config['num_lineups'], config['target_col'], config['use_correlation']
    )
    
    pool = df[(df[target_col] > 0) & (df['salary'] > 0)].reset_index(drop=True)
    roster_size = 9 if sport=="NFL" else 8
    
    valid_lineups = []
    
    for _ in range(num_lineups):
        prob = pulp.LpProblem("Apex_Optimizer", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("P", pool.index, cat='Binary')
        
        prob += pulp.lpSum([pool.loc[i, target_col] * x[i] for i in pool.index]) # Objective
        prob += pulp.lpSum([pool.loc[i, 'salary'] * x[i] for i in pool.index]) <= cap # Salary Cap
        prob += pulp.lpSum([x[i] for i in pool.index]) == roster_size # Roster Size
        
        # Position Logic (NFL Example)
        if sport == "NFL":
            qbs = pool[pool['position'].str.contains('QB', na=False)]
            prob += pulp.lpSum([x[i] for i in qbs.index]) == 1
            if use_correlation:
                # Force QB Stacking
                for qb_idx in qbs.index:
                    team = pool.loc[qb_idx, 'team']
                    stack_partners = pool[(pool['team'] == team) & (pool['position'].str.contains('WR|TE', na=False))]
                    prob += pulp.lpSum([x[i] for i in stack_partners.index]) >= x[qb_idx]
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if prob.status == pulp.LpStatusOptimal:
            sel = [i for i in pool.index if x[i].varValue == 1]
            lineup = pool.loc[sel].copy()
            lineup['Lineup_ID'] = _ + 1
            valid_lineups.append(lineup)
            prob += pulp.lpSum([x[i] for i in sel]) <= len(sel) - 1
        else: break
            
    return pd.concat(valid_lineups) if valid_lineups else None

def kelly_criterion(bankroll, odds_decimal, win_probability, multiplier=1.0):
    """Calculates optimal fraction of bankroll to wager."""
    b = odds_decimal - 1 # Net odds
    f = (b * win_probability - (1 - win_probability)) / b
    return max(0, f * multiplier)

# ==========================================
# 🖥️ 6. UI DASHBOARD
# ==========================================

st.sidebar.title("🦁 TITAN APEX CORE")
sport = st.sidebar.selectbox("Sport", ["NFL", "NBA", "MLB", "NHL"], index=0)
site = st.sidebar.selectbox("Sportsbook", ["DraftKings", "FanDuel", "Yahoo", "PrizePicks", "Underdog"])
st.sidebar.markdown("---")

# AUTO-CAP FOR DISPLAY
default_cap = 50000
if site == "FanDuel": default_cap = 60000
elif site == "Yahoo": default_cap = 200

# STATE MANAGEMENT
if 'master' not in st.session_state: st.session_state['master'] = pd.DataFrame()
if 'cap' not in st.session_state: st.session_state['cap'] = default_cap

tabs = st.tabs(["1. 📂 Data Ingest", "2. 🎯 Player Pool", "3. 🏗️ Optimizer", "4. 💸 Prop Sniper"])

# --- TAB 1: DATA INGEST ---
with tabs[0]:
    st.title("1. 📂 Data & Logic Ingest")
    st.info("Upload Projections, Ownership, and Prop files. Everything will be merged and cleaned.")
    
    files = st.file_uploader("Drop Files", accept_multiple_files=True)
    
    st.markdown("### Game Theory Input")
    c1, c2 = st.columns(2)
    spread = c1.number_input("Vegas Spread (Avg)", -20.0, 20.0, 0.0)
    total = c2.number_input("Vegas Total (Avg)", 150, 250, 210)
    
    if st.button("🚀 Process All Data"):
        with st.spinner("Calculating Shark Scores..."):
            df = process_and_analyze(files, sport, spread, total)
            st.session_state['master'] = df
            st.success("✅ Titan Brain Analysis Complete.")

# --- TAB 2: PLAYER POOL & UPSIDE ---
with tabs[1]:
    st.title("2. 🎯 Player Pool & High Upside")
    df = st.session_state['master']
    
    if df.empty: st.warning("Upload data first.")
    else:
        # Get Pool based on analysis
        pool_df = get_player_pool(df)
        
        st.subheader("Final Player Pool Roster")
        st.info("This list contains players with high Shark Scores, high ceiling, and good value.")
        
        # Player Position Filter
        if 'position' in pool_df.columns:
            all_pos = ['ALL'] + sorted(list(pool_df['position'].unique()))
            pos_filter = st.selectbox("Filter by Position", all_pos)
            if pos_filter != 'ALL':
                pool_df = pool_df[pool_df['position'] == pos_filter]

        st.dataframe(
            pool_df[['name', 'position', 'salary', 'proj_pts', 'shark_score', 'reasoning']].head(40),
            use_container_width=True
        )

        st.subheader("Reasoning & Upside Breakdown")
        for i, row in pool_df.head(3).iterrows():
            st.markdown(f"""
            <div style="padding: 10px; border-bottom: 1px solid #1e293b;">
                <span style="color: #38bdf8; font-weight: bold;">{row['name']} ({row['position']})</span>
                <p class="reasoning-text">
                    **Verdict:** {row['reasoning']}
                </p>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 3: OPTIMIZER ---
with tabs[2]:
    st.title(f"3. 🏗️ {site} Lineup Builder")
    df = st.session_state['master']
    
    if df.empty: st.warning("Upload data first.")
    else:
        c1, c2, c3 = st.columns(3)
        num = c1.number_input("Lineups (MME Max 150)", 1, 150, 10)
        cap = c2.number_input("Salary Cap", value=st.session_state['cap'])
        target = c3.selectbox("Optimization Goal", ["Shark Score (GPP)", "Projected Points (Cash)"])
        
        corr = st.checkbox("✅ Auto-Correlation (Stacking/Pairing)", value=True, help="Forces QB+WR stacks in NFL, etc.")
        
        if st.button("⚡ BUILD SLATE-BREAKING LINEUPS"):
            pool_df = get_player_pool(df) # Ensure only smart players are used
            
            config = {'site': site, 'sport': sport, 'cap': cap, 'num_lineups': num, 'target_col': 'shark_score' if target == "Shark Score (GPP)" else 'proj_pts', 'use_correlation': corr}
            
            with st.spinner(f"Optimizing {num} lineups..."):
                res = optimize_lineup(pool_df, config)
            
            if res is not None:
                st.success(f"Generated {num} Lineups!")
                st.metric("Avg Lineup Score", f"{res.groupby('Lineup_ID')['proj_pts'].sum().mean():.2f}")
                st.markdown(get_csv_download(res), unsafe_allow_html=True)
            else: st.error("Optimization Failed. Check if pool size is too small.")

# --- TAB 4: PROP SNIPER ---
with tabs[3]:
    st.title("4. 💸 Prop Sniper & Bankroll")
    df = st.session_state['master']
    
    if 'prop_line' not in df.columns:
        st.warning("No Prop Lines found in data.")
    else:
        st.subheader("Prop Edge Analysis")
        df['edge'] = ((df['proj_pts'] - df['prop_line']) / df['prop_line']) * 100
        df['pick'] = np.where(df['edge']>0, 'OVER', 'UNDER')
        
        st.dataframe(
            df.sort_values('edge', ascending=False)[['name', 'prop_line', 'proj_pts', 'pick', 'edge']],
            use_container_width=True
        )
        
        st.markdown("---")
        st.subheader("💰 Bankroll Management (Kelly Criterion)")
        st.info("Use the recommended percentage to maximize growth while managing risk.")
        
        c1, c2, c3 = st.columns(3)
        bankroll = c1.number_input("Total Bankroll ($)", value=1000)
        prob_win = c2.number_input("Est. Win Probability (e.g., 0.55)", 0.0, 1.0, 0.55, step=0.01)
        odds_decimal = c3.number_input("Odds (Decimal, e.g., 2.0)", 1.0, 10.0, 2.0)
        
        kelly_frac = kelly_criterion(bankroll, odds_decimal, prob_win, multiplier=0.5)
        
        st.metric("Recommended Wager (50% Kelly)", f"${bankroll * kelly_frac:.2f}", f"{kelly_frac:.2%} of Bankroll")
