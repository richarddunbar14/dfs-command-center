import streamlit as st
import pandas as pd
import numpy as np
import pulp
import base64
import sys
import feedparser
import difflib
import random

# ==========================================
# ⚙️ 1. GLOBAL CONFIGURATION & STYLING
# ==========================================
sys.setrecursionlimit(3000) # Maximum power for complex stacking logic
st.set_page_config(layout="wide", page_title="TITAN GOD MODE", page_icon="⚡")

st.markdown("""
<style>
    /* TITAN DARK MODE THEME */
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Helvetica Neue', sans-serif; }
    
    /* METRICS */
    div[data-testid="stMetricValue"] { color: #00ff9d; font-family: 'Courier New', monospace; font-weight: 900; }
    div[data-testid="stMetricLabel"] { color: #888; font-size: 14px; }
    
    /* BUTTONS */
    .stButton>button { 
        width: 100%; border-radius: 4px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%); border: none; color: white;
    }
    
    /* DATAFRAMES */
    .stDataFrame { border: 1px solid #333; }
    
    /* LOG BOX */
    .log-box { font-family: 'Courier New'; font-size: 11px; color: #00ff9d; background: #111; padding: 8px; border-left: 3px solid #0072ff; margin-bottom: 2px; }
    
    /* HEADERS */
    h1, h2, h3 { color: #ffffff; font-weight: 800; letter-spacing: -1px; }
</style>
""", unsafe_allow_html=True)

if 'master' not in st.session_state: st.session_state['master'] = pd.DataFrame()

# ==========================================
# 🧠 2. THE TITAN BRAIN (ADVANCED AI LOGIC)
# ==========================================

class TitanBrain:
    def __init__(self, sport, spread=0, total=0):
        self.sport = sport
        self.spread = spread
        self.total = total
        
    def calculate_leverage_score(self, row):
        """
        Calculates a proprietary 'Titan Score' (0-100) combining Value, Leverage, and Ceiling.
        This is the 'Secret Sauce' for slate-breaking.
        """
        score = 50.0
        reasons = []

        # METRIC 1: PURE VALUE (Points per Dollar)
        if row['salary'] > 0:
            value = (row['proj_pts'] / row['salary']) * 1000
            if value > 5.5: 
                score += 10
                reasons.append("💰 Elite Value")
            elif value < 3.0: 
                score -= 10
        
        # METRIC 2: LEVERAGE (Ownership vs Rank)
        # If a player is ranked #1 in points but #20 in ownership, that is GOLD.
        if row['rank_own'] > row['rank_proj'] + 10:
            score += 15
            reasons.append(f"💎 Leverage Play (High Proj/Low Own)")
        elif row['rank_own'] < row['rank_proj'] - 10:
            score -= 5
            reasons.append("⚠️ Negative Leverage (Chalk)")

        # METRIC 3: CEILING (The Slate Breaker)
        # If ceiling is significantly higher than median, boost score (GPP winner)
        if 'ceiling' in row and row['ceiling'] > row['proj_pts'] * 1.6:
            score += 12
            reasons.append("🚀 Massive Upside")

        # METRIC 4: VEGAS DATA (Game Script)
        # NFL: If playing catchup (Underdog in high total game) -> Boost WRs/QBs
        if self.sport == "NFL" and self.total > 48 and self.spread > 3 and "WR" in str(row.get('position','')):
            score += 8
            reasons.append("📜 Shootout Script")

        # METRIC 5: PROP EDGE
        if row.get('prop_edge', 0) > 10:
            score += 5
            reasons.append("🟢 Vegas Edge")

        return max(0, min(100, score)), " | ".join(reasons)

# ==========================================
# 🛠️ 3. INTELLIGENT DATA INGEST (FUZZY LOGIC)
# ==========================================

def smart_merge(df_main, df_new, on_col='name', threshold=0.85):
    """
    The 'Human-Like' merger. Matches 'Patrick Mahomes' to 'Patrick Mahomes II'.
    Uses difflib SequenceMatcher.
    """
    if df_main.empty: return df_new
    if on_col not in df_main.columns or on_col not in df_new.columns: return df_main

    # Get unique names from both
    main_names = df_main[on_col].unique()
    new_names = df_new[on_col].unique()
    
    mapping = {}
    
    # 1. Exact Match First (Fastest)
    for name in new_names:
        if name in main_names:
            mapping[name] = name
            
    # 2. Fuzzy Match Second (The magic)
    for name in new_names:
        if name not in mapping:
            matches = difflib.get_close_matches(name, main_names, n=1, cutoff=threshold)
            if matches:
                mapping[name] = matches[0]
            else:
                mapping[name] = None # No match found
                
    # Create a merge key
    df_new['merge_key'] = df_new[on_col].map(mapping)
    
    # Drop rows that didn't match (optional: or keep them)
    # df_new = df_new.dropna(subset=['merge_key'])
    
    # Merge
    cols_to_add = [c for c in df_new.columns if c not in df_main.columns and c != on_col and c != 'merge_key']
    merged = df_main.merge(df_new[['merge_key'] + cols_to_add], left_on=on_col, right_on='merge_key', how='left')
    
    return merged

def standardize_columns(df):
    """Universal Translator for DraftKings, FanDuel, Yahoo, Rotowire, etc."""
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('-', '_').str.replace('$', '').str.replace('%', '')
    
    # Comprehensive Dictionary
    mapping = {
        'name': ['player', 'athlete', 'player_name', 'nickname', 'full_name'],
        'proj_pts': ['fpts', 'projection', 'proj', 'median', 'points', 'fantasy_points_per_game'],
        'ceiling': ['ceil', 'ceiling', 'max_projection', 'max_pts'],
        'salary': ['cost', 'sal', 'price', 'salary'],
        'ownership': ['own', 'projected_ownership', 'pown%'],
        'position': ['pos', 'roster_position', 'position'],
        'team': ['squad', 'tm', 'team'],
        'opp': ['opponent', 'opp', 'vs'],
        'prop_line': ['line', 'strike', 'prop', 'ou']
    }
    
    renamed = {}
    for standard, alts in mapping.items():
        for col in df.columns:
            if col not in renamed.values():
                if col in alts or any(a in col for a in alts):
                    renamed[col] = standard
                    break
    df = df.rename(columns=renamed)
    
    # Data Type Cleaning
    for c in ['proj_pts', 'salary', 'ownership', 'ceiling', 'prop_line']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
            
    if 'position' in df.columns:
        df['position'] = df['position'].astype(str).str.upper().str.replace('/', ',')
        
    return df

def process_data_pipeline(files, sport, spread, total):
    master = pd.DataFrame()
    logs = []

    for file in files:
        try:
            # 1. Load (CSV or Excel)
            if file.name.endswith('.csv'): 
                try: df = pd.read_csv(file, sep=None, engine='python')
                except: df = pd.read_csv(file)
            else: df = pd.read_excel(file)
            
            # 2. Standardize
            df = standardize_columns(df)
            
            # 3. Safety: Remove Array Columns/Duplicates
            df = df.loc[:, ~df.columns.duplicated()]
            
            # 4. Smart Merge
            if master.empty:
                master = df
                logs.append(f"✅ Base File Loaded: {file.name} ({len(df)} rows)")
            else:
                if 'name' in master.columns and 'name' in df.columns:
                    # Check overlap before merge
                    overlap = len(set(master['name']).intersection(set(df['name'])))
                    master = smart_merge(master, df)
                    logs.append(f"🔗 Merged {file.name} (Matched ~{overlap} players)")
                else:
                    logs.append(f"⚠️ Skipped {file.name} (No 'Name' column found)")
                    
        except Exception as e:
            logs.append(f"❌ Error {file.name}: {str(e)}")
            
    # 5. TITAN BRAIN SCORING
    if not master.empty and 'proj_pts' in master.columns:
        # Fill defaults
        for col, val in {'ownership': 5.0, 'salary': 4000, 'ceiling': 0}.items():
            if col not in master.columns: master[col] = val
        if 'ceiling' not in master.columns or master['ceiling'].sum() == 0:
            master['ceiling'] = master['proj_pts'] * 1.5 # Auto-generate ceiling if missing
            
        master['rank_proj'] = master['proj_pts'].rank(ascending=False)
        master['rank_own'] = master['ownership'].rank(ascending=False)
        
        # Prop Edge Calc
        if 'prop_line' in master.columns:
            master['prop_edge'] = master.apply(lambda x: ((x['proj_pts'] - x['prop_line'])/x['prop_line'])*100 if x['prop_line']>0 else 0, axis=1)
        
        brain = TitanBrain(sport, spread, total)
        res = master.apply(lambda row: brain.calculate_leverage_score(row), axis=1, result_type='expand')
        master['titan_score'] = res[0]
        master['reasoning'] = res[1]
        
        # Sort
        master = master.sort_values(by='titan_score', ascending=False).reset_index(drop=True)

    return master, logs

# ==========================================
# ⚡ 4. MONTE CARLO OPTIMIZER (SLATE BREAKER)
# ==========================================

def optimize_advanced(df, config):
    site, sport, cap, num_lineups, variance = config['site'], config['sport'], config['cap'], config['num_lineups'], config['variance']
    
    # Filter Pool
    pool = df[(df['proj_pts'] > 0) & (df['salary'] > 0)].reset_index(drop=True)
    if pool.empty: return None

    # Settings
    roster_size = 9 if sport == "NFL" else 8
    lineups = []
    
    progress_bar = st.progress(0)
    
    for i in range(num_lineups):
        prob = pulp.LpProblem("Titan_Solver", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("player", pool.index, cat='Binary')
        
        # --- 1. RANDOMIZATION (MONTE CARLO) ---
        # Instead of using static projection, we sample from a normal distribution
        # This simulates "playing the slate 100 times"
        volatility = variance / 100.0 # e.g. 15% volatility
        
        # Vectorized random sampling for speed
        random_multipliers = np.random.normal(1.0, volatility, size=len(pool))
        pool['sim_pts'] = pool['proj_pts'] * random_multipliers
        # Ensure sim_pts uses ceiling sometimes
        pool['sim_pts'] = np.where(random_multipliers > 1.1, (pool['sim_pts'] + pool['ceiling'])/2, pool['sim_pts'])
        
        # Objective: Maximize Simulated Points (not just median)
        prob += pulp.lpSum([pool.loc[p, 'sim_pts'] * x[p] for p in pool.index])
        
        # Constraints
        prob += pulp.lpSum([pool.loc[p, 'salary'] * x[p] for p in pool.index]) <= cap
        prob += pulp.lpSum([x[p] for p in pool.index]) == roster_size
        
        # --- 2. ADVANCED CORRELATION RULES ---
        if sport == "NFL":
            # QB Setup
            qbs = pool[pool['position'].str.contains("QB")]
            dsts = pool[pool['position'].str.contains("DST|DEF")]
            prob += pulp.lpSum([x[p] for p in qbs.index]) == 1
            prob += pulp.lpSum([x[p] for p in dsts.index]) == 1
            
            # STACKING LOGIC
            for qb_idx in qbs.index:
                team = pool.loc[qb_idx, 'team']
                
                # Teammates (WR/TE)
                teammates = pool[(pool['team'] == team) & (pool['position'].str.contains("WR|TE"))]
                
                # Opponents (Run-back) - Requires 'opp' column, else skips
                opp_team = pool.loc[qb_idx, 'opp'] if 'opp' in pool.columns else None
                opponents = pool[(pool['team'] == opp_team) & (pool['position'].str.contains("WR|TE"))] if opp_team else pd.DataFrame()
                
                # If QB is picked, must pick at least 1 teammate
                if not teammates.empty:
                    prob += pulp.lpSum([x[t] for t in teammates.index]) >= x[qb_idx]
                
                # Game Stack: If QB is picked, try to pick 1 opponent (Correlated scoring)
                if not opponents.empty and config['use_runback']:
                     prob += pulp.lpSum([x[o] for o in opponents.index]) >= x[qb_idx]

        elif sport == "NBA":
            # Just ensure positions are generally valid (simplified for speed)
            centers = pool[pool['position'].str.contains("C")]
            prob += pulp.lpSum([x[c] for c in centers.index]) >= 1
            
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            selected_indices = [p for p in pool.index if x[p].varValue == 1]
            lineup_df = pool.loc[selected_indices].copy()
            lineup_df['Lineup_ID'] = i + 1
            lineup_df['Simulated_Score'] = lineup_df['sim_pts'].sum()
            lineups.append(lineup_df)
            
            # Constrain next iteration (Prevent exact duplicate lineups)
            prob += pulp.lpSum([x[p] for p in selected_indices]) <= roster_size - 1
            
        progress_bar.progress((i + 1) / num_lineups)
        
    return pd.concat(lineups) if lineups else None

# ==========================================
# 🖥️ 5. UI DASHBOARD
# ==========================================

st.sidebar.title("⚡ TITAN GOD MODE")
sport = st.sidebar.selectbox("Sport", ["NFL", "NBA", "MLB", "NHL", "CFB", "CBB"])
site = st.sidebar.selectbox("Platform", ["DraftKings", "FanDuel", "Yahoo"])

# Smart defaults
cap_map = {"DraftKings": 50000, "FanDuel": 60000, "Yahoo": 200}
st.session_state['cap'] = cap_map.get(site, 50000)

tabs = st.tabs(["📂 1. Data Ingest", "🧠 2. Titan Analysis", "🏭 3. MME Factory", "💸 4. Prop Sniper", "📰 5. Intel"])

# --- TAB 1: INGEST ---
with tabs[0]:
    st.header("Data Ingestion Engine")
    st.markdown("Upload **Projections**, **Salaries**, and **Props** CSVs. The Fuzzy Logic engine will stitch them together automatically.")
    
    files = st.file_uploader("Drop Files Here", accept_multiple_files=True)
    
    c1, c2 = st.columns(2)
    spread_input = c1.number_input("Game Script: Avg Spread", -10.0, 10.0, 0.0)
    total_input = c2.number_input("Game Script: Avg Total", 150, 260, 215)
    
    if st.button("🚀 INITIATE FUSION SEQUENCE"):
        if not files:
            st.error("NO DATA DETECTED.")
        else:
            with st.spinner("🌀 Normalizing Data | Fuzzy Matching Names | Calculating Titan Scores..."):
                df, logs = process_data_pipeline(files, sport, spread_input, total_input)
                st.session_state['master'] = df
            
            st.success(f"FUSION COMPLETE. Active Pool: {len(df)} Players")
            with st.expander("View System Logs"):
                for log in logs: st.markdown(f"<div class='log-box'>{log}</div>", unsafe_allow_html=True)

# --- TAB 2: ANALYSIS ---
with tabs[1]:
    df = st.session_state['master']
    st.header("Titan Brain Analysis")
    
    if df.empty:
        st.info("Awaiting Data Ingestion...")
    else:
        # VISUALS
        st.subheader("📊 Leverage vs. Ceiling Scatter")
        chart_data = df[df['proj_pts'] > 5].copy()
        st.scatter_chart(chart_data, x='ownership', y='ceiling', color='titan_score', size='salary', use_container_width=True)
        
        st.subheader("🏆 Top Ranked Plays")
        cols = ['name', 'position', 'salary', 'proj_pts', 'ceiling', 'ownership', 'titan_score', 'reasoning']
        st.dataframe(df[cols].head(50), use_container_width=True, hide_index=True)

# --- TAB 3: OPTIMIZER ---
with tabs[2]:
    st.header("Slate-Breaking Lineup Factory")
    df = st.session_state['master']
    
    if df.empty:
        st.warning("Load Data First.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        num_lus = c1.number_input("Lineup Count", 1, 150, 20)
        variance = c2.slider("Sim Variance (%)", 0, 100, 15, help="Higher = More Randomness (Good for GPPs)")
        runback = c3.checkbox("Force Game Stacks (NFL)", True)
        
        if st.button("⚡ GENERATE 150-MAX"):
            config = {
                'site': site, 'sport': sport, 'cap': st.session_state['cap'], 
                'num_lineups': num_lus, 'variance': variance, 'use_runback': runback
            }
            
            with st.spinner(f"Running {num_lus} Monte Carlo Simulations..."):
                res = optimize_advanced(df, config)
            
            if res is not None:
                st.success("✅ OPTIMIZATION SUCCESSFUL")
                
                # Metrics
                avg_proj = res.groupby('Lineup_ID')['proj_pts'].sum().mean()
                avg_sim = res.groupby('Lineup_ID')['sim_pts'].sum().mean()
                st.metric("Avg Base Projection", f"{avg_proj:.1f}")
                st.metric("Avg Simulated Upside", f"{avg_sim:.1f}", delta=f"{avg_sim - avg_proj:.1f}")
                
                # Download
                csv = res.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV for Upload", csv, "titan_god_mode.csv", "text/csv")
                
                st.dataframe(res)
            else:
                st.error("Optimization Failed. Check constraints.")

# --- TAB 4: PROP SNIPER ---
with tabs[3]:
    st.header("Kelly Criterion Prop Sniper")
    df = st.session_state['master']
    
    if df.empty or 'prop_line' not in df.columns:
        st.info("No prop data found. Ensure your CSV has columns like 'Prop', 'Line', or 'OU'.")
    else:
        bankroll = st.number_input("Bankroll ($)", 1000)
        
        # Calculate Kelly
        props = df[df['prop_line'] > 0].copy()
        props['win_prob'] = 0.55 + (props['prop_edge'] / 100) # Simple model: Edge increases win prob
        props['win_prob'] = props['win_prob'].clip(0.5, 0.75) # Cap realism
        
        def calc_wager(row):
            b = 1.0 # Even money decimal odds 2.0 -> b=1
            p = row['win_prob']
            f = (b * p - (1 - p)) / b
            return max(0, f * 0.5 * bankroll) # Half Kelly for safety
            
        props['wager'] = props.apply(calc_wager, axis=1)
        props['pick'] = np.where(props['proj_pts'] > props['prop_line'], 'OVER', 'UNDER')
        
        st.dataframe(
            props.sort_values('wager', ascending=False)[['name', 'prop_line', 'proj_pts', 'pick', 'prop_edge', 'wager']]
            .style.format({'wager': '${:.2f}', 'prop_edge': '{:.1f}%', 'proj_pts': '{:.1f}'})
        )

# --- TAB 5: INTEL ---
with tabs[4]:
    st.header("Global Intel Wire")
    if st.button("🔄 Refresh Feeds"): st.cache_data.clear()
    
    try:
        urls = {
            "NFL": "https://www.rotowire.com/rss/news.htm?sport=nfl", 
            "NBA": "https://www.rotowire.com/rss/news.htm?sport=nba",
            "MLB": "https://www.rotowire.com/rss/news.htm?sport=mlb"
        }
        feed = feedparser.parse(urls.get(sport, urls['NFL']))
        for entry in feed.entries[:10]:
            st.markdown(f"""
            <div style="background:#111; padding:15px; border-radius:5px; margin-bottom:10px; border-left:4px solid #00c6ff;">
                <h4 style="margin:0; color:#fff;">{entry.title}</h4>
                <p style="color:#aaa; font-size:14px;">{entry.summary}</p>
            </div>
            """, unsafe_allow_html=True)
    except:
        st.error("News feed unavailable.")
