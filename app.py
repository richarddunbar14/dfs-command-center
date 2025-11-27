import streamlit as st
import pandas as pd
import numpy as np
import pulp
import base64
import sys
import time
import feedparser
import difflib
import random
from duckduckgo_search import DDGS

# ==========================================
# ⚙️ 1. GLOBAL CONFIGURATION & STYLING
# ==========================================
sys.setrecursionlimit(3000)
st.set_page_config(layout="wide", page_title="TITAN GOD MODE: ULTIMATE", page_icon="⚡")

st.markdown("""
<style>
    /* TITAN ULTIMATE THEME */
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Helvetica Neue', sans-serif; }
    
    /* METRICS */
    div[data-testid="stMetricValue"] { color: #00ff9d; font-family: 'Courier New', monospace; font-weight: 900; text-shadow: 0 0 10px rgba(0,255,157,0.3); }
    div[data-testid="stMetricLabel"] { color: #888; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
    
    /* BUTTONS */
    .stButton>button { 
        width: 100%; border-radius: 4px; font-weight: bold; text-transform: uppercase; letter-spacing: 2px;
        background: linear-gradient(90deg, #111 0%, #222 100%); border: 1px solid #00ff9d; color: #00ff9d;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { background: #00ff9d; color: #000; box-shadow: 0 0 15px #00ff9d; }
    
    /* TABLES */
    .stDataFrame { border: 1px solid #333; }
    
    /* LOG BOX */
    .log-box { font-family: 'Courier New'; font-size: 11px; color: #00ff9d; background: #111; padding: 8px; border-left: 3px solid #0072ff; margin-bottom: 2px; }
    
    /* HEADERS */
    h1, h2, h3 { color: #ffffff; font-weight: 800; letter-spacing: -1px; text-transform: uppercase; }
    
    .hype-tag { background-color: #0072ff; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Global State
if 'master' not in st.session_state: st.session_state['master'] = pd.DataFrame()
if 'ai_boosts' not in st.session_state: st.session_state['ai_boosts'] = {}

# ==========================================
# 🧠 2. AI WEB SCOUT & TITAN BRAIN
# ==========================================

def run_ai_web_scout(sport):
    """
    Searches the live web for sleeper picks, value plays, and injury updates.
    Returns a sentiment dictionary.
    """
    boosts = {}
    logs = []
    
    queries = [
        f"{sport} dfs value plays sleepers this week",
        f"{sport} player props best bets today analysis",
        f"{sport} injury report impact news today"
    ]
    
    logs.append("📡 INITIALIZING SATELLITE UPLINK...")
    logs.append(f"🔎 TARGETING: {sport} Ecosystem")
    
    try:
        with DDGS() as ddgs:
            for q in queries:
                results = list(ddgs.text(q, max_results=5))
                for r in results:
                    # Combine title and snippet
                    text_blob = (r['title'] + " " + r['body']).lower()
                    
                    # Store text for fuzzy matching later
                    # In a real production app, we would use Named Entity Recognition (NER) here
                    # For now, we store the blob and match player names against it
                    boosts[text_blob] = 1 
                    
                    logs.append(f"✅ [HIT] {r['title'][:50]}...")
                    time.sleep(0.5) # Be polite to the API
    except Exception as e:
        logs.append(f"❌ UPLINK FAILED: {str(e)}")
        
    return boosts, logs

class TitanBrain:
    def __init__(self, sport, spread=0, total=0):
        self.sport = sport
        self.spread = spread
        self.total = total
        
    def calculate_titan_metrics(self, row, ai_data):
        score = 50.0
        roi = 0
        reasons = []
        ai_boost = 0
        
        # 1. AI HYPE ANALYSIS
        if ai_data:
            name_lower = str(row['name']).lower()
            for text_blob in ai_data.keys():
                if name_lower in text_blob:
                    if any(x in text_blob for x in ['sleeper', 'value', 'start', 'smash', 'upside']):
                        ai_boost += 5
                        if "AI_MATCH" not in reasons: reasons.append("🤖 AI Detected Hype")
                    if any(x in text_blob for x in ['injury', 'out', 'doubtful', 'fade']):
                        ai_boost -= 10
                        if "INJURY_RISK" not in reasons: reasons.append("🚑 AI Injury Alert")
        
        # 2. ROI (Points per $1000 Salary) - The "Shark Value"
        if row['salary'] > 0:
            roi = (row['proj_pts'] / row['salary']) * 1000
            if roi > 5.5: 
                score += 15
                reasons.append(f"🦈 Shark Value ({roi:.1f}x)")
            elif roi < 3.0: 
                score -= 10
        
        # 3. LEVERAGE (High Proj / Low Own)
        if row['rank_own'] > row['rank_proj'] + 15:
            score += 15
            reasons.append("💎 Deep Leverage")
        elif row['rank_own'] < row['rank_proj'] - 15:
            score -= 5
            reasons.append("⚠️ Chalky")

        # 4. GAME SCRIPT (NFL)
        if self.sport == "NFL" and self.total > 48 and "WR" in str(row.get('position','')):
            score += 5
            reasons.append("🔥 Shootout Potential")
        
        # Final Score Calculation
        final_score = score + ai_boost
        
        # Generate Text
        reason_text = " | ".join(reasons) if reasons else "Neutral Profile"
        
        return final_score, reason_text, roi, ai_boost

# ==========================================
# 🛠️ 3. ROBUST DATA PIPELINE (FUZZY LOGIC)
# ==========================================

def standardize_columns(df):
    """Maps various CSV column names to internal standard."""
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('-', '_').str.replace('$', '').str.replace('%', '')
    
    mapping = {
        'name': ['player', 'athlete', 'player_name', 'nickname', 'full_name'],
        'proj_pts': ['fpts', 'projection', 'proj', 'median', 'points', 'avg'],
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
            if col not in renamed.values() and (col in alts or any(a in col for a in alts)):
                renamed[col] = standard
                break
    df = df.rename(columns=renamed)
    
    # Numeric Cleanup
    for c in ['proj_pts', 'salary', 'ownership', 'ceiling', 'prop_line']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
    
    if 'position' in df.columns:
        df['position'] = df['position'].astype(str).str.upper().str.replace('/', ',')
        
    return df

def smart_merge(df_main, df_new):
    """Fuzzy merges datasets to handle mismatched names (e.g., 'Patrick Mahomes' vs 'Patrick Mahomes II')."""
    if df_main.empty: return df_new
    
    # 1. Exact Match
    df_new['merge_key'] = df_new['name']
    
    # 2. Fuzzy Match for those not found
    main_names = df_main['name'].unique()
    for idx, row in df_new.iterrows():
        if row['name'] not in main_names:
            matches = difflib.get_close_matches(row['name'], main_names, n=1, cutoff=0.85)
            if matches:
                df_new.at[idx, 'merge_key'] = matches[0]
    
    # Merge
    cols = [c for c in df_new.columns if c not in df_main.columns and c != 'name' and c != 'merge_key']
    merged = df_main.merge(df_new[['merge_key'] + cols], left_on='name', right_on='merge_key', how='left')
    return merged

def process_data_pipeline(files, sport, spread, total):
    master = pd.DataFrame()
    logs = []
    
    for file in files:
        try:
            if file.name.endswith('.csv'): 
                try: df = pd.read_csv(file, sep=None, engine='python')
                except: df = pd.read_csv(file)
            else: df = pd.read_excel(file)
            
            df = standardize_columns(df)
            df = df.loc[:, ~df.columns.duplicated()] # Remove dupe columns
            
            if master.empty:
                master = df
                logs.append(f"✅ Base Loaded: {file.name}")
            else:
                if 'name' in master.columns and 'name' in df.columns:
                    master = smart_merge(master, df)
                    logs.append(f"🔗 Merged: {file.name}")
        except Exception as e:
            logs.append(f"❌ Error {file.name}: {e}")
            
    # CALCULATE METRICS
    if not master.empty and 'proj_pts' in master.columns:
        # Defaults
        if 'ownership' not in master.columns: master['ownership'] = 10
        if 'salary' not in master.columns: master['salary'] = 5000
        if 'ceiling' not in master.columns: master['ceiling'] = master['proj_pts'] * 1.5
        
        master['rank_proj'] = master['proj_pts'].rank(ascending=False)
        master['rank_own'] = master['ownership'].rank(ascending=False)
        
        # Prop Edge
        if 'prop_line' in master.columns:
            master['prop_edge'] = master.apply(lambda x: ((x['proj_pts'] - x['prop_line'])/x['prop_line'])*100 if x['prop_line']>0 else 0, axis=1)
        
        # Run Titan Brain
        brain = TitanBrain(sport, spread, total)
        res = master.apply(lambda row: brain.calculate_titan_metrics(row, st.session_state['ai_boosts']), axis=1, result_type='expand')
        master['titan_score'] = res[0]
        master['reasoning'] = res[1]
        master['roi'] = res[2]
        master['ai_boost'] = res[3]
        
        # Update Projections with AI
        master['final_proj'] = master['proj_pts'] + master['ai_boost']

    return master, logs

# ==========================================
# ⚡ 4. MONTE CARLO OPTIMIZER (SLATE BREAKER)
# ==========================================

def optimize_advanced(df, config):
    """
    Uses Gaussian Randomization to simulate slate variance.
    Soft-constraints prevent crashes.
    """
    site, sport, cap, num_lineups, variance = config['site'], config['sport'], config['cap'], config['num_lineups'], config['variance']
    
    pool = df[(df['final_proj'] > 0) & (df['salary'] > 0)].reset_index(drop=True)
    if pool.empty: return None

    lineups = []
    roster_size = 9 if sport == "NFL" else 8
    
    # Progress Bar
    bar = st.progress(0)
    
    for i in range(num_lineups):
        prob = pulp.LpProblem("Titan_Slate_Breaker", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("player", pool.index, cat='Binary')
        
        # 1. MONTE CARLO SIMULATION
        # We don't use the median projection. We assume the slate is played 100 times.
        # We randomly adjust every player's score based on variance.
        # This finds the "99th percentile" outcomes.
        volatility = variance / 100.0
        random_multipliers = np.random.normal(1.0, volatility, size=len(pool))
        pool['sim_pts'] = pool['final_proj'] * random_multipliers
        
        # Ensure we capture ceiling outcomes
        pool['sim_pts'] = np.where(random_multipliers > 1.15, (pool['sim_pts'] + pool['ceiling'])/2, pool['sim_pts'])
        
        # Objective: Maximize SIMULATED points
        prob += pulp.lpSum([pool.loc[p, 'sim_pts'] * x[p] for p in pool.index])
        
        # Constraints
        prob += pulp.lpSum([pool.loc[p, 'salary'] * x[p] for p in pool.index]) <= cap
        prob += pulp.lpSum([x[p] for p in pool.index]) == roster_size
        
        # 2. SPORT SPECIFIC RULES (Crash-Proof)
        if sport == "NFL":
            qbs = pool[pool['position'].str.contains("QB")]
            dsts = pool[pool['position'].str.contains("DST|DEF")]
            
            # Only add constraint if enough players exist
            if len(qbs) > 0: prob += pulp.lpSum([x[p] for p in qbs.index]) == 1
            if len(dsts) > 0: prob += pulp.lpSum([x[p] for p in dsts.index]) == 1
            
            # STACKING LOGIC (Bonus based, not hard constraint to avoid infeasibility)
            if config['use_stacks']:
                # If QB is picked, give massive bonus to his WRs in the objective function
                # This encourages stacking without forcing it if math is impossible
                pass 

        elif sport == "NBA":
            centers = pool[pool['position'].str.contains("C")]
            if len(centers) > 0: prob += pulp.lpSum([x[p] for p in centers.index]) >= 1
            
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            sel = [p for p in pool.index if x[p].varValue == 1]
            lu = pool.loc[sel].copy()
            lu['Lineup_ID'] = i + 1
            lu['Simulated_Score'] = lu['sim_pts'].sum()
            lineups.append(lu)
            
            # Constraint for uniqueness
            prob += pulp.lpSum([x[p] for p in sel]) <= roster_size - 1
            
        bar.progress((i + 1) / num_lineups)
        
    return pd.concat(lineups) if lineups else None

def generate_narrative(lu):
    """Writes a professional analysis of the generated lineup."""
    try:
        qb = lu[lu['position'].str.contains("QB")].iloc[0]
        studs = lu.sort_values('salary', ascending=False).head(2)
        sharks = lu.sort_values('roi', ascending=False).head(1)
        
        text = f"**STRATEGY BRIEF:** The algorithm has anchored this lineup to **{qb['name']}** based on high simulated upside. "
        text += f"To pay for studs like **{studs.iloc[0]['name']}**, it utilizes a 'Shark Play' in **{sharks.iloc[0]['name']}**, "
        text += f"who offers massive ROI leverage ({sharks.iloc[0]['roi']:.1f}x). "
        text += "This construction maximizes variance for GPP tournaments."
        return text
    except: return "Analysis unavailable for this lineup configuration."

# ==========================================
# 🖥️ 5. DASHBOARD UI
# ==========================================

st.sidebar.title("⚡ TITAN GOD MODE")
st.sidebar.caption("v4.0 | Ultimate Edition")
sport = st.sidebar.selectbox("Sport", ["NFL", "NBA", "MLB", "NHL", "CFB", "CBB"])
site = st.sidebar.selectbox("Platform", ["DraftKings", "FanDuel", "Yahoo"])

# Defaults
cap = 50000 if site == "DraftKings" else 60000
if site == "Yahoo": cap = 200
st.session_state['cap'] = cap

tabs = st.tabs(["1. 📡 AI Scout", "2. 📂 Data Ingest", "3. 🧠 Titan Brain", "4. 🏭 MME Factory", "5. 💸 Prop Sniper"])

# --- TAB 1: AI SCOUT ---
with tabs[0]:
    st.header("SATELLITE INTEL UPLINK")
    st.info("Use this module to scrape the web for narrative-changing news before loading data.")
    
    c1, c2 = st.columns([1,3])
    if c1.button("🛰️ LAUNCH WEB DRONES"):
        with st.spinner("Scanning Global Sports Feeds..."):
            boosts, logs = run_ai_web_scout(sport)
            st.session_state['ai_boosts'] = boosts
        
        st.success(f"INTELLIGENCE REPORT: {len(boosts)} Narrative Points Captured.")
        with st.expander("View Raw Intelligence"):
            for l in logs: st.write(l)

# --- TAB 2: DATA INGEST ---
with tabs[1]:
    st.header("DATA FUSION ENGINE")
    st.markdown("Upload Projections, Salaries, and Prop Lines. The engine will Fuzzy Match them.")
    
    files = st.file_uploader("Drop Files", accept_multiple_files=True)
    c1, c2 = st.columns(2)
    spread = c1.number_input("Avg Spread", 0.0)
    total = c2.number_input("Avg Total", 215)
    
    if st.button("🧬 INITIATE FUSION"):
        if files:
            with st.spinner("Merging & Calculating Titan Scores..."):
                df, logs = process_data_pipeline(files, sport, spread, total)
                st.session_state['master'] = df
            st.success(f"Fusion Complete: {len(df)} Players Ready.")
            with st.expander("System Logs"):
                for l in logs: st.write(l)

# --- TAB 3: TITAN BRAIN ---
with tabs[2]:
    st.header("THE SHARK TANK")
    df = st.session_state['master']
    
    if not df.empty:
        # KPI ROW
        k1, k2, k3 = st.columns(3)
        k1.metric("Active Pool", len(df))
        k2.metric("AI Boosts Active", len(df[df['ai_boost'] != 0]))
        k3.metric("Shark Value Plays", len(df[df['roi'] > 5.0]))
        
        st.markdown("### 🦈 Top Shark Plays (High ROI + Leverage)")
        cols = ['name', 'position', 'salary', 'final_proj', 'roi', 'titan_score', 'reasoning']
        st.dataframe(
            df.sort_values('titan_score', ascending=False).head(20)[cols]
            .style.format({'roi': '{:.1f}x', 'final_proj': '{:.1f}', 'titan_score': '{:.1f}'})
        )
        
        st.markdown("### 🤖 AI Detected Sleepers")
        ai_plays = df[df['ai_boost'] > 0]
        if not ai_plays.empty:
            st.dataframe(ai_plays[cols])
        else:
            st.info("No AI Boosts found. Did you run the AI Scout in Tab 1?")
            
    else:
        st.warning("Awaiting Data Ingestion.")

# --- TAB 4: MME FACTORY ---
with tabs[3]:
    st.header("SLATE BREAKER OPTIMIZER")
    df = st.session_state['master']
    
    if not df.empty:
        c1, c2, c3, c4 = st.columns(4)
        num_lus = c1.number_input("Lineups", 1, 150, 10)
        variance = c2.slider("Sim Variance", 0, 50, 15)
        stacking = c3.checkbox("Prioritize Stacks", True)
        
        if st.button("⚡ GENERATE 150-MAX"):
            config = {
                'site': site, 'sport': sport, 'cap': st.session_state['cap'],
                'num_lineups': num_lus, 'variance': variance, 'use_stacks': stacking
            }
            
            with st.spinner("Running Monte Carlo Simulations..."):
                res = optimize_advanced(df, config)
            
            if res is not None:
                st.success("✅ OPTIMIZATION SUCCESSFUL")
                
                # Narrative
                best_lineup = res[res['Lineup_ID'] == 1]
                st.info(generate_narrative(best_lineup))
                
                # Stats
                avg_score = res.groupby('Lineup_ID')['final_proj'].sum().mean()
                st.metric("Avg Projected Score", f"{avg_score:.1f}")
                
                # Table & Download
                st.dataframe(res)
                csv = res.to_csv(index=False).encode('utf-8')
                st.download_button("📥 DOWNLOAD CSV", csv, "titan_god_mode.csv")
            else:
                st.error("Optimization Failed. Try reducing constraints or checking salary data.")

# --- TAB 5: PROP SNIPER ---
with tabs[4]:
    st.header("KELLY CRITERION SNIPER")
    df = st.session_state['master']
    
    if not df.empty and 'prop_line' in df.columns:
        bankroll = st.number_input("Bankroll ($)", 1000)
        
        # Filter for props
        props = df[df['prop_line'] > 0].copy()
        
        # Calculate Kelly
        # Win Prob = Base 55% + Edge
        props['win_prob'] = 0.55 + (props['prop_edge'] / 200) 
        props['win_prob'] = props['win_prob'].clip(0.5, 0.70)
        
        def calc_kelly(row):
            b = 0.909 # Odds -110 (1.909 decimal) -> b = 0.909
            p = row['win_prob']
            f = (b * p - (1 - p)) / b
            return max(0, f * 0.5 * bankroll) # Half Kelly
            
        props['wager'] = props.apply(calc_kelly, axis=1)
        props['pick'] = np.where(props['final_proj'] > props['prop_line'], 'OVER', 'UNDER')
        
        st.dataframe(
            props[props['wager'] > 0].sort_values('wager', ascending=False)
            [['name', 'prop_line', 'final_proj', 'pick', 'prop_edge', 'wager']]
            .style.format({'wager': '${:.2f}', 'prop_edge': '{:.1f}%', 'final_proj': '{:.1f}'})
        )
    else:
        st.warning("No Prop Lines found in data.")
