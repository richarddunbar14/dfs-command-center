import streamlit as st
import pandas as pd
import numpy as np
import pulp
import base64
import math
import feedparser
import sys
from duckduckgo_search import DDGS # Required for Research Agent

# Increase recursion limit for the pulp solver
sys.setrecursionlimit(2000)

# ==========================================
# ⚙️ 1. TITAN AI CONFIGURATION & STYLING
# ==========================================
st.set_page_config(layout="wide", page_title="TITAN COMMAND: FINAL FUSION", page_icon="💥")

st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #e2e8f0; font-family: 'Roboto', sans-serif; }
    div[data-testid="stMetricValue"] { color: #38bdf8; font-size: 24px; font-weight: 800; }
    .reasoning-text { font-size: 12px; color: #94a3b8; font-style: italic; border-left: 2px solid #3b82f6; padding-left: 8px; }
    .stButton>button { width: 100%; border-radius: 6px; font-weight: bold; background-color: #3b82f6; border: none; }
</style>
""", unsafe_allow_html=True)

# --- CRITICAL: GLOBAL SESSION STATE INITIALIZATION (FIXES KeyError) ---
if 'master' not in st.session_state: st.session_state['master'] = pd.DataFrame()
if 'props' not in st.session_state: st.session_state['props'] = pd.DataFrame() # Changed to DF for simpler handling

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
                reasons.append(f"💎 High Leverage (Contrarian Value)")
            elif leverage_diff < -10:
                score -= 10
                reasons.append("⚠️ Chalk Trap (Overowned)")
                
        # 2. GAME SCRIPT LOGIC (NFL Focus)
        if self.sport == "NFL" and 'position' in row:
            if abs(self.spread) > 7 and 'WR' in str(row['position']):
                score += 7
                reasons.append("📜 Blowout Script: Garbage Time potential")

        # 3. PROP VALUE LOGIC
        if 'prop_line' in row and 'proj_pts' in row and row['prop_line'] > 0:
            edge = ((row['proj_pts'] - row['prop_line']) / row['proj_pts']) * 100
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
# 🛠️ 3. DATA PROCESSING & UTILITIES
# ==========================================

def standardize_columns(df):
    """Universal Translator & Cleaner (Safest Version)"""
    # Fix column names to be Python/Pandas safe
    df.columns = df.columns.astype(str).str.lower().str.strip().str.replace(' ', '_').str.replace('-', '_').str.replace('$', '').str.replace(',', '').str.replace('%', '')
    column_map = {
        'name': ['player', 'athlete', 'full_name'], 'proj_pts': ['projection', 'proj', 'fpts', 'median'],
        'ownership': ['own', 'projected_ownership'], 'salary': ['cost', 'sal', 'price'],
        'prop_line': ['line', 'prop', 'ou', 'total', 'strike'], 'position': ['pos', 'roster_position'],
        'team': ['squad', 'tm'], 'ceiling': ['ceil', 'max_pts'], 'opp_rank': ['dvp', 'opp_rank']
    }
    renamed = {}
    for std, alts in column_map.items():
        for col in df.columns:
            if col not in renamed.values():
                if any(a in col for a in alts):
                    renamed[col] = std
                    break
    df = df.rename(columns=renamed)
    
    # Clean Numerics
    for c in ['proj_pts', 'salary', 'ownership', 'prop_line', 'ceiling']:
        if c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.replace('$', '').str.replace(',', '').str.replace('%', '')
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
    if 'position' in df.columns:
        df['position'] = df['position'].astype(str).str.upper().str.replace('/', ',')
        
    return df

def process_and_analyze(files, sport, spread, total):
    """Handles multi-file ingestion and runs the Titan Brain."""
    master = pd.DataFrame()
    
    for file in files:
        try:
            if file.name.endswith('.csv'): df = pd.read_csv(file)
            else: df = pd.read_excel(file)
            
            df = standardize_columns(df)
            
            # --- SAFETY CHECK (Prevents the Array Error) ---
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, (list, dict, np.ndarray)) and len(x) > 0).any():
                    df = df.drop(columns=[col]) 
            
            # Merge Logic: If master is empty, set it. Otherwise, merge on name.
            if master.empty: 
                master = df
            elif 'name' in master.columns and 'name' in df.columns:
                cols_to_merge = [c for c in df.columns if c not in master.columns or c == 'name']
                master = master.merge(df[cols_to_merge], on='name', how='left')
                
        except Exception as e: st.error(f"Error loading file: {e}")
            
    # RUN SHARK LOGIC only if possible
    if 'proj_pts' in master.columns and 'ownership' in master.columns:
        master['rank_proj'] = master['proj_pts'].rank(ascending=False)
        master['rank_own'] = master['ownership'].rank(ascending=False)
        
        brain = TitanBrain(sport, spread, total)
        results = master.apply(lambda row: brain.evaluate_player(row), axis=1, result_type='expand')
        master['shark_score'] = results[0]
        master['reasoning'] = results[1]
        
    return master

def get_player_pool(df, top_n_shark=25, top_n_value=15):
    """Generates the final player pool based on multiple criteria."""
    
    # --- CRITICAL FIX FOR KEYERROR ---
    if 'shark_score' not in df.columns: df['shark_score'] = 50.0 
    if 'reasoning' not in df.columns: df['reasoning'] = "N/A - Run Titan Brain First"
    # --- END FIX ---

    if df.empty: return pd.DataFrame()

    pool_shark = df.nlargest(top_n_shark, 'shark_score')
    
    if 'salary' in df.columns and 'proj_pts' in df.columns:
        df['value'] = (df['proj_pts'] / df['salary']) * 1000
        low_sal_thresh = 5000 
        pool_value = df[df['salary'] <= low_sal_thresh].nlargest(top_n_value, 'value')
    else:
        pool_value = pd.DataFrame()

    final_pool = pd.concat([pool_shark, pool_value]).drop_duplicates(subset=['name'])
    return final_pool.sort_values(by='shark_score', ascending=False)

# --- OPTIMIZER & LIVE INTEL (Continued) ---

def optimize_lineup(df, config):
    # This remains the core solver with explicit constraints
    site, sport, cap, num_lineups, target_col, use_correlation = (
        config['site'], config['sport'], config['cap'], config['num_lineups'], config['target_col'], config['use_correlation']
    )
    
    pool = df[(df[target_col] > 0) & (df['salary'] > 0)].reset_index(drop=True)
    roster_size = 9 if sport=="NFL" else 8
    valid_lineups = []
    
    for _ in range(num_lineups):
        prob = pulp.LpProblem("Apex_Optimizer", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("P", pool.index, cat='Binary')
        
        prob += pulp.lpSum([pool.loc[i, target_col] * x[i] for i in pool.index])
        prob += pulp.lpSum([pool.loc[i, 'salary'] * x[i] for i in pool.index]) <= cap
        prob += pulp.lpSum([x[i] for i in pool.index]) == roster_size
        
        # Position Logic (NFL Example)
        if sport == "NFL":
            qbs = pool[pool['position'].str.contains('QB', na=False)]
            dsts = pool[pool['position'].str.contains('DST|DEF', na=False)]
            prob += pulp.lpSum([x[i] for i in qbs.index]) == 1
            prob += pulp.lpSum([x[i] for i in dsts.index]) == 1
            
            if use_correlation:
                for qb_idx in qbs.index:
                    stack_partners = pool[(pool['team'] == pool.loc[qb_idx, 'team']) & (pool['position'].str.contains('WR|TE', na=False))]
                    # Fix: Ensure correct variable used in sum
                    prob += pulp.lpSum([x[i] for i in stack_partners.index]) >= x[qb_idx]
        
        elif sport == "NBA":
             cs = pool[pool['position'].str.contains('C', na=False)]
             prob += pulp.lpSum([x[i] for i in cs.index]) >= 1

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            sel = [i for i in pool.index if x[i].varValue == 1]
            lineup = pool.loc[sel].copy()
            lineup['Lineup_ID'] = _ + 1
            valid_lineups.append(lineup)
            prob += pulp.lpSum([x[i] for i in sel]) <= len(sel) - 1
        else: break
            
    return pd.concat(valid_lineups) if valid_lineups else None

def kelly_criterion(bankroll, odds_decimal, win_probability, multiplier=0.5):
    """Calculates optimal fraction of bankroll to wager."""
    b = odds_decimal - 1
    f = (b * win_probability - (1 - win_probability)) / b
    return max(0, f * multiplier)

def fetch_rotowire_news(sport):
    urls = {"NFL": "https://www.rotowire.com/rss/news.htm?sport=nfl", "NBA": "https://www.rotowire.com/rss/news.htm?sport=nba"}
    try:
        feed = feedparser.parse(urls.get(sport, urls['NFL']))
        return [{"title": x.title, "summary": x.summary} for x in feed.entries[:5]]
    except: return []

def get_csv_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="titan_lineups.csv" class="stButton">📥 Download CSV</a>'

# ==========================================
# 🖥️ 6. UI DASHBOARD
# ==========================================

st.sidebar.title("🦁 TITAN APEX CORE")
sport = st.sidebar.selectbox("Sport", ["NFL", "NBA", "MLB", "NHL"], index=0)
site = st.sidebar.selectbox("Sportsbook", ["DraftKings", "FanDuel", "Yahoo", "PrizePicks", "Underdog"], index=0)

default_cap = 50000
if site == "FanDuel": default_cap = 60000
elif site == "Yahoo": default_cap = 200

st.sidebar.markdown("---")
st.session_state['cap'] = default_cap
st.session_state['site'] = site

tabs = st.tabs(["1. 💾 Data Ingest", "2. 🎯 Player Pool", "3. 🏗️ Optimizer", "4. 💸 Prop Sniper", "5. 📰 Live Intel"])

# --- TAB 1: DATA INGEST ---
with tabs[0]:
    st.title("1. 💾 Data Ingest & Logic Settings")
    
    files = st.file_uploader("Drop Files (Multi-File Supported)", accept_multiple_files=True)
    
    st.markdown("### Game Theory Input")
    c1, c2 = st.columns(2)
    spread = c1.number_input("Vegas Spread (Avg)", -20.0, 20.0, 0.0)
    total = c2.number_input("Vegas Total (Avg)", 150, 250, 210)
    
    if st.button("🚀 Process & Run Titan Brain"):
        with st.spinner("Calculating Shark Scores..."):
            df = process_and_analyze(files, sport, spread, total)
            st.session_state['master'] = df
            st.success("✅ Titan Brain Analysis Complete.")

# --- TAB 2: PLAYER POOL & UPSIDE ---
with tabs[1]:
    df = st.session_state['master']
    st.title("2. 🎯 Final Player Pool & High Upside")

    if df.empty: st.warning("Upload data first.")
    else:
        pool_df = get_player_pool(df)
        
        st.subheader("Final Roster Pool")
        # Player Position Filter
        if 'position' in pool_df.columns:
            all_pos = ['ALL'] + sorted(list(pool_df['position'].unique()))
            pos_filter = st.selectbox("Filter Pool by Position", all_pos)
            if pos_filter != 'ALL':
                pool_df = pool_df[pool_df['position'] == pos_filter]

        st.dataframe(
            pool_df[['name', 'position', 'salary', 'proj_pts', 'shark_score', 'reasoning']],
            use_container_width=True
        )

        st.subheader("Reasoning Breakdown")
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
    df = st.session_state['master']
    st.title(f"3. 🏗️ {st.session_state['site']} Lineup Builder")
    
    if df.empty: st.warning("Upload data first.")
    else:
        c1, c2, c3 = st.columns(3)
        num = c1.number_input("Lineups (MME Max 150)", 1, 150, 10)
        cap = c2.number_input("Salary Cap", value=st.session_state['cap'])
        target = c3.selectbox("Optimization Goal", ["Shark Score (Context/GPP)", "Projected Points (Cash)"])
        
        corr = st.checkbox("✅ Auto-Correlation (Stacking/Pairing)", value=True)
        
        if st.button("⚡ BUILD SLATE-BREAKING LINEUPS"):
            optimizer_pool = get_player_pool(df)
            target_col = 'shark_score' if target == "Shark Score (Context/GPP)" else 'proj_pts'
            
            with st.spinner(f"Optimizing {num} lineups..."):
                config = {'site': site, 'sport': sport, 'cap': cap, 'num_lineups': num, 'target_col': target_col, 'use_correlation': corr}
                res = optimize_lineup(optimizer_pool, config)
            
            if res is not None:
                st.success(f"Generated {num} Lineups!")
                st.metric("Avg Lineup Score", f"{res.groupby('Lineup_ID')['proj_pts'].sum().mean():.2f}")
                st.markdown(get_csv_download(res), unsafe_allow_html=True)
            else: st.error("Optimization Failed. Check pool size or constraints.")

# --- TAB 4: PROP SNIPER ---
with tabs[3]:
    df = st.session_state['master']
    st.title("4. 💸 Prop Sniper & Bankroll")
    
    if 'prop_line' not in df.columns: st.warning("No Prop Lines found in data.")
    else:
        df['edge'] = ((df['proj_pts'] - df['prop_line']) / df['prop_line']) * 100
        df['pick'] = np.where(df['edge']>0, 'OVER', 'UNDER')
        
        st.subheader("Prop Edge Analysis")
        st.dataframe(
            df.sort_values('edge', ascending=False)[['name', 'prop_line', 'proj_pts', 'pick', 'edge']],
            use_container_width=True
        )
        
        st.markdown("---")
        st.subheader("💰 Bankroll Management (Kelly Criterion)")
        
        c1, c2, c3 = st.columns(3)
        bankroll = c1.number_input("Total Bankroll ($)", value=1000)
        prob_win = c2.number_input("Est. Win Probability (e.g., 0.55)", 0.0, 1.0, 0.55, step=0.01)
        odds_decimal = c3.number_input("Odds (Decimal, e.g., 2.0)", 1.0, 10.0, 2.0)
        
        kelly_frac = kelly_criterion(bankroll, odds_decimal, prob_win, multiplier=0.5)
        
        st.metric("Recommended Wager (50% Kelly)", f"${bankroll * kelly_frac:.2f}", f"{kelly_frac:.2%} of Bankroll")

# --- TAB 5: LIVE INTEL ---
with tabs[4]:
    st.title("5. 📰 Live Intel & News Wire")
    
    st.subheader("🚨 Breaking News Wire")
    news = fetch_rotowire_news(sport)
    if st.button("🔄 Refresh News"): st.cache_data.clear()
    
    for item in news:
        st.info(f"**{item['title']}**\n\n{item['summary']}")
