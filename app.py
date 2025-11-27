import streamlit as st
import pandas as pd
import numpy as np
import pulp
import base64
from duckduckgo_search import DDGS
import math
import feedparser

# Try importing NFL data (optional)
try:
    import nfl_data_py as nfl
    NFL_DATA_AVAILABLE = True
except ImportError:
    NFL_DATA_AVAILABLE = False

# ==========================================
# 1. TITAN CONFIG
# ==========================================
st.set_page_config(layout="wide", page_title="TITAN COMMAND: FINAL FUSION", page_icon="💥")

st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    div[data-testid="stMetricValue"] { color: #38bdf8; font-size: 24px; font-weight: 800; }
    .reasoning-text { font-size: 12px; color: #94a3b8; font-style: italic; border-left: 2px solid #3b82f6; padding-left: 8px; }
    .stButton>button { width: 100%; border-radius: 6px; font-weight: bold; background-color: #3b82f6; border: none; }
</style>
""", unsafe_allow_html=True)

# Session State
if 'master' not in st.session_state:
    st.session_state['master'] = pd.DataFrame()

# ==========================================
# 2. TITAN BRAIN
# ==========================================
class TitanBrain:
    def __init__(self, sport, spread=0, total=0):
        self.sport = sport
        self.spread = spread
        self.total = total
        
    def evaluate_player(self, row):
        reasons = []
        score = 50.0
        
        # Leverage
        if 'rank_proj' in row and 'rank_own' in row:
            if row['rank_proj'] > 0 and row['rank_own'] > 0:
                diff = row['rank_own'] - row['rank_proj']
                if diff > 15:
                    score += 15
                    reasons.append("💎 High Leverage")
                elif diff < -10:
                    score -= 10
                    reasons.append("⚠️ Chalk Trap")

        # Game Script
        if self.sport == "NFL" and 'position' in row:
            if abs(self.spread) > 7 and "WR" in str(row['position']):
                score += 7
                reasons.append("📜 Blowout Script")

        # Prop
        if 'prop_line' in row and 'proj_pts' in row:
            if row['prop_line'] > 0:
                edge = ((row['proj_pts'] - row['prop_line']) / row['proj_pts']) * 100
                if edge > 15:
                    score += 15
                    reasons.append(f"💰 Prop Smash ({edge:.1f}%)")

        # Ceiling
        if 'ceiling' in row and 'proj_pts' in row:
            if row['ceiling'] > row['proj_pts'] * 1.5:
                score += 10
                reasons.append("🔥 Massive Ceiling")

        score = max(0, min(100, score))
        verdict = " | ".join(reasons) if reasons else "Neutral"
        return score, verdict

# ==========================================
# 3. DATA CLEANING
# ==========================================
def standardize_columns(df):
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("-", "_")

    mapping = {
        'name': ['player', 'athlete', 'full_name'],
        'proj_pts': ['projection','proj','fpts','median'],
        'ownership': ['own','projected_ownership'],
        'salary': ['cost','sal','price'],
        'prop_line': ['line','prop','ou','total','strike'],
        'position': ['pos'],
        'team': ['tm','squad'],
        'ceiling': ['ceil','max_pts']
    }

    rename = {}
    for std, alts in mapping.items():
        for col in df.columns:
            if any(a in col for a in alts):
                rename[col] = std

    df = df.rename(columns=rename)

    # numeric cleaning
    numeric_cols = ['proj_pts','ownership','salary','prop_line','ceiling']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("$","")
                .str.replace(",","")
                .str.replace("%","")
            )
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df

def process_and_analyze(files, sport, spread, total):
    master = pd.DataFrame()

    for file in files:
        try:
            df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
            df = standardize_columns(df)

            # Remove list/dict garbage columns
            bad_cols = [
                c for c in df.columns
                if df[c].apply(lambda x: isinstance(x, (list, dict, np.ndarray))).any()
            ]
            df = df.drop(columns=bad_cols)

            if master.empty:
                master = df
            else:
                if 'name' in df.columns and 'name' in master.columns:
                    master = master.merge(df, on="name", how="left")

        except Exception as e:
            st.error(f"File Load Error: {e}")

    # Titan scoring
    if 'proj_pts' in master.columns and 'ownership' in master.columns:
        master['rank_proj'] = master['proj_pts'].rank(ascending=False)
        master['rank_own'] = master['ownership'].rank(ascending=False)

        brain = TitanBrain(sport, spread, total)
        res = master.apply(lambda r: brain.evaluate_player(r), axis=1, result_type="expand")
        master['shark_score'] = res[0]
        master['reasoning'] = res[1]

    return master

# ==========================================
# 4. PLAYER POOL
# ==========================================
def get_player_pool(df, top_n_shark=25, top_n_value=15):
    if 'shark_score' not in df.columns:
        df['shark_score'] = 50.0
    if df.empty:
        return pd.DataFrame()

    pool_shark = df.nlargest(top_n_shark, 'shark_score')

    if 'proj_pts' in df.columns and 'salary' in df.columns:
        df['value'] = (df['proj_pts'] / df['salary']) * 1000
        pool_value = df[df['salary'] <= 5000].nlargest(top_n_value, 'value')
    else:
        pool_value = pd.DataFrame()

    final_pool = pd.concat([pool_shark, pool_value]).drop_duplicates(subset='name')
    return final_pool.sort_values('shark_score', ascending=False)

# ==========================================
# 5. OPTIMIZER (FIXED STACKING LOGIC)
# ==========================================
def optimize_lineup(df, config):

    site, sport, cap, num_lineups, target_col, use_correlation = (
        config['site'], config['sport'], config['cap'], config['num_lineups'],
        config['target_col'], config['use_correlation']
    )

    pool = df[df[target_col] > 0].copy().reset_index(drop=True)
    roster_size = 9 if sport == "NFL" else 8

    valid = []

    for L in range(num_lineups):
        prob = pulp.LpProblem("TitanOpt", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("P", pool.index, lowBound=0, upBound=1, cat='Binary')

        # Objective
        prob += pulp.lpSum(pool.loc[i, target_col] * x[i] for i in pool.index)

        # Salary
        prob += pulp.lpSum(pool.loc[i, 'salary'] * x[i] for i in pool.index) <= cap

        # Roster size
        prob += pulp.lpSum(x[i] for i in pool.index) == roster_size

        # NFL rules
        if sport == "NFL":
            qbs = pool[pool['position'].str.contains("QB", na=False)]
            dsts = pool[pool['position'].str.contains("DST|DEF", na=False)]

            prob += pulp.lpSum(x[i] for i in qbs.index) == 1
            prob += pulp.lpSum(x[i] for i in dsts.index) == 1

            # Fixed stacking logic
            if use_correlation:
                for qb_i in qbs.index:
                    same_team = pool[
                        (pool['team'] == pool.loc[qb_i, 'team']) &
                        (pool['position'].str.contains("WR|TE", na=False))
                    ]
                    prob += pulp.lpSum(x[i] for i in same_team.index) >= x[qb_i]

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if prob.status != pulp.LpStatusOptimal:
            break

        chosen = [i for i in pool.index if x[i].value() == 1]

        lineup = pool.loc[chosen].copy()
        lineup['Lineup_ID'] = L + 1
        valid.append(lineup)

        # prevent duplicate lineups
        prob += pulp.lpSum(x[i] for i in chosen) <= roster_size - 1

    return pd.concat(valid) if valid else None

# ==========================================
# 6. UTILS
# ==========================================
def fetch_rotowire_news(sport):
    url = {
        "NFL": "https://www.rotowire.com/rss/news.htm?sport=nfl",
        "NBA": "https://www.rotowire.com/rss/news.htm?sport=nba"
    }.get(sport, "https://www.rotowire.com/rss/news.htm?sport=nfl")

    feed = feedparser.parse(url)
    return [{"title": x.title, "summary": x.summary} for x in feed.entries[:5]]

def get_csv_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a download="titan_lineups.csv" href="data:file/csv;base64,{b64}">📥 Download CSV</a>'

# ==========================================
# 7. STREAMLIT UI
# ==========================================
st.sidebar.title("🦁 TITAN APEX CORE")

sport = st.sidebar.selectbox("Sport", ["NFL","NBA","MLB","NHL"])
site = st.sidebar.selectbox("Sportsbook", ["DraftKings","FanDuel","Yahoo","PrizePicks","Underdog"])

default_cap = 60000 if site == "FanDuel" else (200 if site=="Yahoo" else 50000)
st.session_state['cap'] = default_cap

tabs = st.tabs(["1. Data", "2. Player Pool", "3. Optimizer", "4. Props", "5. News"])

# === TAB 1 ===
with tabs[0]:
    st.header("Upload Data")
    files = st.file_uploader("Upload CSV/XLSX files", accept_multiple_files=True)

    spread = st.number_input("Vegas Spread", -20.0, 20.0, 0.0)
    total = st.number_input("Vegas Total", 100, 260, 210)

    if st.button("Process"):
        df = process_and_analyze(files, sport, spread, total)
        st.session_state['master'] = df
        st.success("Processed Successfully!")

# === TAB 2 ===
with tabs[1]:
    df = st.session_state['master']
    if df.empty:
        st.warning("No data.")
    else:
        pool = get_player_pool(df)
        st.dataframe(pool)

# === TAB 3 ===
with tabs[2]:
    df = st.session_state['master']
    if df.empty:
        st.warning("Upload data.")
    else:
        num = st.number_input("Lineups", 1, 150, 10)
        cap = st.number_input("Cap", value=st.session_state['cap'])
        target = st.selectbox("Target", ["shark_score", "proj_pts"])
        corr = st.checkbox("Enable Stacking", True)

        if st.button("Optimize"):
            config = {
                "site": site,
                "sport": sport,
                "cap": cap,
                "num_lineups": num,
                "target_col": target,
                "use_correlation": corr
            }
            pool = get_player_pool(df)
            result = optimize_lineup(pool, config)
            if result is None:
                st.error("No valid lineup found.")
            else:
                st.dataframe(result)
                st.markdown(get_csv_download(result), unsafe_allow_html=True)

# === TAB 4 ===
with tabs[3]:
    df = st.session_state['master']
    if df.empty or 'prop_line' not in df.columns:
        st.warning("No prop data.")
    else:
        df['edge'] = ((df['proj_pts'] - df['prop_line']) / df['prop_line']) * 100
        df['pick'] = np.where(df['edge']>0,"OVER","UNDER")
        st.dataframe(df[['name','prop_line','proj_pts','edge','pick']])

# === TAB 5 ===
with tabs[4]:
    st.header("Live News")
    if st.button("Refresh"):
        st.cache_data.clear()
    for item in fetch_rotowire_news(sport):
        st.info(f"**{item['title']}**\n\n{item['summary']}")
