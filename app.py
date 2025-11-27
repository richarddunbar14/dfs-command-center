import streamlit as st
import pandas as pd
import numpy as np
import pulp
import base64
import math
import feedparser

# ==========================================
# STREAMLIT CONFIG
# ==========================================
st.set_page_config(layout="wide", page_title="Titan Command", page_icon="💥")

st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    .reasoning-text { font-size: 12px; color: #94a3b8; font-style: italic; border-left: 2px solid #3b82f6; padding-left: 8px; }
    .stButton>button { width: 100%; border-radius: 6px; font-weight: bold; background-color: #3b82f6; border: none; }
</style>
""", unsafe_allow_html=True)

if "master" not in st.session_state:
    st.session_state["master"] = pd.DataFrame()

# ==========================================
# TITAN BRAIN
# ==========================================
class TitanBrain:
    def __init__(self, sport, spread=0, total=0):
        self.sport = sport
        self.spread = spread
        self.total = total

    def evaluate_player(self, row):
        reasons = []
        score = 50.0  # baseline

        # Leverage logic
        if "rank_proj" in row and "rank_own" in row:
            diff = row["rank_own"] - row["rank_proj"]
            if diff > 15:
                score += 15
                reasons.append("💎 High Leverage Advantage")
            elif diff < -10:
                score -= 10
                reasons.append("⚠️ Overowned Risk")

        # Game script logic
        if self.sport == "NFL":
            if "WR" in str(row.get("position", "")) and abs(self.spread) > 7:
                score += 7
                reasons.append("📜 Garbage Time Boost")

        # Prop logic
        if "prop_line" in row and "proj_pts" in row:
            if row["prop_line"] > 0:
                edge = ((row["proj_pts"] - row["prop_line"]) / row["proj_pts"]) * 100
                if edge > 15:
                    score += 15
                    reasons.append(f"💰 Prop Edge {edge:.1f}%")

        # Ceiling
        if "ceiling" in row and "proj_pts" in row:
            if row["ceiling"] > row["proj_pts"] * 1.5:
                score += 10
                reasons.append("🔥 High Ceiling Indicator")

        final_score = max(0, min(100, score))
        verdict = " | ".join(reasons) if reasons else "Neutral Profile"
        return final_score, verdict


# ==========================================
# CLEANING / FILE PROCESSING
# ==========================================
def standardize_columns(df):
    # 🔥 FIX: Ensure all column names are strings
    df.columns = [str(c) for c in df.columns]

    df.columns = (
        df.columns
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("$", "")
        .str.replace(",", "")
        .str.replace("%", "")
    )

    mapping = {
        "name": ["player", "full_name", "athlete"],
        "proj_pts": ["proj", "projection", "points", "median", "fpts"],
        "ownership": ["own", "projected_ownership"],
        "salary": ["salary", "sal", "cost", "price"],
        "prop_line": ["line", "prop", "total", "strike"],
        "position": ["pos", "roster_position"],
        "team": ["tm", "team"],
        "ceiling": ["ceil", "max_pts"],
    }

    rename_dict = {}
    for std, alts in mapping.items():
        for col in df.columns:
            if any(a in col for a in alts):
                rename_dict[col] = std

    df = df.rename(columns=rename_dict)

    # Numeric cleansing
    numeric_cols = ["proj_pts", "ownership", "salary", "prop_line", "ceiling"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace("$", "")
                .str.replace(",", "")
                .str.replace("%", "")
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Drop any columns full of dict/list/DataFrame objects
    drop_cols = []
    for c in df.columns:
        if df[c].apply(lambda x: isinstance(x, (dict, list, pd.DataFrame, np.ndarray))).any():
            drop_cols.append(c)

    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def process_and_analyze(files, sport, spread, total):
    master = pd.DataFrame()

    for file in files:
        try:
            df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
            df = standardize_columns(df)

            if master.empty:
                master = df
            else:
                if "name" in master.columns and "name" in df.columns:
                    master = master.merge(df, on="name", how="left")

        except Exception as e:
            st.error(f"File Load Error: {e}")

    # Titan scoring
    if "proj_pts" in master.columns and "ownership" in master.columns:
        master["rank_proj"] = master["proj_pts"].rank(ascending=False)
        master["rank_own"] = master["ownership"].rank(ascending=False)

        brain = TitanBrain(sport, spread, total)
        res = master.apply(lambda r: brain.evaluate_player(r), axis=1, result_type="expand")
        master["shark_score"] = res[0]
        master["reasoning"] = res[1]

    return master


# ==========================================
# PLAYER POOL
# ==========================================
def get_player_pool(df, n_shark=25, n_value=15):
    if df.empty:
        return pd.DataFrame()

    if "shark_score" not in df.columns:
        df["shark_score"] = 50.0

    top_shark = df.nlargest(n_shark, "shark_score")

    if "proj_pts" in df.columns and "salary" in df.columns:
        df["value"] = (df["proj_pts"] / df["salary"]) * 1000
        top_value = df[df["salary"] <= 5000].nlargest(n_value, "value")
    else:
        top_value = pd.DataFrame()

    pool = pd.concat([top_shark, top_value]).drop_duplicates(subset="name")
    return pool.sort_values("shark_score", ascending=False)


# ==========================================
# OPTIMIZER (FIXED CBC + STACKING)
# ==========================================
def optimize_lineup(df, config):
    site, sport, cap, n_lineups, target_col, use_corr = (
        config["site"],
        config["sport"],
        config["cap"],
        config["n"],
        config["target_col"],
        config["use_corr"],
    )

    df = df[df[target_col] > 0].reset_index(drop=True)

    roster_size = 9 if sport == "NFL" else 8
    final = []

    for L in range(n_lineups):
        prob = pulp.LpProblem("Titan", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("P", df.index, 0, 1, cat="Binary")

        # Objective
        prob += pulp.lpSum(df.loc[i, target_col] * x[i] for i in df.index)

        # Salary
        prob += pulp.lpSum(df.loc[i, "salary"] * x[i] for i in df.index) <= cap

        # Roster count
        prob += pulp.lpSum(x[i] for i in df.index) == roster_size

        if sport == "NFL":
            qbs = df[df["position"].str.contains("QB", na=False)]
            dst = df[df["position"].str.contains("DST|DEF", na=False)]

            prob += pulp.lpSum(x[i] for i in qbs.index) == 1
            prob += pulp.lpSum(x[i] for i in dst.index) == 1

            # Stacking
            if use_corr:
                for qb in qbs.index:
                    same_team = df[
                        (df["team"] == df.loc[qb, "team"]) &
                        (df["position"].str.contains("WR|TE"))
                    ]
                    prob += pulp.lpSum(x[i] for i in same_team.index) >= x[qb]

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if prob.status != 1:
            break

        lineup = df.loc[[i for i in df.index if x[i].value() == 1]].copy()
        lineup["Lineup_ID"] = L + 1
        final.append(lineup)

        # Prevent repetition
        prob += pulp.lpSum(x[i] for i in lineup.index) <= roster_size - 1

    return pd.concat(final) if final else None


# ==========================================
# NEWS
# ==========================================
def fetch_news(sport):
    urls = {
        "NFL": "https://www.rotowire.com/rss/news.htm?sport=nfl",
        "NBA": "https://www.rotowire.com/rss/news.htm?sport=nba",
    }
    feed = feedparser.parse(urls.get(sport, urls["NFL"]))
    return [{"title": e.title, "summary": e.summary} for e in feed.entries[:5]]


# ==========================================
# CSV EXPORT
# ==========================================
def get_csv_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="lineups.csv">📥 Download CSV</a>'


# ==========================================
# STREAMLIT UI
# ==========================================
st.sidebar.title("Titan APEX")

sport = st.sidebar.selectbox("Sport", ["NFL", "NBA", "MLB", "NHL"])
site = st.sidebar.selectbox("Site", ["DraftKings", "FanDuel", "Yahoo", "PrizePicks", "Underdog"])

cap = 60000 if site == "FanDuel" else (200 if site == "Yahoo" else 50000)

tabs = st.tabs(["Data", "Pool", "Optimizer", "Props", "News"])

# DATA TAB
with tabs[0]:
    st.header("Upload Data")
    files = st.file_uploader("Upload CSV or Excel", accept_multiple_files=True)

    spread = st.number_input("Vegas Spread", -20.0, 20.0, 0.0)
    total = st.number_input("Vegas Total", 100, 260, 210)

    if st.button("Process"):
        df = process_and_analyze(files, sport, spread, total)
        st.session_state["master"] = df
        st.success("Data Loaded")

# POOL TAB
with tabs[1]:
    df = st.session_state["master"]
    if df.empty:
        st.warning("No data.")
    else:
        pool = get_player_pool(df)
        st.dataframe(pool)

# OPTIMIZER TAB
with tabs[2]:
    df = st.session_state["master"]
    if df.empty:
        st.warning("No data.")
    else:
        n = st.number_input("Lineups", 1, 150, 10)
        target = st.selectbox("Target Metric", ["shark_score", "proj_pts"])
        corr = st.checkbox("Enable Correlation", True)

        if st.button("Optimize"):
            cfg = {
                "site": site,
                "sport": sport,
                "cap": cap,
                "n": n,
                "target_col": target,
                "use_corr": corr,
            }
            res = optimize_lineup(pool, cfg)
            if res is None:
                st.error("No lineups found.")
            else:
                st.dataframe(res)
                st.markdown(get_csv_download(res), unsafe_allow_html=True)

# NEWS TAB
with tabs[4]:
    st.header("Latest News")
    for item in fetch_news(sport):
        st.info(f"**{item['title']}**\n\n{item['summary']}")
