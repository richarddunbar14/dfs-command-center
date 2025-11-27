import streamlit as st
import pandas as pd
import numpy as np
import pulp
import base64
import io
import requests
import feedparser
import nfl_data_py as nfl

# Try importing pybaseball, handle error if missing
try:
    from pybaseball import batting_stats, pitching_stats
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False

# ==========================================
# ⚙️ 1. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(layout="wide", page_title="TITAN COMMAND: INFINITY", page_icon="🧬")

st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    
    /* Live Indicators */
    .live-dot { height: 10px; width: 10px; background-color: #ef4444; border-radius: 50%; display: inline-block; animation: blink 2s infinite; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { color: #38bdf8; font-size: 26px; font-weight: 800; }
    
    /* Tables */
    [data-testid="stDataFrame"] { border: 1px solid #1e293b; border-radius: 8px; }
    
    /* Buttons */
    .stButton>button { background-color: #3b82f6; color: white; border: none; font-weight: 700; border-radius: 6px; }
    .stButton>button:hover { background-color: #2563eb; }
    
    /* Badges */
    .stat-badge { background-color: #8b5cf6; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 📡 2. DEEP DATA UPLINKS (STATMUSE / B-REF)
# ==========================================

@st.cache_data(ttl=3600)
def fetch_mlb_stats(stat_type="Batting"):
    """
    Uplink to Baseball Reference via PyBaseball.
    """
    if not PYBASEBALL_AVAILABLE:
        return pd.DataFrame()
    
    try:
        if stat_type == "Batting":
            # Fetches current season batting stats
            data = batting_stats(2024, qual=50) # min 50 ABs
        else:
            data = pitching_stats(2024, qual=10)
        return data
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_nfl_ngs(stat_type="Passing"):
    """
    Uplink to NFL Next Gen Stats (The "StatMuse" Data).
    """
    try:
        if stat_type == "Passing":
            return nfl.import_ngs_data(stat_type='passing', years=[2024])
        elif stat_type == "Rushing":
            return nfl.import_ngs_data(stat_type='rushing', years=[2024])
        elif stat_type == "Receiving":
            return nfl.import_ngs_data(stat_type='receiving', years=[2024])
    except:
        return pd.DataFrame()

# ==========================================
# 📡 3. LIVE INTEL (NEWS & WEATHER)
# ==========================================

STADIUMS = {
    "BUF (Highmark)": {"lat": 42.7738, "lon": -78.7870},
    "GB (Lambeau)":   {"lat": 44.5013, "lon": -88.0622},
    "CHI (Soldier)":  {"lat": 41.8623, "lon": -87.6167},
    "KC (Arrowhead)": {"lat": 39.0489, "lon": -94.4839},
    "CLE (FirstEnergy)": {"lat": 41.5061, "lon": -81.6995},
    "DEN (Mile High)": {"lat": 39.7439, "lon": -105.0201}
}

@st.cache_data(ttl=900)
def get_live_weather(stadium_name):
    coords = STADIUMS.get(stadium_name)
    if not coords: return None
    url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&current_weather=true&temperature_unit=fahrenheit&windspeed_unit=mph"
    try: return requests.get(url).json()['current_weather']
    except: return None

@st.cache_data(ttl=300)
def fetch_rotowire_news(sport):
    rss_urls = {
        "NFL": "https://www.rotowire.com/rss/news.htm?sport=nfl",
        "NBA": "https://www.rotowire.com/rss/news.htm?sport=nba",
        "MLB": "https://www.rotowire.com/rss/news.htm?sport=mlb",
        "NHL": "https://www.rotowire.com/rss/news.htm?sport=nhl"
    }
    if sport not in rss_urls: return []
    feed = feedparser.parse(rss_urls[sport])
    return [{"title": x.title, "summary": x.summary} for x in feed.entries[:8]]

# ==========================================
# 🛠️ 4. CORE LOGIC (TRANSLATORS & OPTIMIZERS)
# ==========================================

def standardize_columns(df):
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('-', '_').str.replace('%', '')
    column_map = {
        'name': ['player', 'athlete', 'full_name', 'name_id', 'player_name'],
        'team': ['squad', 'tm', 'team_id'],
        'position': ['pos', 'position_id', 'roster_position'],
        'proj_pts': ['projection', 'proj', 'fpts', 'fantasy_points', 'median'],
        'salary': ['cost', 'sal', 'price'],
        'prop_line': ['line', 'prop', 'ou', 'total', 'over_under', 'strike'],
        'opp_rank': ['dvp', 'opp_rank', 'defense_rank'],
    }
    renamed_cols = {}
    for standard, variations in column_map.items():
        if standard not in df.columns:
            for v in variations:
                match = next((c for c in df.columns if v in c), None)
                if match:
                    renamed_cols[match] = standard
                    break
    if renamed_cols: df = df.rename(columns=renamed_cols)
    
    # Numeric Cleanup
    for c in ['proj_pts', 'salary', 'prop_line', 'opp_rank']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
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
            
            # Platform Detection
            fname = file.name.lower()
            if 'pick6' in fname: plat = "Pick6"
            elif 'prize' in fname: plat = "PrizePicks"
            elif 'underdog' in fname: plat = "Underdog"
            elif 'salary' in df.columns: plat = "DFS_Projection"
            else: plat = "Generic"
            
            df['platform'] = plat
            
            if plat == "DFS_Projection":
                if master_df.empty: master_df = df
                else:
                    cols = [c for c in df.columns if c not in master_df.columns or c == 'name']
                    if 'name' in df.columns: master_df = master_df.merge(df[cols], on='name', how='left')
            else:
                prop_dfs.append(df)
        except: pass
    return master_df, prop_dfs

def run_shark_engine(df, config):
    df['notes'] = ""
    df['shark_score'] = 50.0 
    
    # Value Boost
    if 'salary' in df.columns and 'proj_pts' in df.columns:
        df['value'] = np.where(df['salary']>0, (df['proj_pts']/df['salary'])*1000, 0)
        df['shark_score'] += (df['value'] - 4.5) * 8
        
    # Weather Impact (Live Uplink)
    if config.get('wind_impact'):
        if 'position' in df.columns:
            mask = df['position'].str.contains('QB|WR|K', na=False)
            df.loc[mask, 'shark_score'] -= 15
            df.loc[mask, 'notes'] += "🌪️ Wind Fade. "
    return df

def optimize_lineup(df, salary_cap, sport, num_lineups, target_col):
    valid_lineups = []
    df = df[(df['proj_pts'] > 0) & (df['salary'] > 0)].reset_index(drop=True)
    
    for _ in range(num_lineups):
        prob = pulp.LpProblem("DFS", pulp.LpMaximize)
        player_vars = pulp.LpVariable.dicts("P", df.index, cat='Binary')
        
        prob += pulp.lpSum([df.loc[i, target_col] * player_vars[i] for i in df.index])
        prob += pulp.lpSum([df.loc[i, 'salary'] * player_vars[i] for i in df.index]) <= salary_cap
        
        if sport == "NFL":
            prob += pulp.lpSum([player_vars[i] for i in df.index]) == 9
            qbs = df[df['position'].str.contains('QB', na=False)]
            prob += pulp.lpSum([player_vars[i] for i in qbs.index]) == 1
        elif sport == "NBA":
            prob += pulp.lpSum([player_vars[i] for i in df.index]) == 8
            
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            selected = [i for i in df.index if player_vars[i].varValue == 1]
            lineup = df.loc[selected].copy()
            lineup['Lineup_ID'] = _ + 1
            valid_lineups.append(lineup)
            prob += pulp.lpSum([player_vars[i] for i in selected]) <= len(selected) - 1
        else: break
    return pd.concat(valid_lineups) if valid_lineups else None

def get_csv_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="lineups.csv" class="stButton">📥 Download CSV</a>'

# ==========================================
# 🖥️ 5. MAIN UI
# ==========================================
st.sidebar.title("🧬 TITAN COMMAND")
sport = st.sidebar.selectbox("Sport", ["NBA", "NFL", "MLB", "NHL"])
platform = st.sidebar.selectbox("Workbook", ["DraftKings DFS", "FanDuel DFS", "PrizePicks", "Underdog"])

if 'master' not in st.session_state: st.session_state['master'] = pd.DataFrame()
if 'props' not in st.session_state: st.session_state['props'] = []
if 'weather_alert' not in st.session_state: st.session_state['weather_alert'] = False

tabs = st.tabs(["1. 📡 Live Intel", "2. 📂 Omni-Ingest", "3. 🔬 Shark Lab", "4. 🏗️ Optimizer", "5. 💸 Prop Sniper"])

# --- TAB 1: LIVE INTEL ---
with tabs[0]:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader(f"🚨 {sport} News Wire")
        if st.button("🔄 Refresh News"): st.cache_data.clear()
        news = fetch_rotowire_news(sport)
        for item in news:
            st.info(f"**{item['title']}**\n\n{item['summary']}")
            
    with c2:
        st.subheader("🌪️ Weather Radar")
        if sport in ["NFL", "MLB"]:
            selected_stadium = st.selectbox("Stadium", list(STADIUMS.keys()))
            weather = get_live_weather(selected_stadium)
            if weather:
                w_speed = weather['windspeed']
                if w_speed > 15: 
                    st.error(f"⚠️ HIGH WIND: {w_speed} mph")
                    st.session_state['weather_alert'] = True
                else: 
                    st.success(f"Clear: {w_speed} mph")
                    st.session_state['weather_alert'] = False
        
        st.markdown("---")
        st.subheader("📊 Deep Stat Uplink")
        st.caption("Fetch advanced metrics from external databases.")
        
        if sport == "MLB":
            if PYBASEBALL_AVAILABLE:
                if st.button("⚾ Fetch Baseball Ref Stats (2024)"):
                    with st.spinner("Accessing Baseball Reference..."):
                        b_stats = fetch_mlb_stats("Batting")
                        st.dataframe(b_stats.head(20))
            else:
                st.warning("PyBaseball library required for B-Ref stats.")
                
        elif sport == "NFL":
            if st.button("🏈 Fetch Next Gen Stats"):
                with st.spinner("Accessing NFL.com NGS..."):
                    ngs = fetch_nfl_ngs("Passing")
                    st.dataframe(ngs.head(20))

# --- TAB 2: DATA INGEST ---
with tabs[1]:
    st.info("Upload Projections & Prop CSVs (Multi-File Supported)")
    files = st.file_uploader("Drop Files", accept_multiple_files=True)
    if files and st.button("🚀 Process"):
        m, p = process_files(files)
        st.session_state['master'] = m
        st.session_state['props'] = p
        st.success(f"Merged {len(m)} players.")

# --- TAB 3: SHARK LAB ---
with tabs[2]:
    df = st.session_state['master']
    if not df.empty:
        config = {'wind_impact': st.session_state['weather_alert']}
        df = run_shark_engine(df, config)
        st.session_state['master'] = df
        st.dataframe(df.sort_values('shark_score', ascending=False), use_container_width=True)

# --- TAB 4: OPTIMIZER ---
with tabs[3]:
    if "DFS" in platform:
        df = st.session_state['master']
        if not df.empty:
            c1, c2 = st.columns(2)
            num = c1.number_input("Lineups", 1, 20, 5)
            cap = c2.number_input("Cap", 50000)
            if st.button("⚡ Build"):
                res = optimize_lineup(df, cap, sport, num, 'shark_score')
                if res is not None:
                    st.success("Built!")
                    st.markdown(get_csv_download(res), unsafe_allow_html=True)
                    st.dataframe(res)
    else: st.info("Switch Platform to DFS.")

# --- TAB 5: PROP SNIPER ---
with tabs[4]:
    if "DFS" not in platform:
        props = st.session_state['props']
        active_props = [p for p in props if platform in p['platform'].iloc[0]]
        if active_props:
            df = active_props[0]
            # Simple merge logic for demo
            if 'proj_pts' in df.columns and 'prop_line' in df.columns:
                df['edge'] = ((df['proj_pts'] - df['prop_line']) / df['prop_line']) * 100
                st.dataframe(df.sort_values('edge', ascending=False), use_container_width=True)
        else: st.warning(f"No {platform} files.")
    else: st.info("Switch Platform to Betting.")
