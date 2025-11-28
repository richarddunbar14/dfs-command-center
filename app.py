import streamlit as st
import pandas as pd
import numpy as np
import pulp
import requests
import sqlite3
import plotly.express as px
from datetime import datetime

# ==========================================
# ⚙️ 1. PROFESSIONAL UI CONFIGURATION (STOKASTIC STYLE)
# ==========================================
st.set_page_config(layout="wide", page_title="TITAN KRONOS", page_icon="⚡")

# Custom CSS for Professional Data Grid Look
st.markdown("""
<style>
    /* Main Background - Professional Dark Grey */
    .stApp { background-color: #121212; color: #ffffff; font-family: 'Roboto', sans-serif; }
    
    /* Sidebar - Darker contrast */
    section[data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #333; }
    
    /* Headers */
    h1, h2, h3 { font-family: 'Oswald', sans-serif; text-transform: uppercase; letter-spacing: 1px; color: #00e5ff; }
    
    /* Dataframes - High Density, "Excel-like" */
    .stDataFrame { border: 1px solid #333; }
    div[data-testid="stDataFrame"] div[role="grid"] { color: #eee; background-color: #1e1e1e; }
    
    /* Buttons - Teal/Cyan Accent (Like Stokastic) */
    .stButton>button { 
        width: 100%; border-radius: 4px; font-weight: 700; text-transform: uppercase;
        background-color: #00b8d4; color: #000; border: none; padding: 10px;
    }
    .stButton>button:hover { background-color: #00e5ff; box-shadow: 0 0 10px #00e5ff; }
    
    /* Custom Status Badges */
    .badge-value { background-color: #00c853; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; }
    .badge-fade { background-color: #d50000; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- GLOBAL SESSION STATE ---
if 'master_pool' not in st.session_state: st.session_state['master_pool'] = pd.DataFrame()
if 'bankroll' not in st.session_state: st.session_state['bankroll'] = 1000.0

# API KEYS (Hardcoded for immediate access)
SPORTSDATAIO_KEY = "dda563b328d34b80a38c26cd43223614"
ODDS_API_KEY = "a31f629075c27927fe99b097b51e1717"

# ==========================================
# 📡 2. AUTO-DATA ENGINE (AUTO-PULL)
# ==========================================

@st.cache_data(ttl=3600) # Cache data for 1 hour to prevent API limits
def fetch_api_data(sport):
    """
    Automatically pulls data on load. 
    Combines SportsDataIO (Teams/News) + TheOddsAPI (Props).
    """
    master_data = []
    
    # 1. FETCH ODDS (TheOddsAPI)
    sport_keys = {'NFL': 'americanfootball_nfl', 'NBA': 'basketball_nba', 'MLB': 'baseball_mlb', 'NHL': 'icehockey_nhl'}
    key = sport_keys.get(sport)
    
    if key:
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{key}/events/upcoming/odds"
            params = {
                'apiKey': ODDS_API_KEY,
                'regions': 'us',
                'markets': 'player_points,player_assists,player_rebounds',
                'oddsFormat': 'american'
            }
            res = requests.get(url, params=params)
            if res.status_code == 200:
                data = res.json()
                for event in data:
                    game_info = f"{event.get('home_team')} vs {event.get('away_team')}"
                    commence = event.get('commence_time')
                    
                    for book in event.get('bookmakers', []):
                        if book['key'] == 'draftkings': # Priority Source
                            for market in book.get('markets', []):
                                for outcome in market.get('outcomes', []):
                                    master_data.append({
                                        'Name': outcome['description'],
                                        'Team': 'N/A', # OddsAPI doesn't link players to teams easily
                                        'Opp': 'N/A',
                                        'Position': 'FLEX',
                                        'Salary': 0, # To be filled by CSV
                                        'Projection': 0.0, # To be filled by CSV or Algo
                                        'Prop Line': outcome.get('point', 0),
                                        'Odds': outcome.get('price', 0),
                                        'Prop Type': market['key'],
                                        'Game': game_info,
                                        'Time': commence,
                                        'Sport': sport,
                                        'Source': 'API_LIVE'
                                    })
        except Exception as e:
            print(f"Odds API Error: {e}")

    # 2. FETCH SPORTSDATAIO (News/Rosters)
    # Note: Free tier is restricted. We attempt to get News/Standings to enrich data.
    try:
        sd_url = f"https://api.sportsdata.io/v3/{sport.lower()}/scores/json/News?key={SPORTSDATAIO_KEY}"
        sd_res = requests.get(sd_url)
        # We use this primarily to verify the API connection is live
        # Real projection data usually requires paid tier, but we check anyway.
    except: pass

    return pd.DataFrame(master_data)

class DataRefinery:
    @staticmethod
    def normalize_csv(df, sport):
        """Standardizes uploaded CSVs (DK/FD/Rotowire)."""
        df.columns = df.columns.astype(str).str.upper().str.strip()
        std = pd.DataFrame()
        
        # Mappings
        col_map = {
            'Name': ['PLAYER', 'NAME', 'WHO', 'PLAYER NAME'],
            'Salary': ['SAL', 'SALARY', 'CAP', 'COST'],
            'Projection': ['FPTS', 'PROJ', 'AVG FPTS', 'PTS', 'FC PROJ'],
            'Position': ['POS', 'POSITION', 'ROSTER POSITION'],
            'Team': ['TEAM', 'TM', 'SQUAD'],
            'Opp': ['OPP', 'OPPONENT', 'VS'],
            'Game': ['GAME INFO', 'MATCHUP', 'GAME']
        }
        
        for target, sources in col_map.items():
            for s in sources:
                if s in df.columns:
                    if target in ['Salary', 'Projection']:
                        std[target] = pd.to_numeric(df[s], errors='coerce').fillna(0)
                    else:
                        std[target] = df[s].astype(str).str.strip()
                    break
        
        # Fill missing
        if 'Name' not in std.columns: return pd.DataFrame()
        for c in ['Salary', 'Projection']:
            if c not in std.columns: std[c] = 0.0
        if 'Position' not in std.columns: std['Position'] = 'FLEX'
        
        std['Sport'] = sport
        std['Source'] = 'CSV_UPLOAD'
        return std

# ==========================================
# 🧠 3. ALGORITHMIC BRAIN
# ==========================================

def calculate_value(row):
    """Calculates DFS Value (Pts/$) and Prop Value (Diff from Line)."""
    # DFS Value
    if row['Salary'] > 0:
        val = (row['Projection'] / row['Salary']) * 1000
    else:
        val = 0
        
    # Prop Edge
    edge = 0
    pick = "-"
    if row.get('Prop Line', 0) > 0 and row['Projection'] > 0:
        edge = (row['Projection'] - row['Prop Line']) / row['Prop Line']
        pick = "OVER" if edge > 0 else "UNDER"
        
    return pd.Series([val, edge * 100, pick], index=['Value', 'Edge %', 'Pick'])

# ==========================================
# 🏭 4. OPTIMIZER ENGINE (STRICT RULES)
# ==========================================

def optimize_lineup(df, config):
    # Filter by Sport and Slate
    pool = df[df['Sport'] == config['sport']].copy()
    
    # 1. Slate Logic (Filter games if selected)
    if config['slate'] and 'Game' in pool.columns:
        pool = pool[pool['Game'].isin(config['slate'])]
        
    pool = pool[pool['Projection'] > 0].reset_index(drop=True)
    if pool.empty: return None

    # Constants
    cap = 50000 if config['site'] == 'DK' else 60000
    if config['site'] == 'Yahoo': cap = 200
    
    # Roster Rules
    size = 8
    constraints = []
    
    if config['sport'] == 'NFL':
        size = 9
        if config['site'] == 'DK':
            constraints = [('QB', 1, 1), ('DST', 1, 1), ('RB', 2, 3), ('WR', 3, 4), ('TE', 1, 2)]
        else:
            constraints = [('QB', 1, 1), ('DEF', 1, 1), ('RB', 2, 3), ('WR', 3, 4), ('TE', 1, 2)]
    
    elif config['sport'] == 'NBA':
        size = 8 if config['site'] == 'DK' else 9
        if config['site'] == 'DK':
            constraints = [('PG', 1, 3), ('SG', 1, 3), ('SF', 1, 3), ('PF', 1, 3), ('C', 1, 2)]
    
    # Solver
    lineups = []
    for i in range(config['count']):
        prob = pulp.LpProblem("Titan", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("p", pool.index, cat='Binary')
        
        # Sim Randomness
        vol = 0.10 if config['randomness'] else 0.0
        pool['sim'] = pool['Projection'] * np.random.normal(1.0, vol, len(pool))
        
        prob += pulp.lpSum([pool.loc[p, 'sim'] * x[p] for p in pool.index])
        prob += pulp.lpSum([pool.loc[p, 'Salary'] * x[p] for p in pool.index]) <= cap
        prob += pulp.lpSum([x[p] for p in pool.index]) == size
        
        # Position Constraints (Regex)
        for role, min_q, max_q in constraints:
            idx = pool[pool['Position'].str.contains(role, regex=True, na=False)].index
            if not idx.empty:
                prob += pulp.lpSum([x[p] for p in idx]) >= min_q
                prob += pulp.lpSum([x[p] for p in idx]) <= max_q
                
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == 1:
            sel = [p for p in pool.index if x[p].varValue == 1]
            lu = pool.loc[sel].copy()
            lu['Lineup_ID'] = i+1
            lineups.append(lu)
            # Uniqueness Constraint
            prob += pulp.lpSum([x[p] for p in sel]) <= size - 1
            
    return pd.concat(lineups) if lineups else None

# ==========================================
# 🖥️ 5. MASTER DASHBOARD
# ==========================================

# SIDEBAR CONTROLS
with st.sidebar:
    st.title("TITAN KRONOS")
    st.caption("v10.0 | God Mode")
    
    # Global Settings
    sport = st.selectbox("SPORT", ["NFL", "NBA", "MLB", "NHL", "CFB"])
    site = st.selectbox("SITE", ["DK", "FD", "Yahoo", "PrizePicks"])
    
    # Bankroll
    st.metric("BANKROLL", f"${st.session_state['bankroll']:,.2f}")
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()

# MAIN LAYOUT
st.markdown(f"## {sport} COMMAND CENTER")

# 1. AUTO-DATA INGESTION
api_data = fetch_api_data(sport)
file_data = pd.DataFrame()

# CSV Uploader (Seamlessly Blends)
with st.expander("📂 Import Custom Projections (Rotowire/Stokastic CSV)", expanded=True):
    uploaded = st.file_uploader("Drag & Drop CSVs", accept_multiple_files=True)
    if uploaded:
        ref = DataRefinery()
        for f in uploaded:
            try:
                # Handle encoding issues
                try: df_raw = pd.read_csv(f, encoding='utf-8-sig')
                except: df_raw = pd.read_excel(f)
                
                clean = ref.normalize_csv(df_raw, sport)
                file_data = pd.concat([file_data, clean])
            except Exception as e:
                st.error(f"Error loading {f.name}: {e}")

# MERGE LOGIC: Combine API + CSV
# If we have CSV data, use it as base. If we have API props, map them to the players.
master = file_data.copy() if not file_data.empty else api_data.copy()

if not master.empty and not api_data.empty and not file_data.empty:
    # Merge Props from API into CSV Projections
    # We strip names to ensure matches (e.g. "Patrick Mahomes II" vs "Patrick Mahomes")
    master['match_name'] = master['Name'].str.lower().str.replace(r'[^\w\s]', '')
    api_data['match_name'] = api_data['Name'].str.lower().str.replace(r'[^\w\s]', '')
    
    # Merge
    merged = master.merge(
        api_data[['match_name', 'Prop Line', 'Odds', 'Prop Type']], 
        on='match_name', 
        how='left', 
        suffixes=('', '_api')
    )
    # Fill Coalesce
    merged['Prop Line'] = merged['Prop Line_api'].fillna(merged['Prop Line'])
    master = merged.drop(columns=['match_name', 'Prop Line_api'])

# CALCULATE METRICS
if not master.empty:
    # Run Math Engine
    metrics = master.apply(calculate_value, axis=1)
    master = pd.concat([master, metrics], axis=1)
    st.session_state['master_pool'] = master

    # --- TABS FOR UI ---
    tab_dfs, tab_props, tab_slate = st.tabs(["🏰 DFS WAR ROOM", "🎯 PROP SNIPER", "🗓️ SLATE FILTER"])

    # === TAB 1: DFS OPTIMIZER ===
    with tab_dfs:
        st.markdown("### 🏰 LINEUP BUILDER")
        
        # Grid View
        cols_to_show = ['Name', 'Position', 'Team', 'Salary', 'Projection', 'Value', 'Game']
        # Styling the dataframe to look like a tool
        st.dataframe(
            master[cols_to_show].sort_values('Value', ascending=False).style.background_gradient(subset=['Value'], cmap='Greens'),
            use_container_width=True,
            height=300
        )
        
        col_build, col_res = st.columns([1, 2])
        
        with col_build:
            st.markdown("#### ⚙️ Build Config")
            num_lineups = st.slider("Count", 1, 150, 10)
            randomness = st.checkbox("Sim Randomness (GPP)", value=True)
            
            # Slate Selection (Dynamic)
            if 'Game' in master.columns:
                unique_games = master['Game'].dropna().unique()
                selected_games = st.multiselect("Select Slate", unique_games, default=unique_games)
            else:
                selected_games = []

            if st.button("⚡ GENERATE OPTIMAL"):
                config = {
                    'sport': sport, 'site': site, 'mode': 'Classic', 
                    'count': num_lineups, 'randomness': randomness, 
                    'slate': selected_games, 'bans': []
                }
                res = optimize_lineup(master, config)
                st.session_state['results'] = res

        with col_res:
            if 'results' in st.session_state and st.session_state['results'] is not None:
                res = st.session_state['results']
                
                # Stats
                avg_proj = res.groupby('Lineup_ID')['Projection'].sum().mean()
                st.info(f"Generated {len(res['Lineup_ID'].unique())} Lineups | Avg Proj: {avg_proj:.2f}")
                
                # Show Lineups
                st.dataframe(res[['Lineup_ID', 'Position', 'Name', 'Salary', 'Projection']], use_container_width=True, height=400)
                
                # Download
                csv = res.to_csv(index=False).encode()
                st.download_button("📥 Export CSV", csv, "titan_lineups.csv")

    # === TAB 2: PROP SNIPER ===
    with tab_props:
        st.markdown("### 🎯 +EV FINDER")
        
        # Filter only rows with Props
        props = master[master['Prop Line'] > 0].copy()
        
        if props.empty:
            st.warning("No prop lines found via API or CSV.")
        else:
            # Sort by Edge
            best_props = props.sort_values('Edge %', ascending=False)
            
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.dataframe(
                    best_props[['Name', 'Prop Type', 'Prop Line', 'Projection', 'Pick', 'Edge %', 'Odds']],
                    use_container_width=True,
                    height=500
                )
            
            with col_b:
                st.markdown("#### 💎 Top Plays")
                top_5 = best_props.head(5)
                for _, row in top_5.iterrows():
                    color = "#00c853" if row['Pick'] == "OVER" else "#d50000"
                    st.markdown(f"""
                    <div style="background:#1e1e1e; border-left:4px solid {color}; padding:10px; margin-bottom:5px; border-radius:4px;">
                        <div style="font-weight:bold; font-size:14px;">{row['Name']}</div>
                        <div style="display:flex; justify-content:space-between; font-size:12px; color:#aaa;">
                            <span>{row['Prop Type']}</span>
                            <span style="color:{color}; font-weight:bold;">{row['Pick']} {row['Prop Line']}</span>
                        </div>
                        <div style="text-align:right; font-size:11px; margin-top:2px;">Edge: {row['Edge %']:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

else:
    st.info("👋 Welcome to Titan Kronos. Upload a CSV or wait for API Sync to begin.")
