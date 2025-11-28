import streamlit as st
import pandas as pd
import numpy as np
import pulp
import re
import time
import sqlite3
import base64
import requests
import plotly.express as px
from datetime import datetime

# 🛡️ SAFE IMPORT
try:
    from duckduckgo_search import DDGS
    HAS_AI = True
except ImportError:
    HAS_AI = False

# ==========================================
# ⚙️ 1. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="TITAN OMNI: V43.0", page_icon="🌌")

st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    .titan-card { background: #111; border: 1px solid #333; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #29b6f6; }
    div[data-testid="stDataFrame"] { border: 1px solid #333; border-radius: 5px; }
    .stButton>button { width: 100%; border-radius: 4px; font-weight: 800; text-transform: uppercase; background: linear-gradient(90deg, #111 0%, #222 100%); color: #29b6f6; border: 1px solid #29b6f6; transition: 0.3s; }
    .stButton>button:hover { background: #29b6f6; color: #000; box-shadow: 0 0 15px #29b6f6; }
    .value-badge { background-color: #00e676; color: black; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.8em; }
</style>
""", unsafe_allow_html=True)

# Session State
if 'dfs_pool' not in st.session_state: st.session_state['dfs_pool'] = pd.DataFrame()
if 'prop_pool' not in st.session_state: st.session_state['prop_pool'] = pd.DataFrame()
if 'api_log' not in st.session_state: st.session_state['api_log'] = []

# ==========================================
# 📡 2. MULTI-VERSE API ROUTER
# ==========================================
class MultiVerseGateway:
    def __init__(self, api_key):
        self.key = api_key
        self.headers = {"X-RapidAPI-Key": self.key, "Content-Type": "application/json"}

    def fetch_data(self, sport):
        data = []
        try:
            if sport == 'NBA':
                url = "https://api-nba-v1.p.rapidapi.com/games"
                self.headers["X-RapidAPI-Host"] = "api-nba-v1.p.rapidapi.com"
                res = requests.get(url, headers=self.headers, params={"date": datetime.now().strftime("%Y-%m-%d")})
                if res.status_code == 200:
                    for g in res.json().get('response', []):
                        data.append({'game_info': f"{g['teams']['visitors']['code']} @ {g['teams']['home']['code']}"})
            elif sport == 'NFL':
                url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getDailyGameList"
                self.headers["X-RapidAPI-Host"] = "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
                res = requests.get(url, headers=self.headers, params={"gameDate": datetime.now().strftime("%Y%m%d")})
                if res.status_code == 200:
                    for g in res.json().get('body', []):
                        data.append({'game_info': g.get('gameID')})
        except: pass
        return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def cached_api_fetch(sport, key):
    gw = MultiVerseGateway(key)
    return gw.fetch_data(sport)

# ==========================================
# 💾 3. DATABASE
# ==========================================
def init_db():
    conn = sqlite3.connect('titan_multiverse.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS bankroll (id INTEGER PRIMARY KEY, date TEXT, amount REAL, notes TEXT)''')
    c.execute('SELECT count(*) FROM bankroll')
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO bankroll (date, amount, notes) VALUES (?, ?, ?)", (datetime.now().strftime("%Y-%m-%d"), 1000.0, 'Genesis'))
        conn.commit()
    return conn

def get_bankroll(conn):
    try:
        df = pd.read_sql("SELECT * FROM bankroll ORDER BY id DESC LIMIT 1", conn)
        return df.iloc[0]['amount'] if not df.empty else 1000.0
    except: return 1000.0

def update_bankroll(conn, amount, note):
    c = conn.cursor()
    c.execute("INSERT INTO bankroll (date, amount, notes) VALUES (?, ?, ?)", (datetime.now().strftime("%Y-%m-%d"), float(amount), note))
    conn.commit()

# ==========================================
# 🧠 4. TITAN BRAIN
# ==========================================
class TitanBrain:
    def __init__(self, bankroll):
        self.bankroll = bankroll

    def apply_strategic_boosts(self, row, sport):
        proj = row.get('projection', 0.0)
        notes = []
        
        try:
            # 1. HOT HAND
            if row.get('l3_fpts', 0) > 0 and row.get('avg_fpts', 0) > 0:
                diff_pct = (row['l3_fpts'] - row['avg_fpts']) / row['avg_fpts']
                if diff_pct > 0.20:
                    proj *= 1.05
                    notes.append("🔥 Hot Form")
                elif diff_pct < -0.20:
                    proj *= 0.95
                    notes.append("❄️ Cold Streak")

            # 2. WEATHER
            if sport in ['NFL', 'MLB'] and 'wind' in row and 'precip' in row:
                if row['wind'] > 15 or row['precip'] > 50:
                    pos = str(row.get('position', ''))
                    if 'QB' in pos or 'WR' in pos or 'TE' in pos:
                        proj *= 0.85 
                        notes.append("🌪️ Weather Fade")
                    elif 'RB' in pos or 'DST' in pos:
                        proj *= 1.05 
                        notes.append("🛡️ Weather Boost")

            # 3. MATCHUP
            if 'opp_rank' in row and row['opp_rank'] > 0:
                if row['opp_rank'] >= 25: 
                    proj *= 1.08
                    notes.append("🟢 Elite Matchup")
                elif row['opp_rank'] <= 5: 
                    proj *= 0.92
                    notes.append("🔴 Tough Matchup")

        except Exception: pass
        return proj, " | ".join(notes)

    def calculate_slip_ev(self, slip_df, book, stake):
        probs = slip_df['Win Prob %'].values / 100.0
        legs = len(probs)
        payout = 0
        
        if book == 'PrizePicks':
            if legs == 2: payout = 3.0
            elif legs == 3: payout = 5.0
            elif legs == 4: payout = 10.0
            elif legs == 5: payout = 10.0 
            elif legs == 6: payout = 25.0
        elif book == 'Underdog':
            if legs == 2: payout = 3.0
            elif legs == 3: payout = 6.0
            elif legs == 4: payout = 10.0
            elif legs == 5: payout = 20.0
        
        win_chance = np.prod(probs)
        expected_return = (win_chance * payout * stake) - stake
        roi = (expected_return / stake) * 100
        
        return win_chance * 100, payout, expected_return, roi

    def analyze_lineup(self, lineup_df, sport, slate_size):
        msg = []
        stacks = lineup_df['team'].value_counts()
        heavy_stack = stacks[stacks >= 2].index.tolist()
        
        if heavy_stack: msg.append(f"🔗 **Stack:** {', '.join(heavy_stack)}")
        else: msg.append(f"🧩 **Scatter Build**")
            
        avg_proj = lineup_df['projection'].mean()
        msg.append(f"📊 **Proj:** {avg_proj:.1f}")
        
        salary_used = lineup_df['salary'].sum()
        msg.append(f"💰 **Cap:** ${salary_used:,}")
        
        if slate_size <= 4 and len(heavy_stack) < 1: 
            msg.append("⚠️ Small Slate needs correlation!")
        
        return " | ".join(msg)

# ==========================================
# 📂 5. DATA REFINERY
# ==========================================
class DataRefinery:
    @staticmethod
    def clean_curr(val):
        try: return float(re.sub(r'[^\d.-]', '', str(val).strip()))
        except: return 0.0

    @staticmethod
    def normalize_pos(pos):
        p = str(pos).upper().strip()
        if 'QUARTER' in p: return 'QB'
        if 'RUNNING' in p: return 'RB'
        if 'RECEIVER' in p: return 'WR'
        if 'TIGHT' in p: return 'TE'
        if 'DEF' in p or 'DST' in p: return 'DST'
        if 'GUARD' in p: return 'G'
        if 'FORWARD' in p: return 'F'
        if 'CENTER' in p: return 'C'
        # CBB Fix
        if 'G/UTIL' in p: return 'G'
        if 'F/UTIL' in p: return 'F'
        # NHL Fix
        if 'LEFT WING' in p: return 'LW'
        if 'RIGHT WING' in p: return 'RW'
        if 'GOALIE' in p: return 'G'
        return p

    @staticmethod
    def ingest(df, sport_tag, source_tag="Generic"):
        df.columns = df.columns.astype(str).str.upper().str.strip()
        std = pd.DataFrame()
        
        maps = {
            'name': ['PLAYER', 'NAME', 'WHO', 'ATHLETE'],
            'projection': ['ROTOWIRE PROJECTION', 'PROJECTION', 'FPTS', 'PROJ', 'AVG FPTS', 'PTS'], 
            'salary': ['SAL', 'SALARY', 'CAP', 'COST', 'PRICE'],
            'position': ['POS', 'POSITION', 'ROSTER POSITION', 'SLOT'],
            'team': ['TEAM', 'TM', 'SQUAD'],
            'opp': ['OPP', 'OPPONENT', 'VS'], 
            'prop_line': ['LINE', 'PROP', 'PLAYER PROP', 'STRIKE', 'TOTAL'], 
            'market': ['MARKET NAME', 'MARKET', 'STAT'],
            'game_info': ['GAME INFO', 'GAME', 'MATCHUP', 'OPPONENT'],
            'date': ['DATE', 'GAME DATE'],
            'time': ['TIME', 'GAME TIME', 'TIME (ET)'],
            'status': ['STATUS', 'INJURY', 'AVAILABILITY'], 
            'tm_score': ['TM SCORE', 'IMPLIED TOTAL', 'TEAM TOTAL'], 
            'game_total': ['O/U', 'GAME TOTAL', 'OVER/UNDER'],
            'rst_own': ['RST%', 'OWN%', 'PROJECTED OWNERSHIP'], 
            'l3_fpts': ['L3 FPTS', 'LAST 3'],
            'avg_fpts': ['AVG FPTS', 'SEASON AVG'],
            'opp_rank': ['OPP RANK', 'OPP PR', 'DVP'],
            'wind': ['WIND', 'WIND SPEED'],
            'precip': ['PRECIP', 'PRECIP %', 'RAIN'],
            'is_home': ['HOME?', 'HOME', 'VENUE'],
            'factor_pickem': ["DFS PICK'EM SITES FACTOR", "PICK'EM FACTOR"],
            'factor_sportsbook': ["SPORTSBOOKS FACTOR", "BOOKS FACTOR"],
            'factor_hit_rate': ["HIT RATE FACTOR", "HIT RATE"],
            'factor_proj': ["ROTOWIRE PROJECTION FACTOR", "PROJ FACTOR"]
        }
        
        site_maps = {
            'prizepicks_line': ['PRIZEPICKS LINE', 'PRIZEPICKS'],
            'underdog_line': ['UNDERDOG LINE', 'UNDERDOG'],
            'sleeper_line': ['SLEEPER LINE', 'SLEEPER'],
            'pick6_line': ['DRAFTKINGS PICK6 LINE', 'PICK6 LINE', 'PICK6']
        }
        
        for target, sources in maps.items():
            for s in sources:
                if s in df.columns:
                    if target in ['projection', 'salary', 'prop_line', 'tm_score', 'game_total', 'rst_own', 'l3_fpts', 'avg_fpts', 'opp_rank', 'wind', 'precip', 'factor_pickem', 'factor_sportsbook', 'factor_hit_rate', 'factor_proj']:
                        std[target] = df[s].apply(DataRefinery.clean_curr)
                    elif target == 'name':
                        std[target] = df[s].astype(str).apply(lambda x: x.split('(')[0].strip())
                    elif target == 'position':
                        std[target] = df[s].apply(DataRefinery.normalize_pos)
                    elif target == 'status':
                        std[target] = df[s].astype(str).str.strip()
                    else:
                        std[target] = df[s].astype(str).str.strip()
                    break
        
        for target, sources in site_maps.items():
            for s in sources:
                if s in df.columns:
                    std[target] = df[s].apply(DataRefinery.clean_curr)
                    break
            if target not in std.columns: std[target] = 0.0

        if 'name' not in std.columns: return pd.DataFrame()
        
        # 🟢 FIX: AUTO-GENERATE GAME INFO
        if 'game_info' not in std.columns:
            if 'team' in std.columns and 'opp' in std.columns:
                std['game_info'] = std['team'] + ' vs ' + std['opp']
                if 'time' in std.columns:
                     std['game_info'] += ' (' + std['time'].astype(str) + ')'
            else:
                std['game_info'] = 'All Games'
        
        if 'status' not in std.columns: std['status'] = 'Active'
        else: std['status'].fillna('Active', inplace=True)
        
        defaults = {'projection':0.0, 'salary':0.0, 'prop_line':0.0, 'position':'FLEX', 'team':'N/A', 'opp':'N/A',
                   'market':'Standard', 'date':datetime.now().strftime("%Y-%m-%d"), 'time':'TBD', 'rst_own':10.0,
                   'l3_fpts':0.0, 'avg_fpts':0.0, 'opp_rank':16.0, 'wind':0.0, 'precip':0.0, 'is_home':'No'}
        for k,v in defaults.items():
            if k not in std.columns: std[k] = v
        
        if 'SPORT' in df.columns: std['sport'] = df['SPORT'].str.strip().str.upper()
        else: std['sport'] = sport_tag.upper()
        
        if 'salary' in std.columns and 'projection' in std.columns:
            std['value_score'] = np.where(std['salary'] > 0, (std['projection'] / std['salary']) * 1000, 0)
        
        factors = ['factor_pickem', 'factor_sportsbook', 'factor_hit_rate', 'factor_proj']
        valid_factors = [f for f in factors if f in std.columns]
        if valid_factors:
            for f in valid_factors:
                std[f] = pd.to_numeric(std[f], errors='coerce')
            std['spike_score'] = std[valid_factors].mean(axis=1).fillna(0)
        else:
            std['spike_score'] = 0.0 
        
        if source_tag == 'PrizePicks' and std['prizepicks_line'].sum() == 0:
            std['prizepicks_line'] = std['prop_line']
        elif source_tag == 'Underdog' and std['underdog_line'].sum() == 0:
            std['underdog_line'] = std['prop_line']
        elif source_tag == 'Sleeper' and std['sleeper_line'].sum() == 0:
            std['sleeper_line'] = std['prop_line']
        elif source_tag == 'DraftKings Pick6' and std['pick6_line'].sum() == 0:
            std['pick6_line'] = std['prop_line']
            
        return std

    @staticmethod
    def merge(base, new_df):
        if base.empty: return new_df
        if new_df.empty: return base
        combined = pd.concat([base, new_df])
        
        numeric_cols = [
            'projection', 'salary', 'prop_line', 'value_score', 'tm_score', 'game_total', 'rst_own',
            'l3_fpts', 'avg_fpts', 'opp_rank', 'wind', 'precip',
            'prizepicks_line', 'underdog_line', 'sleeper_line', 'pick6_line',
            'factor_pickem', 'factor_sportsbook', 'factor_hit_rate', 'factor_proj', 'spike_score'
        ]
        meta_cols = ['position', 'team', 'game_info', 'date', 'time', 'status', 'is_home', 'opp']
        
        agg_dict = {col: 'max' for col in numeric_cols if col in combined.columns}
        for col in meta_cols:
            if col in combined.columns: agg_dict[col] = 'last'
            
        try:
            fused = combined.groupby(['name', 'sport', 'market'], as_index=False).agg(agg_dict)
            return fused
        except:
            return combined.drop_duplicates(subset=['name', 'sport', 'market'], keep='last')

# ==========================================
# 📡 6. AI SCOUT
# ==========================================
def run_web_scout(sport):
    intel = {}
    if HAS_AI:
        try:
            with DDGS() as ddgs:
                queries = [f"{sport} dfs winning strategy matchups today", f"{sport} player prop sleepers"]
                for q in queries:
                    for r in list(ddgs.text(q, max_results=3)):
                        intel[(r['title'] + " " + r['body']).lower()] = 1
        except: pass
    return intel

# ==========================================
# 🏭 7. OPTIMIZER (ALL SPORTS + CBB)
# ==========================================
def get_roster_rules(sport, site, mode):
    rules = {'size': 0, 'cap': 50000, 'constraints': []}
    
    if site == 'DK': rules['cap'] = 50000
    elif site == 'FD': rules['cap'] = 60000
    elif site == 'Yahoo': rules['cap'] = 200
    else: rules['cap'] = 999999 
    
    if mode == 'Showdown':
        if site == 'DK':
            rules['size'] = 6
            rules['constraints'].append(('CPT', 1, 1))
        elif site == 'FD':
            rules['size'] = 5
            rules['constraints'].append(('MVP', 1, 1))
        return rules

    if sport == 'NFL':
        if site == 'DK':
            rules['size'] = 9
            rules['constraints'] = [('QB', 1, 1), ('DST', 1, 1), ('RB', 2, 3), ('WR', 3, 4), ('TE', 1, 2)]
        elif site == 'FD':
            rules['size'] = 9
            rules['constraints'] = [('QB', 1, 1), ('DST', 1, 1), ('RB', 2, 3), ('WR', 3, 4), ('TE', 1, 2)]
    elif sport == 'NBA':
        if site == 'DK':
            rules['size'] = 8
            rules['constraints'] = [('PG', 1, 3), ('SG', 1, 3), ('SF', 1, 3), ('PF', 1, 3), ('C', 1, 2)]
        elif site == 'FD':
            rules['size'] = 9
            rules['constraints'] = [('PG', 2, 2), ('SG', 2, 2), ('SF', 2, 2), ('PF', 2, 2), ('C', 1, 1)]
    elif sport == 'MLB':
        if site == 'DK':
            rules['size'] = 10
            rules['constraints'] = [('P', 2, 2), ('C', 0, 1), ('1B', 0, 1), ('2B', 0, 1), ('3B', 0, 1), ('SS', 0, 1), ('OF', 3, 3)]
    elif sport == 'NHL':
        if site == 'DK':
            rules['size'] = 9
            rules['constraints'] = [('C', 2, 3), ('W', 3, 4), ('D', 2, 3), ('G', 1, 1)]
    elif sport == 'PGA':
        rules['size'] = 6
    elif sport == 'CBB':
        if site == 'DK':
            rules['size'] = 8
            rules['constraints'] = [('G', 3, 5), ('F', 3, 5)]

    if rules['size'] == 0: rules['size'] = 6
    return rules

def optimize_lineup(df, config):
    target_sport = config['sport'].strip().upper()
    pool = df[df['sport'].str.strip().str.upper() == target_sport].copy()
    
    if config['site'] not in ['PrizePicks', 'Underdog', 'Sleeper', 'DraftKings Pick6'] and not config['ignore_salary']:
        pool = pool[pool['salary'] > 0]
    
    if 'status' in pool.columns:
        pool = pool[~pool['status'].str.contains('Out|IR|NA|Doubtful', case=False, na=False)]
    
    if pool.empty:
        st.error(f"❌ No DFS players found for {target_sport}. Please use the **Left Uploader** in Tab 1.")
        return None

    brain = TitanBrain(0)
    try:
        applied = pool.apply(lambda row: brain.apply_strategic_boosts(row, target_sport), axis=1, result_type='expand')
        pool['projection'] = applied[0]
        pool['notes'] = applied[1]
    except: pass

    unique_games = pool['game_info'].nunique()
    is_small_slate = unique_games <= 4
    st.info(f"📊 **Slate Context:** {unique_games} Games. Strategy: {'Aggressive/Correlation' if is_small_slate else 'Standard'}")

    if config['slate_games']:
        pool = pool[pool['game_info'].isin(config['slate_games'])].reset_index(drop=True)
    
    if config['positions']:
        pool = pool[pool['position'].isin(config['positions'])].reset_index(drop=True)

    pool = pool[pool['projection'] > 0].reset_index(drop=True)
    if config['bans']: pool = pool[~pool['name'].isin(config['bans'])].reset_index(drop=True)
    
    if config['mode'] == 'Showdown':
        flex = pool.copy(); flex['pos_id'] = 'FLEX'
        cpt = pool.copy(); cpt['pos_id'] = 'CPT'
        cpt['name'] += " (CPT)"
        cpt['projection'] *= 1.5
        if config['site'] == 'DK': cpt['salary'] *= 1.5
        pool = pd.concat([cpt, flex]).reset_index(drop=True)
    else:
        pool['pos_id'] = pool['position'].replace({'D': 'DST', 'DEF': 'DST'})

    rules = get_roster_rules(config['sport'], config['site'], config['mode'])
    lineups = []
    player_exposure = {i: 0 for i in pool.index}
    
    def solve_lp(relax_stacking=False):
        prob = pulp.LpProblem("Titan", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("p", pool.index, cat='Binary')
        
        sim_noise = 0.5
        if 'factor_hit_rate' in pool.columns:
            sim_noise = 1.0 - (pool['factor_hit_rate'].fillna(50) / 100.0)
        randomness = np.random.normal(0, sim_noise, len(pool))
        
        prob += pulp.lpSum([(pool.loc[p, 'projection'] + randomness[p]) * x[p] for p in pool.index])
        prob += pulp.lpSum([pool.loc[p, 'salary'] * x[p] for p in pool.index]) <= rules['cap']
        prob += pulp.lpSum([x[p] for p in pool.index]) == rules['size']
        
        for role, min_req, max_req in rules['constraints']:
            idx = pool[pool['pos_id'].str.contains(role, regex=True, na=False)].index
            if not idx.empty:
                prob += pulp.lpSum([x[p] for p in idx]) >= min_req
                prob += pulp.lpSum([x[p] for p in idx]) <= max_req
        
        for lock in config['locks']:
            l_idx = pool[pool['name'].str.contains(lock, regex=False)].index
            if not l_idx.empty: prob += pulp.lpSum([x[p] for p in l_idx]) >= 1

        if not relax_stacking and config['smart_stack'] and config['sport'] == 'NFL':
            qbs = pool[pool['pos_id'] == 'QB'].index
            for qb in qbs:
                team = pool.loc[qb, 'team']
                teammates = pool[(pool['team'] == team) & (pool['pos_id'].isin(['WR', 'TE']))].index
                if not teammates.empty:
                    prob += pulp.lpSum([x[t] for t in teammates]) >= x[qb]

        if config['max_exposure'] < 100:
            max_lineups = max(1, int(config['count'] * (config['max_exposure'] / 100.0)))
            for p_idx in pool.index:
                if player_exposure[p_idx] >= max_lineups: prob += x[p_idx] == 0

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        return prob, x

    for i in range(config['count']):
        prob, x = solve_lp(relax_stacking=False)
        if prob.status != 1: prob, x = solve_lp(relax_stacking=True)
        if prob.status == 1:
            sel = [p for p in pool.index if x[p].varValue == 1]
            lu = pool.loc[sel].copy()
            lu['Lineup_ID'] = i+1
            lineups.append(lu)
            for p in sel: player_exposure[p] += 1
            
    return pd.concat(lineups) if lineups else None

# ==========================================
# 🧩 8. PROP OPTIMIZER (ROBUST FIX)
# ==========================================
def optimize_slips(df, config):
    target_sport = config['sport'].strip().upper()
    pool = df[df['sport'].str.strip().str.upper() == target_sport].copy()
    
    if 'projection' not in pool.columns: pool['projection'] = 0.0
    
    line_col = 'prop_line'
    if config['book'] == 'PrizePicks': line_col = 'prizepicks_line'
    elif config['book'] == 'Underdog': line_col = 'underdog_line'
    elif config['book'] == 'Sleeper': line_col = 'sleeper_line'
    elif config['book'] == 'DraftKings Pick6': line_col = 'pick6_line'
    
    brain = TitanBrain(0)
    
    try:
        applied = pool.apply(lambda row: brain.apply_strategic_boosts(row, target_sport), axis=1, result_type='expand')
        pool['smart_projection'] = applied[0]
        pool['notes'] = applied[1]
    except:
        pool['smart_projection'] = pool['projection']
        pool['notes'] = ""
    
    pool['smart_projection'] = pd.to_numeric(pool['smart_projection'], errors='coerce').fillna(0)
    pool[line_col] = pd.to_numeric(pool[line_col], errors='coerce').fillna(0)
    pool['prop_line'] = pd.to_numeric(pool['prop_line'], errors='coerce').fillna(0)
    
    def calc_score(row):
        line = row[line_col] if row[line_col] > 0 else row['prop_line']
        proj = row['smart_projection']
        
        if line <= 0 or proj <= 0: return 0
        edge = abs(proj - line) / line
        score = edge * 100
        if row.get('spike_score', 0) > 70: score *= 1.2
        if row.get('factor_hit_rate', 0) > 60: score *= 1.1
        if 'Weather' in str(row.get('notes', '')): score *= 1.1 
        return score

    pool['opt_score'] = pool.apply(calc_score, axis=1)
    pool = pool[pool['opt_score'] > 5].sort_values('opt_score', ascending=False)
    
    if pool.empty: return None
    
    slips = []
    for i in range(config['count']):
        current_slip = []
        slip_players = set()
        slip_teams = set()
        
        for idx, row in pool.iterrows():
            if len(current_slip) >= config['legs']: break
            if row['name'] in slip_players: continue 
            
            # 🧩 CORRELATION BOOST
            if config['correlation'] and row['team'] in slip_teams:
                 row['opt_score'] *= 1.1
            
            line = row[line_col] if row[line_col] > 0 else row['prop_line']
            proj = row['smart_projection']
            pick_type = "OVER" if proj > line else "UNDER"
            
            prop_data = {
                'Player': row['name'], 'Market': row['market'], 'Line': line,
                'Smart Proj': proj, 'Pick': pick_type, 'Score': row['opt_score'], 'Notes': row['notes']
            }
            current_slip.append(prop_data)
            slip_players.add(row['name'])
            slip_teams.add(row['team'])
            
        if len(current_slip) == config['legs']:
            slips.append(pd.DataFrame(current_slip))
            if not pool.empty: pool = pool.iloc[1:] 
        else:
            break
            
    return slips

def get_csv_download(df):
    return df.to_csv(index=False).encode('utf-8')

# ==========================================
# 🖥️ 9. DASHBOARD (SPLIT UI)
# ==========================================
conn = init_db()
st.sidebar.title("TITAN OMNI")
st.sidebar.caption("Hydra Edition 43.0 (Complete)")

try: API_KEY = st.secrets["rapid_api_key"]
except: API_KEY = st.sidebar.text_input("Enter RapidAPI Key", type="password")

current_bank = get_bankroll(conn)
st.sidebar.metric("🏦 Bankroll", f"${current_bank:,.2f}")

sport = st.sidebar.selectbox("Sport", ["NBA", "NFL", "MLB", "CFB", "NHL", "SOCCER", "PGA", "TENNIS", "CS2", "CBB"])
site = st.sidebar.selectbox("Site/Book", ["DK", "FD", "Yahoo", "PrizePicks", "Underdog", "Sleeper", "DraftKings Pick6"])

tabs = st.tabs(["1. 📡 Fusion", "2. 🏰 Optimizer", "3. 🔮 Simulation", "4. 🚀 Props", "5. 🧮 Parlay", "6. 🧩 Prop Opt"])

# --- TAB 1: DATA ---
with tabs[0]:
    st.markdown("### 📡 Data Fusion")
    if st.button("🗑️ Clear All Data"):
        st.session_state['dfs_pool'] = pd.DataFrame()
        st.session_state['prop_pool'] = pd.DataFrame()
        st.success("Pools Cleared.")

    col_api, col_file = st.columns(2)
    with col_api:
        if st.button("☁️ AUTO-SYNC APIS"):
             if not API_KEY: st.error("No API Key Provided")
             else:
                 with st.spinner("Syncing..."):
                    cached_api_fetch(sport, API_KEY)
                    st.success("Sync Complete")
    
    st.divider()
    c1, c2 = st.columns(2)
    
    with c1:
        st.success("🏰 **DFS UPLOAD (Left Lane)**")
        st.caption("Upload DraftKings / FanDuel CSVs with Salaries here.")
        dfs_files = st.file_uploader("Upload DFS CSVs", accept_multiple_files=True, key="dfs_up")
        if st.button("🧬 Load DFS Data"):
            if dfs_files:
                ref = DataRefinery()
                new_data = pd.DataFrame()
                for f in dfs_files:
                    try:
                        raw = pd.read_csv(f)
                        st.caption(f"Ingesting {f.name}...")
                        new_data = ref.merge(new_data, ref.ingest(raw, sport, "Generic"))
                    except: st.error(f"Error {f.name}")
                st.session_state['dfs_pool'] = new_data
                st.success(f"Loaded {len(new_data)} DFS Players.")

    with c2:
        st.info("🚀 **PROP UPLOAD (Right Lane)**")
        st.caption("Upload PrizePicks / Underdog / Sleeper CSVs here.")
        prop_source = st.selectbox("Prop Source:", ["PrizePicks", "Underdog", "Sleeper", "DraftKings Pick6"], key="prop_src")
        prop_files = st.file_uploader(f"Upload {prop_source} CSVs", accept_multiple_files=True, key="prop_up")
        if st.button("🧬 Load Prop Data"):
            if prop_files:
                ref = DataRefinery()
                new_data = pd.DataFrame()
                for f in prop_files:
                    try:
                        raw = pd.read_csv(f)
                        st.caption(f"Ingesting {f.name}...")
                        new_data = ref.merge(new_data, ref.ingest(raw, sport, prop_source))
                    except: st.error(f"Error {f.name}")
                st.session_state['prop_pool'] = new_data
                st.success(f"Loaded {len(new_data)} Prop Players.")
    
    if st.button("🛰️ Run AI Scout"):
        st.session_state['ai_intel'] = run_web_scout(sport)
        st.success("AI Intel Gathered")

# --- TAB 2: OPTIMIZER ---
with tabs[1]:
    st.markdown("### 🏰 Lineup Generator (DFS Only)")
    pool = st.session_state['dfs_pool']
    active = pool[pool['sport'].str.strip().str.upper() == sport.strip().upper()] if not pool.empty else pd.DataFrame()
    
    if active.empty:
        st.warning(f"No DFS data found for {sport}. Please use the **Left Uploader** in Tab 1.")
    else:
        if 'value_score' in active.columns:
            top_value = active.sort_values('value_score', ascending=False).head(6)
            st.markdown("##### 💎 Core Plays (Best Value/Salary)")
            cols = st.columns(6)
            for i, (idx, row) in enumerate(top_value.iterrows()):
                cols[i].metric(row['name'], f"${row['salary']}", f"{row['value_score']:.1f}x")

        games = sorted(active['game_info'].astype(str).unique())
        slate = st.multiselect("🗓️ Filter Slate", games, default=games)
        
        c1, c2, c3 = st.columns(3)
        mode = c1.radio("Mode", ["Classic", "Showdown"])
        count = c2.slider("Lineups", 1, 50, 10)
        
        with st.expander("🧠 Grandmaster Strategy", expanded=True):
            smart_stack = st.checkbox("Milly Maker Logic (Correlation + Bring Back)", value=True)
            game_script = st.checkbox("Game Script Boost (High Totals)", value=True)
            chalk_fade = st.slider("Chalk Fade %", 0, 50, 0)
            max_exposure = st.slider("Max Player Exposure %", 10, 100, 60)
            ignore_salary = st.checkbox("⚠️ Ignore Salary Cap (For Testing)", value=False)
            locks = st.multiselect("Lock Players", sorted(active['name'].unique()))
            pos_filter = st.multiselect("Filter Positions", sorted(active['position'].unique()))
        
        if st.button("⚡ Generate & Analyze"):
            cfg = {
                'sport':sport, 'site':site, 'mode':mode, 'count':count, 'locks':locks, 'bans':[], 
                'slate_games':slate, 'sim':False, 'positions':pos_filter,
                'smart_stack': smart_stack, 'game_script': game_script, 'max_exposure': max_exposure,
                'chalk_fade': chalk_fade, 'ignore_salary': ignore_salary
            }
            res = optimize_lineup(st.session_state['dfs_pool'], cfg)
            
            if res is not None:
                st.dataframe(res)
                top_lu = res[res['Lineup_ID']==1]
                brain = TitanBrain(current_bank)
                slate_size = len(slate) if slate else len(games)
                feedback = brain.analyze_lineup(top_lu, sport, slate_size)
                st.info(f"💡 **Lineup 1 Analysis:** {feedback}")
                
                csv_data = get_csv_download(res)
                st.download_button("📥 Export CSV", data=csv_data, file_name="titan_lineups.csv", mime="text/csv")
            else:
                st.error("Optimization Failed. Try loosening exposure limits.")

# --- TAB 3: SIMULATION ---
with tabs[2]:
    st.markdown("### 🔮 Monte Carlo Simulation")
    if st.button("🎲 Run Simulation (200 Lineups)"):
        with st.spinner("Simulating..."):
            sim_count = st.slider("Sim Count", 50, 1000, 200)
            cfg = {'sport':sport, 'site':site, 'mode':"Classic", 'count':sim_count, 'locks':[], 'bans':[], 'stack':True, 'sim':True, 'slate_games':[], 'positions':[], 'smart_stack':False, 'game_script':False, 'max_exposure':100, 'chalk_fade':0, 'ignore_salary':False}
            res = optimize_lineup(st.session_state['dfs_pool'], cfg)
            if res is not None:
                exp = res['name'].value_counts(normalize=True).mul(100).reset_index()
                st.plotly_chart(px.bar(exp.head(15), x='proportion', y='name', orientation='h', title="Simulated Exposure"))

# --- TAB 4: PROPS ---
with tabs[3]:
    st.markdown("### 🚀 Prop Analyzer (Props Only)")
    pool = st.session_state['prop_pool']
    active = pool[pool['sport'].str.strip().str.upper() == sport.strip().upper()].copy() if not pool.empty else pd.DataFrame()
    
    if active.empty: st.warning(f"No prop data found for {sport}. Please use the **Right Uploader** in Tab 1.")
    else:
        brain = TitanBrain(current_bank)
        target_line_col = 'prop_line'
        if site == 'Underdog': target_line_col = 'underdog_line'
        elif site == 'Sleeper': target_line_col = 'sleeper_line'
        elif site == 'PrizePicks': target_line_col = 'prizepicks_line'
        elif site == 'DraftKings Pick6': target_line_col = 'pick6_line'
        
        st.info(f"🔎 Analyzing for **{site}**. Using column: `{target_line_col}`.")
        
        active['final_line'] = active.apply(lambda x: x[target_line_col] if x.get(target_line_col, 0) > 0 else x['prop_line'], axis=1)
        
        try:
            applied = active.apply(lambda row: brain.apply_strategic_boosts(row, sport), axis=1, result_type='expand')
            active['smart_projection'] = applied[0]
            active['notes'] = applied[1]
        except:
            active['smart_projection'] = active['projection']
            active['notes'] = ""
        
        active['pick'] = np.where(active['smart_projection'] > active['final_line'], "OVER", "UNDER")
        
        res = active.apply(lambda x: brain.calculate_prop_edge(x, 'final_line'), axis=1, result_type='expand')
        active['units'] = res[0]
        active['rating'] = res[1]
        active['win_prob'] = res[2]
        active['titan_logic'] = res[3]
        
        valid = active[(active['projection'] > 0) & (active['final_line'] > 0)].copy()
        
        show_spikes = st.checkbox("🔥 Show Only 'Spiked' Players (High Factor Score)")
        if show_spikes:
            valid = valid[valid['spike_score'] > 75]
        
        cols = ['date', 'time', 'name', 'market', 'team', 'final_line', 'smart_projection', 'pick', 'win_prob', 'rating', 'titan_logic', 'spike_score']
        final_cols = [c for c in cols if c in valid.columns]
        
        st.dataframe(
            valid[final_cols]
            .sort_values('win_prob', ascending=False)
            .style.format({'final_line':'{:.1f}', 'smart_projection':'{:.1f}', 'win_prob':'{:.1f}%', 'spike_score':'{:.0f}'})
        )

# --- TAB 5: PARLAY ---
with tabs[4]:
    st.markdown("### 🧮 Parlay & Correlation Architect")
    pool = st.session_state['prop_pool']
    if not pool.empty:
        active = pool[pool['sport'].str.strip().str.upper() == sport.strip().upper()].copy()
        target_line_col = 'prop_line'
        if site == 'Underdog': target_line_col = 'underdog_line'
        elif site == 'Sleeper': target_line_col = 'sleeper_line'
        
        active['final_line'] = active.apply(lambda x: x.get(target_line_col, 0) if x.get(target_line_col, 0) > 0 else x.get('prop_line', 0), axis=1)
        active = active[active['final_line'] > 0]
        active['pick'] = np.where(active['projection'] > active['final_line'], "OVER", "UNDER")
        active['uid'] = active['name'] + " (" + active['market'] + ": " + active['pick'] + ")"
        
        st.subheader("🔗 Correlation Station")
        teams = active['team'].unique()
        selected_team = st.selectbox("Find Correlated Pairs for Team:", teams)
        
        team_props = active[active['team'] == selected_team]
        if not team_props.empty:
            st.dataframe(team_props[['name', 'market', 'pick', 'win_prob', 'titan_logic']])
        
        st.divider()
        selection = st.multiselect("Build Custom Slip", active['uid'].unique())
        if selection:
            total_prob = 1.0
            for item in selection:
                row = active[active['uid'] == item].iloc[0]
                edge = abs(row['projection'] - row['final_line']) / row['final_line']
                prob = min(0.75, 0.52 + (edge * 0.5))
                total_prob *= prob
            st.metric("Total Probability", f"{total_prob*100:.1f}%")

# --- TAB 6: PROP OPTIMIZER ---
with tabs[5]:
    st.markdown("### 🧩 Prop Slip Optimizer")
    pool = st.session_state['prop_pool']
    
    valid_props_count = len(pool) if not pool.empty else 0
    st.caption(f"Status: {valid_props_count} Valid Props Available")
    
    if pool.empty:
        st.warning("No props loaded.")
    else:
        c1, c2, c3 = st.columns(3)
        book_select = c1.selectbox("Sportsbook", ["PrizePicks", "Underdog", "Sleeper", "DraftKings Pick6"])
        legs = c2.slider("Legs (Picks per Slip)", 2, 8, 5)
        num_slips = c3.slider("Number of Slips", 1, 10, 3)
        corr_boost = st.checkbox("Boost Correlation (Same Team)", value=True)
        
        wager = st.number_input("Wager Amount ($)", value=20)
        
        if st.button("⚡ Optimize Slips"):
            cfg = {'sport':sport, 'book':book_select, 'legs':legs, 'count':num_slips, 'correlation':corr_boost}
            slips = optimize_slips(pool, cfg)
            
            if slips:
                for i, slip in enumerate(slips):
                    brain = TitanBrain(0)
                    win_prob_slip, payout, exp_ret, roi = brain.calculate_slip_ev(slip, book_select, wager)
                    
                    st.markdown(f"#### 🎫 Slip #{i+1} | Payout: {payout}x | ROI: {roi:.1f}%")
                    st.caption(f"Bet: ${wager} ➡️ Potential Win: ${wager*payout:.2f} | **Expected Value: ${exp_ret:.2f}**")
                    
                    if exp_ret > 0:
                        st.success("✅ **POSITIVE EV - Recommended Bet**")
                    else:
                        st.error("❌ **NEGATIVE EV - Do Not Bet**")
                        
                    st.table(slip)
                    st.divider()
            else:
                st.error("Could not generate valid slips. Check data pool.")
