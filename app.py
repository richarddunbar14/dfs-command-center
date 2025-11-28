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

# SAFE IMPORT
try:
    from duckduckgo_search import DDGS
    HAS_AI = True
except ImportError:
    HAS_AI = False

# --- CONFIG ---
st.set_page_config(layout="wide", page_title="TITAN OMNI V39", page_icon="🌌")
st.markdown("""<style>.stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
.titan-card { background: #111; border: 1px solid #333; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #29b6f6; }
div[data-testid="stDataFrame"] { border: 1px solid #333; border-radius: 5px; }
.stButton>button { width: 100%; border-radius: 4px; font-weight: 800; text-transform: uppercase; background: linear-gradient(90deg, #111 0%, #222 100%); color: #29b6f6; border: 1px solid #29b6f6; transition: 0.3s; }
.stButton>button:hover { background: #29b6f6; color: #000; box-shadow: 0 0 15px #29b6f6; }</style>""", unsafe_allow_html=True)

if 'dfs_pool' not in st.session_state: st.session_state['dfs_pool'] = pd.DataFrame()
if 'prop_pool' not in st.session_state: st.session_state['prop_pool'] = pd.DataFrame()

# --- API ---
class MultiVerseGateway:
    def __init__(self, api_key): self.headers = {"X-RapidAPI-Key": api_key, "Content-Type": "application/json"}
    def fetch_data(self, sport):
        data = []
        try:
            if sport == 'NBA':
                res = requests.get("https://api-nba-v1.p.rapidapi.com/games", headers=self.headers, params={"date": datetime.now().strftime("%Y-%m-%d")})
                if res.status_code == 200:
                    for g in res.json().get('response', []): data.append({'game_info': f"{g['teams']['visitors']['code']} @ {g['teams']['home']['code']}", 'sport': 'NBA'})
            elif sport == 'NFL':
                res = requests.get("https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getDailyGameList", headers=self.headers, params={"gameDate": datetime.now().strftime("%Y%m%d")})
                if res.status_code == 200:
                    for g in res.json().get('body', []): data.append({'game_info': g.get('gameID'), 'sport': 'NFL'})
        except: pass
        return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def cached_api_fetch(sport, key): return MultiVerseGateway(key).fetch_data(sport)

# --- DB ---
def init_db():
    conn = sqlite3.connect('titan_multiverse.db'); c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS bankroll (id INTEGER PRIMARY KEY, date TEXT, amount REAL, notes TEXT)''')
    c.execute('SELECT count(*) FROM bankroll')
    if c.fetchone()[0] == 0: c.execute("INSERT INTO bankroll (date, amount, notes) VALUES (?, ?, ?)", (datetime.now().strftime("%Y-%m-%d"), 1000.0, 'Genesis')); conn.commit()
    return conn
def get_bankroll(conn):
    try: return pd.read_sql("SELECT * FROM bankroll ORDER BY id DESC LIMIT 1", conn).iloc[0]['amount']
    except: return 1000.0

# --- BRAIN ---
class TitanBrain:
    def apply_strategic_boosts(self, row, sport):
        proj = row.get('projection', 0.0); notes = []
        try:
            if row.get('l3_fpts', 0) > 0 and row.get('avg_fpts', 0) > 0:
                diff = (row['l3_fpts'] - row['avg_fpts']) / row['avg_fpts']
                if diff > 0.2: proj *= 1.05; notes.append("🔥 Hot")
                elif diff < -0.2: proj *= 0.95; notes.append("❄️ Cold")
            if sport in ['NFL','MLB'] and row.get('wind', 0) > 15:
                if 'QB' in str(row.get('position')) or 'WR' in str(row.get('position')): proj *= 0.85; notes.append("🌪️ Wind Fade")
            if row.get('opp_rank', 16) >= 25: proj *= 1.08; notes.append("🟢 Elite Matchup")
            elif row.get('opp_rank', 16) <= 5: proj *= 0.92; notes.append("🔴 Tough Matchup")
        except: pass
        return proj, " | ".join(notes)

    def calculate_slip_ev(self, slip_df, book, stake):
        probs = slip_df['Win Prob %'].values / 100.0; legs = len(probs); payout = 0
        if book == 'PrizePicks': payout = {2:3, 3:5, 4:10, 5:10, 6:25}.get(legs, 0)
        elif book == 'Underdog': payout = {2:3, 3:6, 4:10, 5:20}.get(legs, 0)
        win = np.prod(probs)
        return win * 100, payout, (win * payout * stake) - stake

# --- DATA ---
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
        if 'DEF' in p: return 'DST'
        if 'GUARD' in p or 'G/UTIL' in p: return 'G'
        if 'FORWARD' in p or 'F/UTIL' in p: return 'F'
        if 'CENTER' in p: return 'C'
        if 'GOALIE' in p: return 'G'
        return p
    @staticmethod
    def ingest(df, sport_tag, source_tag="Generic"):
        df.columns = df.columns.astype(str).str.upper().str.strip(); std = pd.DataFrame()
        maps = {'name': ['PLAYER','NAME'], 'projection': ['FPTS','PROJ','ROTOWIRE PROJECTION'], 'salary': ['SAL','SALARY'], 'position': ['POS','POSITION'], 
                'team': ['TEAM','TM'], 'opp': ['OPP','VS'], 'prop_line': ['LINE','PROP'], 'market': ['MARKET'], 'game_info': ['GAME INFO','GAME'], 
                'status': ['STATUS','INJURY'], 'l3_fpts': ['L3 FPTS','LAST 3'], 'avg_fpts': ['AVG FPTS'], 'opp_rank': ['OPP RANK','DVP'], 
                'wind': ['WIND'], 'factor_hit_rate': ['HIT RATE FACTOR']}
        for target, sources in maps.items():
            for s in sources:
                if s in df.columns:
                    if target in ['projection','salary','prop_line','l3_fpts','avg_fpts','opp_rank','wind','factor_hit_rate']: std[target] = df[s].apply(DataRefinery.clean_curr)
                    elif target == 'position': std[target] = df[s].apply(DataRefinery.normalize_pos)
                    else: std[target] = df[s].astype(str).str.strip()
                    break
        if 'name' not in std.columns: return pd.DataFrame()
        if 'game_info' not in std.columns and 'team' in std.columns: std['game_info'] = std['team'] + ' vs ' + std.get('opp', 'Opp')
        std['sport'] = sport_tag.upper(); std['status'] = std.get('status', pd.Series(['Active']*len(std))).fillna('Active')
        std['salary'] = std.get('salary', 0.0); std['projection'] = std.get('projection', 0.0); std['prop_line'] = std.get('prop_line', 0.0)
        std['factor_hit_rate'] = std.get('factor_hit_rate', 50.0)
        if source_tag == 'PrizePicks': std['prizepicks_line'] = std['prop_line']
        elif source_tag == 'Underdog': std['underdog_line'] = std['prop_line']
        return std
    @staticmethod
    def merge(base, new_df):
        if base.empty: return new_df
        combined = pd.concat([base, new_df])
        agg_dict = {c: 'max' for c in combined.select_dtypes(include=np.number).columns}
        for c in ['position','team','game_info','status','opp']: 
            if c in combined.columns: agg_dict[c] = 'last'
        try: return combined.groupby(['name','sport','market'], as_index=False).agg(agg_dict)
        except: return combined.drop_duplicates(subset=['name','sport'], keep='last')

# --- AI ---
def run_web_scout(sport):
    intel = {}
    if HAS_AI:
        try:
            with DDGS() as ddgs:
                for q in [f"{sport} dfs winning strategy", f"{sport} prop sleepers"]:
                    for r in list(ddgs.text(q, max_results=2)): intel[r['title']] = 1
        except: pass
    return intel

# --- OPTIMIZER ---
def get_roster_rules(sport, site, mode):
    rules = {'size': 6, 'cap': 50000, 'constraints': []}
    if site in ['PrizePicks', 'Underdog']: rules['cap'] = 999999
    if mode == 'Showdown':
        if site == 'DK': rules.update({'size':6, 'constraints':[('CPT',1,1)]})
        elif site == 'FD': rules.update({'size':5, 'constraints':[('MVP',1,1)]})
        return rules
    if sport == 'NFL':
        if site == 'DK': rules.update({'size':9, 'constraints':[('QB',1,1),('DST',1,1),('RB',2,3),('WR',3,4),('TE',1,2)]})
        elif site == 'FD': rules.update({'size':9, 'constraints':[('QB',1,1),('DST',1,1),('RB',2,3),('WR',3,4),('TE',1,2)]})
    elif sport == 'NBA':
        if site == 'DK': rules.update({'size':8, 'constraints':[('PG',1,3),('SG',1,3),('SF',1,3),('PF',1,3),('C',1,2)]})
        elif site == 'FD': rules.update({'size':9, 'constraints':[('PG',2,2),('SG',2,2),('SF',2,2),('PF',2,2),('C',1,1)]})
    elif sport == 'CBB':
        if site == 'DK': rules.update({'size':8, 'constraints':[('G',3,5),('F',3,5)]})
    return rules

def optimize_lineup(df, config):
    target_sport = config['sport'].strip().upper()
    pool = df[df['sport'] == target_sport].copy()
    if pool.empty: st.error("No Data"); return None
    brain = TitanBrain(0)
    res = pool.apply(lambda r: brain.apply_strategic_boosts(r, target_sport), axis=1, result_type='expand')
    pool['projection'] = res[0]; pool['notes'] = res[1]
    if config['site'] not in ['PrizePicks','Underdog'] and not config.get('ignore_salary'): pool = pool[pool['salary']>0]
    if 'status' in pool.columns: pool = pool[~pool['status'].str.contains('Out|IR', case=False, na=False)]
    if config['slate_games']: pool = pool[pool['game_info'].isin(config['slate_games'])].reset_index(drop=True)
    if config['mode'] == 'Showdown':
        flex = pool.copy(); flex['pos_id'] = 'FLEX'
        cpt = pool.copy(); cpt['pos_id'] = 'CPT'; cpt['projection']*=1.5; cpt['salary']*=1.5
        pool = pd.concat([cpt, flex]).reset_index(drop=True)
    else: pool['pos_id'] = pool['position']
    rules = get_roster_rules(target_sport, config['site'], config['mode'])
    lineups = []
    for i in range(config['count']):
        prob = pulp.LpProblem("Titan", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("p", pool.index, cat='Binary')
        noise = np.random.normal(0, (100 - pool['factor_hit_rate'].fillna(50))/100.0 * 0.5, len(pool))
        prob += pulp.lpSum([(pool.loc[p, 'projection'] + noise[p]) * x[p] for p in pool.index])
        prob += pulp.lpSum([pool.loc[p, 'salary'] * x[p] for p in pool.index]) <= rules['cap']
        prob += pulp.lpSum([x[p] for p in pool.index]) == rules['size']
        for role, min_req, max_req in rules['constraints']:
            idx = pool[pool['pos_id'].str.contains(role, na=False)].index
            prob += pulp.lpSum([x[p] for p in idx]) >= min_req
            prob += pulp.lpSum([x[p] for p in idx]) <= max_req
        for lock in config['locks']:
            idx = pool[pool['name'] == lock].index
            if not idx.empty: prob += pulp.lpSum([x[p] for p in idx]) >= 1
        if config.get('smart_stack') and target_sport == 'NFL':
            qbs = pool[pool['pos_id'] == 'QB'].index
            for qb in qbs:
                team = pool.loc[qb, 'team']
                mates = pool[(pool['team']==team) & (pool['pos_id'].isin(['WR','TE']))].index
                if not mates.empty: prob += pulp.lpSum([x[m] for m in mates]) >= x[qb]
        for l in lineups: prob += pulp.lpSum([x[p] for p in l['index_id']]) <= rules['size'] - 1
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if prob.status == 1:
            sel = [p for p in pool.index if x[p].varValue == 1]
            lu = pool.loc[sel].copy(); lu['Lineup_ID'] = i+1; lu['index_id'] = sel
            lineups.append(lu)
    return pd.concat(lineups) if lineups else None

def optimize_slips(df, config):
    pool = df[df['sport'] == config['sport'].upper()].copy()
    if pool.empty: return None
    line_col = 'prop_line'
    if config['book'] == 'PrizePicks': line_col = 'prizepicks_line'
    elif config['book'] == 'Underdog': line_col = 'underdog_line'
    brain = TitanBrain(0)
    res = pool.apply(lambda r: brain.apply_strategic_boosts(r, config['sport']), axis=1, result_type='expand')
    pool['smart_proj'] = res[0]; pool['notes'] = res[1]
    def score_row(row):
        line = row.get(line_col, 0) or row['prop_line']
        if line <= 0: return 0
        edge = abs(row['smart_proj'] - line) / line
        return edge * (1.2 if row.get('factor_hit_rate',0)>60 else 1.0)
    pool['score'] = pool.apply(score_row, axis=1)
    pool = pool[pool['score'] > 0.05].sort_values('score', ascending=False)
    slips = []
    for i in range(config['count']):
        slip = []; players = set(); teams = set()
        for idx, row in pool.iterrows():
            if len(slip) >= config['legs']: break
            if row['name'] in players: continue
            if config['corr'] and row['team'] in teams: row['score'] *= 1.15
            line = row.get(line_col, 0) or row['prop_line']
            pick = "OVER" if row['smart_proj'] > line else "UNDER"
            slip.append({'Player':row['name'], 'Market':row['market'], 'Line':line, 'Pick':pick, 'Win Prob %': 50 + row['score']*40})
            players.add(row['name']); teams.add(row['team'])
        if len(slip) == config['legs']: slips.append(pd.DataFrame(slip))
        if not pool.empty: pool = pool.iloc[1:]
    return slips

# --- DASHBOARD ---
conn = init_db()
st.sidebar.title("TITAN OMNI V39")
current_bank = get_bankroll(conn)
st.sidebar.metric("Bankroll", f"${current_bank:,.2f}")
sport = st.sidebar.selectbox("Sport", ["NBA", "NFL", "MLB", "NHL", "CBB", "PGA"])
site = st.sidebar.selectbox("Site", ["DK", "FD", "PrizePicks", "Underdog"])
t1, t2, t3 = st.tabs(["1. Data", "2. Optimizer", "3. Props"])
with t1:
    c1, c2 = st.columns(2)
    with c1:
        dfs_files = st.file_uploader("DFS CSVs", accept_multiple_files=True, key="dfs")
        if st.button("Load DFS"):
            ref = DataRefinery(); new_data = pd.DataFrame()
            for f in dfs_files: new_data = ref.merge(new_data, ref.ingest(pd.read_csv(f), sport, "Generic"))
            st.session_state['dfs_pool'] = new_data; st.success(f"Loaded {len(new_data)} DFS")
    with c2:
        prop_files = st.file_uploader("Prop CSVs", accept_multiple_files=True, key="prop")
        if st.button("Load Props"):
            ref = DataRefinery(); new_data = pd.DataFrame()
            for f in prop_files: new_data = ref.merge(new_data, ref.ingest(pd.read_csv(f), sport, site))
            st.session_state['prop_pool'] = new_data; st.success(f"Loaded {len(new_data)} Props")
    if st.button("Run AI Scout"): st.success(f"Intel: {run_web_scout(sport)}")
with t2:
    pool = st.session_state['dfs_pool']
    if not pool.empty:
        mode = st.radio("Mode", ["Classic", "Showdown"], horizontal=True)
        count = st.slider("Lineups", 1, 150, 20)
        smart = st.checkbox("Milly Maker Logic", True)
        ignore_sal = st.checkbox("Ignore Salary", False)
        if st.button("Run Optimizer"):
            cfg = {'sport':sport, 'site':site, 'mode':mode, 'count':count, 'smart_stack':smart, 'locks':[], 'bans':[], 'slate_games':[], 'positions':[], 'max_exposure':100, 'ignore_salary':ignore_sal}
            res = optimize_lineup(pool, cfg)
            if res is not None: st.dataframe(res); st.download_button("Export", res.to_csv(index=False), "lineups.csv")
with t3:
    pool = st.session_state['prop_pool']
    if not pool.empty:
        legs = st.slider("Legs", 2, 6, 5)
        wager = st.number_input("Wager", 10)
        if st.button("Generate Slips"):
            slips = optimize_slips(pool, {'sport':sport, 'book':site, 'legs':legs, 'count':3, 'corr':True})
            if slips:
                for i, s in enumerate(slips):
                    brain = TitanBrain(0)
                    prob, payout, profit = brain.calculate_slip_ev(s, site, wager)
                    st.write(f"Slip {i+1}: Payout {payout}x | EV: ${profit:.2f}"); st.table(s)
