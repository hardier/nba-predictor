import streamlit as st
import pandas as pd
import nba_db  # Handles Cloud/Local DB connection
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import requests
import os

# ==========================================
# 1. CONFIGURATION & METADATA
# ==========================================
LEGACY_CSV = "nba_history_cache_v9.csv" 
BASE_URL = "https://nbafantasy.nba.com/api"
POSITION_MAP = {1: 'BC', 2: 'FC', 3: 'FC'} 

st.set_page_config(layout="wide", page_title="NBA AI Hybrid Predictor")

@st.cache_data(ttl=3600)
def fetch_metadata():
    try:
        data = requests.get(f"{BASE_URL}/bootstrap-static/").json()
        teams = {t['id']: t['short_name'] for t in data['teams']}
        players = {}
        for p in data['elements']:
            players[p['id']] = {
                'name': p['web_name'],
                'team': teams.get(p['team'], 'UNK'),
                'pos': POSITION_MAP.get(p['element_type'], 'UNK')
            }
        return players
    except:
        return {}

meta_map = fetch_metadata()

# ==========================================
# 2. DATA LOADING
# ==========================================
def load_hybrid_training_data():
    data_frames = []
    
    # A. Legacy CSV
    if os.path.exists(LEGACY_CSV):
        try:
            df_legacy = pd.read_csv(LEGACY_CSV)
            data_frames.append(df_legacy)
        except: pass

    # B. Live Cloud DB
    try:
        df_new = nba_db.get_training_data()
        if not df_new.empty:
            df_new = df_new.rename(columns={
                'final_net_transfers': 'net_transfers', 
                'final_selected': 'selected',
                'start_price': 'value_start',
                'date': 'kickoff_time'
            })
            if 'points' not in df_new.columns: df_new['points'] = 0 
            if 'minutes' not in df_new.columns: df_new['minutes'] = 30
            df_new['target_class'] = df_new['target_class'].astype(int)
            data_frames.append(df_new)
    except: pass
        
    if not data_frames: return pd.DataFrame()
    return pd.concat(data_frames, ignore_index=True)

# ==========================================
# 3. MODEL ENGINE
# ==========================================
def train_model(history_df):
    if history_df.empty: return None
    
    features = ['net_transfers', 'selected', 'points', 'minutes', 'value_start']
    for col in features:
        if col not in history_df.columns: history_df[col] = 0

    X = history_df[features]
    y = history_df['target_class']
    
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced', 
        max_depth=6, min_samples_leaf=20 
    )
    model.fit(X, y)
    return model

def get_probabilities(model, df_features):
    if model is None: return np.zeros(len(df_features)), np.zeros(len(df_features))
    probs = model.predict_proba(df_features)
    classes = model.classes_
    
    try: idx_rise = np.where(classes == 1)[0][0]
    except: idx_rise = None
    try: idx_fall = np.where(classes == -1)[0][0]
    except: idx_fall = None
    
    p_rise = probs[:, idx_rise] if idx_rise is not None else np.zeros(len(df_features))
    p_fall = probs[:, idx_fall] if idx_fall is not None else np.zeros(len(df_features))
    return p_rise, p_fall

# ==========================================
# 4. MAIN APP UI
# ==========================================
st.title("🏀 NBA Fantasy AI (Hybrid System)")

with st.spinner("Loading hybrid data..."):
    df_train = load_hybrid_training_data()

if not df_train.empty:
    model = train_model(df_train)
    st.sidebar.success(f"🧠 AI Trained on {len(df_train)} events.")
else:
    st.error("No training data found.")
    st.stop()

# ---------------------------------------------------------
# SECTION 1: LIVE PREDICTIONS
# ---------------------------------------------------------
st.header("1. 🔮 Live Predictions (Tonight)")

conn = sqlite3.connect(nba_db.DB_NAME) if hasattr(nba_db, 'DB_NAME') else None
try:
    df_latest = nba_db.get_latest_snapshot()
    
    if not df_latest.empty:
        last_time = df_latest.iloc[0]['timestamp']
        st.caption(f"Snapshot Time: {last_time}")
        
        # Mapping & Fixing
        df_latest['value_start'] = df_latest['now_cost'] / 10.0
        df_latest['points'] = 15
        df_latest['minutes'] = 30
        
        # Map Metadata
        df_latest['Team'] = df_latest['player_id'].map(lambda x: meta_map.get(x, {}).get('team', 'UNK'))
        df_latest['Pos'] = df_latest['player_id'].map(lambda x: meta_map.get(x, {}).get('pos', 'UNK'))

        # Predict
        feats = ['net_transfers', 'selected', 'points', 'minutes', 'value_start']
        p_rise, p_fall = get_probabilities(model, df_latest[feats])
        
        df_latest['Prob_Rise'] = p_rise * 100
        df_latest['Prob_Fall'] = p_fall * 100
        
        # Highlight Hot Players
        def add_hot_icon(row, col_name):
            if row[col_name] >= 60:
                return f"🔥 {row['web_name']}"
            return row['web_name']

        df_latest['Display_Name_Rise'] = df_latest.apply(lambda r: add_hot_icon(r, 'Prob_Rise'), axis=1)
        df_latest['Display_Name_Fall'] = df_latest.apply(lambda r: add_hot_icon(r, 'Prob_Fall'), axis=1)

        # Configs
        prog_config = st.column_config.ProgressColumn("Confidence", format="%.1f%%", min_value=0, max_value=100)
        curr_config = st.column_config.NumberColumn("Price", format="$%.1f")
        net_config = st.column_config.NumberColumn("Net Transfers", format="%d")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 Predicted Risers")
            risers = df_latest[df_latest['Prob_Rise'] > 10].sort_values(by='Prob_Rise', ascending=False).head(10)
            st.dataframe(
                risers[['Display_Name_Rise', 'Team', 'net_transfers', 'value_start', 'Prob_Rise']],
                column_config={
                    "Display_Name_Rise": "Player",
                    "net_transfers": net_config,
                    "value_start": curr_config, 
                    "Prob_Rise": prog_config
                },
                hide_index=True, use_container_width=True
            )
            
        with col2:
            st.subheader("📉 Predicted Fallers")
            fallers = df_latest[df_latest['Prob_Fall'] > 10].sort_values(by='Prob_Fall', ascending=False).head(10)
            st.dataframe(
                fallers[['Display_Name_Fall', 'Team', 'net_transfers', 'value_start', 'Prob_Fall']],
                column_config={
                    "Display_Name_Fall": "Player",
                    "net_transfers": net_config,
                    "value_start": curr_config, 
                    "Prob_Fall": prog_config
                },
                hide_index=True, use_container_width=True
            )
    else:
        st.info("Waiting for Scheduler...")
except Exception as e:
    st.error(f"DB Error: {e}")

st.divider()

# ---------------------------------------------------------
# SECTION 2: HISTORY TIME MACHINE
# ---------------------------------------------------------
st.header("2. 🕰️ Time Machine (Validation)")

if not df_train.empty:
    df_train['kickoff_time'] = df_train['kickoff_time'].astype(str).str.slice(0, 10)
    dates = sorted(df_train['kickoff_time'].unique(), reverse=True)
    sel_date = st.selectbox("Select Past Date", dates)
    
    # Filter & Predict
    day_df = df_train[df_train['kickoff_time'] == sel_date].copy()
    feats = ['net_transfers', 'selected', 'points', 'minutes', 'value_start']
    for f in feats: 
        if f not in day_df.columns: day_df[f] = 0
            
    p_rise, p_fall = get_probabilities(model, day_df[feats])
    day_df['AI_Rise'] = p_rise * 100
    day_df['AI_Fall'] = p_fall * 100
    
    day_df['Name'] = day_df['player_id'].map(lambda x: meta_map.get(x, {}).get('name', f"ID {x}"))
    day_df['Team'] = day_df['player_id'].map(lambda x: meta_map.get(x, {}).get('team', 'UNK'))
    day_df['Pos'] = day_df['player_id'].map(lambda x: meta_map.get(x, {}).get('pos', 'UNK'))
    
    # Configs for History
    net_config_hist = st.column_config.NumberColumn("Net Transfers", format="%d")
    
    # --- A. HITS ---
    st.subheader("A. Correct Predictions (Hits >= 60%)")
    def get_hit_marker(row):
        if row['AI_Rise'] >= 60 and row['actual_change_val'] > 0: return "✅ HIT (+0.1)"
        if row['AI_Fall'] >= 60 and row['actual_change_val'] < 0: return "✅ HIT (-0.1)"
        return None

    day_df['Hit_Status'] = day_df.apply(get_hit_marker, axis=1)
    hits = day_df[day_df['Hit_Status'].notnull()].copy()
    hits['Confidence'] = np.where(hits['actual_change_val'] > 0, hits['AI_Rise'], hits['AI_Fall'])
    hits = hits.sort_values(by='Confidence', ascending=False).head(10)
    
    if not hits.empty:
        st.dataframe(
            hits[['Name', 'Team', 'Pos', 'net_transfers', 'Hit_Status', 'Confidence']],
            column_config={
                "net_transfers": net_config_hist,
                "Confidence": prog_config
            },
            hide_index=True, use_container_width=True
        )
    else:
        st.info("No correct high-confidence (>=60%) predictions.")

    # --- B. MISSED ---
    st.subheader("B. ⚠️ Missed by AI (Underconfident)")
    missed = day_df[
        (day_df['actual_change_val'] != 0) & 
        (day_df['Hit_Status'].isnull())
    ].copy()
    
    if not missed.empty:
        missed['AI_Predicted_Chance'] = np.where(missed['actual_change_val'] > 0, missed['AI_Rise'], missed['AI_Fall'])
        missed['Actual Move'] = np.where(missed['actual_change_val'] > 0, "📈 ROSE", "📉 FELL")
        missed = missed.sort_values(by='AI_Predicted_Chance', ascending=False).head(10)
        
        st.dataframe(
            missed[['Name', 'Team', 'Pos', 'net_transfers', 'Actual Move', 'AI_Predicted_Chance']],
            column_config={
                "net_transfers": net_config_hist,
                "AI_Predicted_Chance": prog_config
            },
            hide_index=True, use_container_width=True
        )
    else:
        st.success("AI caught all moves with high confidence!")

    # --- C. FALSE ALARMS ---
    st.subheader("C. ⚠️ False Alarms")
    false_alarms = day_df[
        (day_df['actual_change_val'] == 0) & 
        ((day_df['AI_Rise'] >= 60) | (day_df['AI_Fall'] >= 60))
    ].copy()
    
    if not false_alarms.empty:
        false_alarms['Predicted'] = np.where(false_alarms['AI_Rise'] > false_alarms['AI_Fall'], "📈 Rise", "📉 Fall")
        false_alarms['Confidence'] = np.where(false_alarms['AI_Rise'] > false_alarms['AI_Fall'], false_alarms['AI_Rise'], false_alarms['AI_Fall'])
        false_alarms = false_alarms.sort_values(by='Confidence', ascending=False).head(10)
        
        st.dataframe(
            false_alarms[['Name', 'Team', 'Pos', 'net_transfers', 'Predicted', 'Confidence']],
            column_config={
                "net_transfers": net_config_hist,
                "Confidence": prog_config
            },
            hide_index=True, use_container_width=True
        )
    else:
        st.success("No false alarms (>=60%).")