import os
import pandas as pd
import pytz
from datetime import datetime
from sqlalchemy import create_engine, text

# 1. SETUP CONNECTION
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DATABASE_URL)
    print("✅ nba_db: Connected to Cloud PostgreSQL")
else:
    DB_NAME = "nba_fantasy.db"
    engine = create_engine(f"sqlite:///{DB_NAME}")
    print("✅ nba_db: Connected to Local SQLite")

def init_db():
    pass # Tables created automatically by to_sql

def save_snapshot(df_players):
    """Saves current state with explicit IN/OUT event data"""
    pst = pytz.timezone('US/Pacific')
    now_str = datetime.now(pst).strftime('%Y-%m-%d %H:%M:%S')
    
    snapshot_data = []
    for _, p in df_players.iterrows():
        try: sel = float(p['selected_by_percent']) * 1000 
        except: sel = 0
        
        # Capture the explicit event data
        in_event = p.get('transfers_in_event', 0)
        out_event = p.get('transfers_out_event', 0)
        net = in_event - out_event
        
        snapshot_data.append({
            'timestamp': now_str,
            'player_id': p['id'],
            'web_name': p['web_name'],
            'now_cost': p['now_cost'],
            'selected': sel,
            'transfers_in_event': in_event,   # NEW COLUMN
            'transfers_out_event': out_event, # NEW COLUMN
            'net_transfers': net
        })
    
    df = pd.DataFrame(snapshot_data)
    
    # Save to SQL
    # Note: If using existing DB, this might error if columns don't exist.
    # Quick fix for SQLite/Postgres: It usually handles new columns or you might need to drop table first.
    try:
        df.to_sql('snapshots', engine, if_exists='append', index=False)
        print(f"[{now_str}] Snapshot saved ({len(df)} records).")
    except Exception as e:
        print(f"⚠️ DB Schema Mismatch (Drop old table to fix): {e}")

def save_daily_events(events_list):
    """Saves finalized daily events"""
    if not events_list: return
    
    # Columns: date, player_id, start_price, end_price, price_change, 
    #          final_net_transfers, final_selected, target_class
    # Note: We keep this simple for training, focusing on Net result
    df = pd.DataFrame(events_list, columns=[
        'date', 'player_id', 'start_price', 'end_price', 'price_change', 
        'final_net_transfers', 'final_selected', 'target_class'
    ])
    
    df.to_sql('daily_events', engine, if_exists='append', index=False)
    print(f"✅ Saved {len(df)} daily events.")

def get_training_data():
    try:
        return pd.read_sql("SELECT * FROM daily_events", engine)
    except:
        return pd.DataFrame() 

def get_latest_snapshot():
    try:
        # Get data from the very last timestamp
        query = """
            SELECT * FROM snapshots 
            WHERE timestamp = (SELECT MAX(timestamp) FROM snapshots)
        """
        return pd.read_sql(query, engine)
    except:
        return pd.DataFrame()