import os
import pandas as pd
import pytz
from datetime import datetime
from sqlalchemy import create_engine, text

# 1. SETUP CONNECTION
# Check if we have a Cloud Database URL (from Render/Neon)
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # CLOUD MODE: PostgreSQL
    # Fix for SQLAlchemy requiring 'postgresql://' instead of 'postgres://'
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DATABASE_URL)
    print("✅ nba_db: Connected to Cloud PostgreSQL")
else:
    # LOCAL MODE: SQLite
    DB_NAME = "nba_fantasy.db"
    engine = create_engine(f"sqlite:///{DB_NAME}")
    print("✅ nba_db: Connected to Local SQLite")

def init_db():
    """Creates tables if they don't exist."""
    # We use raw SQL wrapped in text() for compatibility
    with engine.connect() as conn:
        # PostgreSQL syntax is slightly different but simple CREATE TABLE usually works cross-db
        # However, pandas 'to_sql' will create tables for us automatically!
        # We only need to ensure the connection works here.
        pass

def save_snapshot(df_players):
    """Saves current state to DB (Auto-creates table if missing)"""
    pst = pytz.timezone('US/Pacific')
    now_str = datetime.now(pst).strftime('%Y-%m-%d %H:%M:%S')
    
    # Prepare Data
    snapshot_data = []
    for _, p in df_players.iterrows():
        try: sel = float(p['selected_by_percent']) * 1000 
        except: sel = 0
        net = p['transfers_in_event'] - p['transfers_out_event']
        
        snapshot_data.append({
            'timestamp': now_str,
            'player_id': p['id'],
            'web_name': p['web_name'],
            'now_cost': p['now_cost'],
            'selected': sel,
            'net_transfers': net
        })
    
    df = pd.DataFrame(snapshot_data)
    
    # Save to SQL (appends to table 'snapshots')
    # if_exists='append' handles table creation automatically
    df.to_sql('snapshots', engine, if_exists='append', index=False)
    print(f"[{now_str}] Snapshot saved ({len(df)} records).")

def save_daily_events(events_list):
    """Saves finalized daily events"""
    if not events_list: return
    
    # Convert list of tuples to DataFrame
    cols = ['date', 'player_id', 'start_price', 'end_price', 'price_change', 
            'final_net_transfers', 'final_selected', 'target_class']
    df = pd.DataFrame(events_list, columns=cols)
    
    df.to_sql('daily_events', engine, if_exists='append', index=False)
    print(f"✅ Saved {len(df)} daily events to DB.")

def get_training_data():
    """Fetches training history"""
    try:
        query = "SELECT * FROM daily_events"
        return pd.read_sql(query, engine)
    except:
        return pd.DataFrame() # Return empty if table doesn't exist yet

def get_latest_snapshot():
    """Get the most recent snapshot data"""
    try:
        # Optimized query to find max timestamp first
        # Syntax compatible with both PG and SQLite
        query = """
            SELECT * FROM snapshots 
            WHERE timestamp = (SELECT MAX(timestamp) FROM snapshots)
        """
        return pd.read_sql(query, engine)
    except:
        return pd.DataFrame()