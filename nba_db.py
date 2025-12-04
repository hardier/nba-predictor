import sqlite3
import pandas as pd
from datetime import datetime
import pytz

DB_NAME = "nba_fantasy.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Table 1: High Frequency Snapshots (Every 15 mins)
    # Stores the state of a player at a specific time
    c.execute('''CREATE TABLE IF NOT EXISTS snapshots (
                    timestamp DATETIME,
                    player_id INTEGER,
                    web_name TEXT,
                    now_cost REAL,
                    selected INTEGER,
                    net_transfers INTEGER
                )''')

    # Table 2: Daily Finalized Events (The "Truth" for training)
    # Created after 8:00 PM to record exactly what happened
    c.execute('''CREATE TABLE IF NOT EXISTS daily_events (
                    date DATE,
                    player_id INTEGER,
                    start_price REAL,
                    end_price REAL,
                    price_change REAL,
                    final_net_transfers INTEGER,
                    final_selected INTEGER,
                    target_class INTEGER
                )''')
    
    conn.commit()
    conn.close()

def save_snapshot(df_players):
    """Saves current state of all players to DB"""
    conn = sqlite3.connect(DB_NAME)
    # Add timestamp
    pst = pytz.timezone('US/Pacific')
    now = datetime.now(pst).strftime('%Y-%m-%d %H:%M:%S')
    
    data = []
    for _, p in df_players.iterrows():
        try: sel = float(p['selected_by_percent']) * 1000 
        except: sel = 0
        net = p['transfers_in_event'] - p['transfers_out_event']
        
        data.append((now, p['id'], p['web_name'], p['now_cost'], sel, net))
        
    c = conn.cursor()
    c.executemany('INSERT INTO snapshots VALUES (?,?,?,?,?,?)', data)
    conn.commit()
    conn.close()
    print(f"[{now}] Snapshot saved for {len(data)} players.")

def get_training_data():
    """Fetches the daily events for model training"""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM daily_events", conn)
    conn.close()
    return df

# Initialize on first run
if __name__ == "__main__":
    init_db()
    print("Database initialized.")