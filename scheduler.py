import time
import schedule
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
import nba_db  # Import our DB manager

BASE_URL = "https://nbafantasy.nba.com/api/bootstrap-static/"

# Global variable to hold the "Pre-Deadline" state
pre_deadline_state = {}

def fetch_api():
    try:
        r = requests.get(BASE_URL, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return pd.DataFrame(data['elements'])
    except Exception as e:
        print(f"API Error: {e}")
    return pd.DataFrame()

def job_15min_snapshot():
    """Runs every 15 mins to save transfer trends"""
    df = fetch_api()
    if not df.empty:
        nba_db.save_snapshot(df)

def job_pre_deadline_lock():
    """Runs at 7:55 PM PST: Locks the 'Before' prices"""
    global pre_deadline_state
    print("🔒 Locking Pre-Deadline Prices...")
    df = fetch_api()
    if not df.empty:
        # Save to memory for comparison later
        pre_deadline_state = {
            row['id']: {
                'cost': row['now_cost'], 
                'transfers': row['transfers_in_event'] - row['transfers_out_event'],
                'selected': row['selected_by_percent']
            } 
            for _, row in df.iterrows()
        }
        # Also save to DB as a snapshot
        nba_db.save_snapshot(df)

def job_post_deadline_check():
    """Runs at 8:15 PM PST: Checks who ACTUALLY changed"""
    global pre_deadline_state
    print("🔓 Checking Post-Deadline Changes...")
    
    if not pre_deadline_state:
        print("⚠️ No pre-deadline state found. Skipping comparison.")
        return

    df_now = fetch_api()
    if df_now.empty: return

    conn = nba_db.sqlite3.connect(nba_db.DB_NAME)
    c = conn.cursor()
    
    pst = pytz.timezone('US/Pacific')
    today_str = datetime.now(pst).strftime('%Y-%m-%d')
    
    events = []
    
    for _, row in df_now.iterrows():
        pid = row['id']
        if pid in pre_deadline_state:
            old_data = pre_deadline_state[pid]
            
            start_price = old_data['cost']
            end_price = row['now_cost']
            price_change = end_price - start_price
            
            # Target Class: 1 (Rise), -1 (Fall), 0 (Same)
            if price_change > 0: target = 1
            elif price_change < 0: target = -1
            else: target = 0
            
            # Only save if we have valid transfer data
            try: sel = float(old_data['selected']) * 1000
            except: sel = 0
            
            events.append((
                today_str,
                pid,
                start_price,
                end_price,
                price_change,
                old_data['transfers'], # Use the transfers known BEFORE the change
                sel,
                target
            ))

    c.executemany('INSERT INTO daily_events VALUES (?,?,?,?,?,?,?,?)', events)
    conn.commit()
    conn.close()
    print(f"✅ Daily processing complete. {len(events)} records saved.")
    
    # Clear memory
    pre_deadline_state = {}

# --- SCHEDULING ---
# Convert PST times to server time (assuming server is UTC? Adjust as needed)
# Easier approach: Check time in loop

print("🚀 Scheduler Started...")
nba_db.init_db()

while True:
    pst = pytz.timezone('US/Pacific')
    now = datetime.now(pst)
    current_time_str = now.strftime("%H:%M")
    
    # 1. 15-Minute Snapshot (Approx)
    if now.minute % 15 == 0 and now.second < 10:
        job_15min_snapshot()
        time.sleep(60) # Sleep to avoid double run
        
    # 2. Pre-Deadline Lock (7:55 PM PST)
    if current_time_str == "19:55":
        job_pre_deadline_lock()
        time.sleep(60)
        
    # 3. Post-Deadline Check (8:15 PM PST - giving 15 mins for API to update)
    if current_time_str == "20:15":
        job_post_deadline_check()
        time.sleep(60)
        
    time.sleep(1)