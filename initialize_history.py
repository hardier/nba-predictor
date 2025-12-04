# initialize_history.py
import requests
import pandas as pd
import time
import os

BASE_URL = "https://nbafantasy.nba.com/api"
FILENAME = "nba_history_cache_v9.csv"
PLAYER_SCAN_LIMIT = 200

def fetch_bootstrap():
    print("Fetching player list...")
    try:
        return requests.get(f"{BASE_URL}/bootstrap-static/").json()
    except Exception as e:
        print(f"Error: {e}")
        return None

def generate_history():
    data = fetch_bootstrap()
    if not data: return

    # Sort by ownership to get top players
    all_sorted = sorted(data['elements'], key=lambda x: float(x['selected_by_percent']), reverse=True)
    targets = all_sorted[:PLAYER_SCAN_LIMIT]
    
    all_history = []
    print(f"Scanning history for top {PLAYER_SCAN_LIMIT} players. This will take about 1-2 minutes...")
    
    for i, p in enumerate(targets):
        pid = p['id']
        if i % 10 == 0: print(f"Processing {i}/{PLAYER_SCAN_LIMIT}...")
            
        try:
            r = requests.get(f"{BASE_URL}/element-summary/{pid}/")
            if r.status_code == 200:
                p_data = r.json()
                hist = p_data.get('history', [])
                hist.sort(key=lambda x: x['round']) 
                
                for j in range(1, len(hist)):
                    today = hist[j]
                    prev = hist[j-1]
                    
                    actual_diff = today['value'] - prev['value']
                    
                    # Target Class Logic
                    if actual_diff > 0: target = 1
                    elif actual_diff < 0: target = -1
                    else: target = 0
                    
                    row = {
                        'player_id': pid,
                        'round': today['round'],
                        'kickoff_time': today['kickoff_time'][:10],
                        'value_start': prev['value'],
                        'actual_change_val': actual_diff,
                        'target_class': target,
                        'net_transfers': today['transfers_balance'],
                        'selected': today['selected'],
                        'points': today['total_points'],
                        'minutes': today['minutes']
                    }
                    all_history.append(row)
            # Sleep briefly to avoid API bans
            time.sleep(0.05)
        except:
            continue

    if all_history:
        df = pd.DataFrame(all_history)
        df.to_csv(FILENAME, index=False)
        print(f"SUCCESS! Saved {len(df)} records to '{FILENAME}'.")
        print("You can now restart your Streamlit app.")
    else:
        print("Failed to collect data.")

if __name__ == "__main__":
    generate_history()