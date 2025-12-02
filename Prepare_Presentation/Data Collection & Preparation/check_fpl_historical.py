import requests
import json

# Check current season gameweeks
url = "https://fantasy.premierleague.com/api/bootstrap-static/"
response = requests.get(url)
data = response.json()

events = data['events']
print(f"Total gameweeks defined: {len(events)}")
completed = sum(1 for e in events if e['finished'])
print(f"Completed gameweeks: {completed}")

current_gw = next((e['id'] for e in events if e.get('is_current')), None)
print(f"Current gameweek: {current_gw}")

# Check a player's historical data
player_id = 1  # Random player
url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
response = requests.get(url)
player_data = response.json()

print(f"\nPlayer history available:")
print(f"  Current season: {len(player_data['history'])} gameweeks")
print(f"  Past seasons summary: {len(player_data['history_past'])} seasons")

# Show gameweek range
if len(player_data['history']) > 0:
    history = player_data['history']
    gw_start = history[0]['round']
    gw_end = history[-1]['round']
    print(f"  Gameweek range: GW{gw_start} - GW{gw_end}")
    print(f"  Total detailed records: {len(history)} gameweeks")
