import requests
import pandas as pd
import os
import time
import re


# 1. Fetch base champion list
def fetch_base_champions():
    url = "https://mcp-api.op.gg/mcp"
    headers = {"Content-Type": "application/json"}
    positions = {"TOP": "Top", "JUNGLE": "Jungle", "MID": "Mid", "ADC": "Adc", "SUPPORT": "Support"}
    champ_list = []

    print("🔍 [Step 1] Fetching champion list by lane...")
    for pos_key, class_name in positions.items():
        payload = {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "lol_list_lane_meta_champions",
                "arguments": {"region": "kr", "game_mode": "ranked", "position": pos_key}
            }
        }
        try:
            response = requests.post(url, json=payload, headers=headers)
            raw_text = response.json()['result']['content'][0]['text']
            pattern = rf'{class_name}\("(.*?)",'
            matches = re.findall(pattern, raw_text)
            for m in matches:
                champ_list.append({"name": m, "position": pos_key})
            time.sleep(0.5)
        except Exception as e:
            print(f"❌ Error fetching {pos_key} list: {e}")

    print(f"✅ Successfully acquired {len(champ_list)} champion targets.\n")
    return champ_list


# 2. Advanced data collection (Robust regex parsing applied)
def fetch_advanced_data(champion_list):
    url = "https://mcp-api.op.gg/mcp"
    headers = {"Content-Type": "application/json"}
    all_rows = []

    print(f"🚀 [Step 2] Initiating advanced metadata collection...")

    for champ in champion_list:
        name = champ['name']
        pos = champ['position']

        payload = {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "lol_get_champion_analysis",
                "arguments": {
                    "region": "kr", "game_mode": "ranked",
                    "champion": name, "position": pos
                }
            }
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            res_json = response.json()

            if 'result' not in res_json:
                continue

            raw_text = res_json['result']['content'][0]['text']

            # 1. Extract win rate and tier (Defends against non-numeric values like -1, null)
            win_rate, tier = 0.0, 0
            stats_match = re.search(r'AverageStats\([^,]+,([^,]+),[^,]+,[^,]+,[^,]+,([^,]+)', raw_text)
            if stats_match:
                try:
                    win_rate = round(float(stats_match.group(1)) * 100, 2)
                    tier = int(stats_match.group(2))
                except:
                    pass  # Keep default 0.0 if garbage value

            # 2. Extract core items (Strips brackets and quotes)
            items_matches = re.findall(r'CoreItems\(\[[^\]]*\],\[(.*?)\]', raw_text)
            items_list = []
            for match in items_matches:
                names = [n.strip('\"').strip() for n in match.split(',')]
                items_list.extend([n for n in names if n])

            # Remove duplicates, extract max 5
            core_items = ", ".join(list(dict.fromkeys(items_list))[:5])

            # 3. Extract counters (Catches both StrongCounter and Counter)
            counter_matches = re.findall(r'(?:Strong)?Counter\([^,]+,"([^"]+)"', raw_text)
            hard_counters = ", ".join(list(dict.fromkeys(counter_matches))[:5])

            all_rows.append({
                "champion": name,
                "position": pos,
                "win_rate": win_rate,
                "tier": tier,
                "core_items": core_items,
                "hard_counters": hard_counters
            })

            # Print real-time status to verify data extraction
            print(f"✅ {name} ({pos}) - Win Rate: {win_rate}%, Counters: {bool(hard_counters)}")

        except Exception as e:
            print(f"❌ System error while processing {name}: {e}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        os.makedirs('data', exist_ok=True)
        output_path = 'data/master_meta.csv'
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print("\n" + "=" * 50)
        print(f"🏆 Successfully built RAG master dataset for {len(df)} champions!")
        print("=" * 50)


if __name__ == "__main__":
    champs = fetch_base_champions()
    if champs:
        fetch_advanced_data(champs)