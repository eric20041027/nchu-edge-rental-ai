"""
update_commute_data.py
Uses ArcGIS Geocoding (very accurate for Taiwan) and OSRM (Open Source Routing Machine) 
for real-world road distance and time calculation.
"""
import csv
import os
import time
import requests
import urllib.parse

# NCHU Main Gate Coordinates
NCHU_LAT = 24.1252
NCHU_LON = 120.6818

def get_real_commute(address):
    """
    1. Geocodes address via ArcGIS.
    2. Uses OSRM to get real walking and driving durations.
    """
    # Step 1: Geocoding (ArcGIS)
    search_addr = address if "台中" in address else f"台中市{address}"
    # Remove floor info
    search_addr = search_addr.split('樓')[0].split('F')[0].split('f')[0]
    
    encoded_addr = urllib.parse.quote(search_addr)
    geo_url = f"https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates?address={encoded_addr}&f=json&maxLocations=1"
    
    lat, lon = None, None
    try:
        resp = requests.get(geo_url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('candidates'):
                loc = data['candidates'][0]['location']
                lat, lon = loc['y'], loc['x']
    except Exception as e:
        print(f"  Geocode error: {e}")
        return None

    if not lat: return None

    # Step 2: OSRM Routing
    # Walking (foot)
    walk_url = f"http://router.project-osrm.org/route/v1/foot/{lon},{lat};{NCHU_LON},{NCHU_LAT}?overview=false"
    # Driving/Scooter (car)
    drive_url = f"http://router.project-osrm.org/route/v1/car/{lon},{lat};{NCHU_LON},{NCHU_LAT}?overview=false"
    
    results = {"dist": 0, "walk_mins": 0, "scooter_mins": 0}
    
    try:
        # Walk
        w_resp = requests.get(walk_url, timeout=10).json()
        if w_resp.get('routes'):
            route = w_resp['routes'][0]
            results["dist"] = round(route['distance'] / 1000, 2)
            results["walk_mins"] = round(route['duration'] / 60)
            
        # Drive/Scooter
        d_resp = requests.get(drive_url, timeout=10).json()
        if d_resp.get('routes'):
            results["scooter_mins"] = round(d_resp['routes'][0]['duration'] / 60)
            # Add buffer for scooter parking/lights
            results["scooter_mins"] += 2
            
        # Safety check: if walk_mins is suspiciously low for distance
        # Average walking speed is ~12-15 mins per km
        if results["dist"] > 0:
            min_expected_walk = round(results["dist"] * 12)
            if results["walk_mins"] < min_expected_walk:
                results["walk_mins"] = min_expected_walk
                
        return results
    except Exception as e:
        print(f"  Routing error: {e}")
        return None

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "../../data/raw/nchu_rental_info.csv")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    
    fieldnames = list(rows[0].keys())
    for f in ["walk_mins", "scooter_mins"]:
        if f not in fieldnames: fieldnames.append(f)

    print(f"Updating {len(rows)} properties using ArcGIS + OSRM...")
    
    count = 0
    for row in rows:
        walk_val = row.get("walk_mins", "").strip()
        scooter_val = row.get("scooter_mins", "").strip()
        
        if walk_val and scooter_val and walk_val != "0" and scooter_val != "0":
            count += 1
            continue

        addr = row.get("地址", "")
        # Remove unwanted prefix like "402" or "台中市南區" if duplicated
        clean_addr = addr.replace("402", "").strip()
        
        print(f"[{count+1}/{len(rows)}] {clean_addr[:30]}...")
        
        data = get_real_commute(clean_addr)
        if data:
            row["距離(km)"] = str(data["dist"])
            row["walk_mins"] = str(data["walk_mins"])
            row["scooter_mins"] = str(data["scooter_mins"])
            print(f"  -> {data['dist']}km | Walk: {data['walk_mins']}m | Scooter: {data['scooter_mins']}m")
        else:
            # Better fallback based on distance if routing fails
            d = float(row.get("距離(km)", "1.0") or "1.0")
            row["walk_mins"] = str(round(d * 15))
            row["scooter_mins"] = str(round(d * 3) + 2)
            print(f"  -> Failed, estimated from distance: {row['walk_mins']}m")
            
        count += 1
        
        # Periodic Save every 10 properties
        if count % 10 == 0:
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"  [Auto-Save] Progress saved at {count}/{len(rows)}")

        time.sleep(0.5) # ArcGIS is more lenient than Nominatim

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nSuccess! Real-world commute data updated.")

if __name__ == "__main__":
    main()
