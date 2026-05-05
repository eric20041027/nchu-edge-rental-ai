
import pandas as pd
import os

import re

def normalize_address(address):
    """Normalize Chinese numerals and common address variations."""
    if not isinstance(address, str): return ""
    # Convert Chinese numerals to Arabic
    mapping = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5', 
               '六': '6', '七': '7', '八': '8', '九': '9', '○': '0', '零': '0'}
    for cn, ar in mapping.items():
        address = address.replace(cn, ar)
    # Remove common whitespace and punctuation
    address = re.sub(r'[\s\-,.，。]', '', address)
    return address

def merge_datasets():
    base_path = os.path.join(os.path.dirname(__file__), "../../data/raw")
    main_file = os.path.join(base_path, "nchu_rental_info.csv")
    official_file = os.path.join(base_path, "nchu_official_raw.csv")

    if not os.path.exists(official_file):
        print("Error: Official data file not found.")
        return

    df_main = pd.read_csv(main_file)
    df_official = pd.read_csv(official_file)

    print(f"Main dataset: {len(df_main)} rows")
    print(f"Official dataset: {len(df_official)} rows")

    # Combine
    df_combined = pd.concat([df_main, df_official], ignore_index=True)

    # Advanced De-duplication
    initial_count = len(df_combined)
    
    # 1. Primary check: URL
    df_combined = df_combined.drop_duplicates(subset=["網址"], keep="first")
    
    # 2. Secondary check: Normalized Address + Price Tolerance (5%)
    df_combined["norm_address"] = df_combined["地址"].apply(normalize_address)
    # Convert rent to numeric for comparison (e.g., "6500 元" -> 6500)
    def parse_rent(x):
        try: return int(re.search(r'\d+', str(x)).group())
        except: return 0
    df_combined["rent_val"] = df_combined["租金"].apply(parse_rent)

    # Sort to ensure consistent behavior
    df_combined = df_combined.sort_values(by=["norm_address", "rent_val"])
    
    final_rows = []
    seen_addresses = {} # address -> list of prices

    for _, row in df_combined.iterrows():
        addr = row["norm_address"]
        price = row["rent_val"]
        
        is_duplicate = False
        if addr in seen_addresses:
            for seen_price in seen_addresses[addr]:
                # If price is within 5% tolerance, consider it a duplicate
                if abs(price - seen_price) / (seen_price + 1e-5) < 0.05:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            final_rows.append(row)
            if addr not in seen_addresses: seen_addresses[addr] = []
            seen_addresses[addr].append(price)

    df_final = pd.DataFrame(final_rows)
    
    # Cleanup temporary columns
    df_final = df_final.drop(columns=["norm_address", "rent_val"])

    print(f"Merged count: {len(df_final)} (Removed {initial_count - len(df_final)} duplicates)")

    # Fill missing columns and reorder
    for col in df_main.columns:
        if col not in df_final.columns:
            df_final[col] = ""
    df_final = df_final[df_main.columns]

    # Save back
    df_final.to_csv(main_file, index=False, encoding='utf-8-sig')
    print(f"Success: Updated {main_file} with refined merged data.")

if __name__ == "__main__":
    merge_datasets()
