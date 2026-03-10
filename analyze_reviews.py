"""
RAW Dataset Review Analysis Script
Analyzes the ORIGINAL review counts from Excel files (before sampling)
"""

import pandas as pd
import os
import re
import glob

INPUT_FOLDER = "csv_uploads"

def extract_asin_from_filename(filename):
    """Extract ASIN from filename"""
    match = re.search(r'([A-Z0-9]{10})', filename)
    return match.group(1) if match else None

def extract_rating(val):
    """Parse rating from various formats"""
    try:
        if isinstance(val, (int, float)):
            return int(val)
        text = str(val).strip()
        if text.isdigit():
            return int(text)
        match = re.search(r'(\d)\.0 out of 5 stars', text)
        if match:
            return int(match.group(1))
        return 0
    except:
        return 0

print("=" * 70)
print("RAW DATA REVIEW ANALYSIS (Before Sampling)")
print("=" * 70)
print(f"\n📂 Scanning folder: {INPUT_FOLDER}\n")

# Scan all Excel files
files = glob.glob(os.path.join(INPUT_FOLDER, "*.xlsx"))
print(f"📄 Found {len(files)} Excel files\n")

# Count reviews per ASIN
asin_counts = {}

for filepath in files:
    filename = os.path.basename(filepath)
    asin = extract_asin_from_filename(filename)
    
    if not asin:
        continue
    
    try:
        df = pd.read_excel(filepath)
        
        # Find rating and body columns
        rating_col = None
        body_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if "rating" in col_lower:
                rating_col = col
            elif "content" in col_lower or "body" in col_lower or "review" in col_lower:
                body_col = col
        
        if not rating_col or not body_col:
            continue
        
        # Count valid reviews (rating > 0, text > 30 chars)
        valid_count = 0
        total_rows = len(df)
        
        for _, row in df.iterrows():
            rating = extract_rating(row[rating_col])
            body = str(row[body_col]) if pd.notna(row[body_col]) else ""
            if rating > 0 and len(body.strip()) > 30:
                valid_count += 1
        
        asin_counts[asin] = {
            "filename": filename,
            "total_rows": total_rows,
            "valid_reviews": valid_count
        }
        
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")

# Sort by valid reviews (descending)
sorted_products = sorted(asin_counts.items(), key=lambda x: x[1]["valid_reviews"], reverse=True)

print("-" * 70)
print("PRODUCTS SORTED BY RAW REVIEW COUNT (Descending)")
print("-" * 70)

for i, (asin, data) in enumerate(sorted_products):
    print(f"{i+1:2}. {asin} | {data['valid_reviews']:4} valid | {data['total_rows']:4} total | {data['filename'][:40]}")

# Stats
valid_counts = [d["valid_reviews"] for _, d in sorted_products]
print("\n" + "-" * 70)
print("SUMMARY STATISTICS")
print("-" * 70)
print(f"   Total products: {len(sorted_products)}")
print(f"   Minimum reviews: {min(valid_counts) if valid_counts else 0}")
print(f"   Maximum reviews: {max(valid_counts) if valid_counts else 0}")
print(f"   Average reviews: {sum(valid_counts)/len(valid_counts):.1f}" if valid_counts else "   N/A")
print(f"   Total reviews:   {sum(valid_counts)}")

# Top and bottom products
if sorted_products:
    print("\n" + "=" * 70)
    top = sorted_products[0]
    print(f"� MOST REVIEWS:  {top[0]} with {top[1]['valid_reviews']} reviews")
    
    bottom = sorted_products[-1]
    print(f"🔴 LEAST REVIEWS: {bottom[0]} with {bottom[1]['valid_reviews']} reviews")
    print("=" * 70)
