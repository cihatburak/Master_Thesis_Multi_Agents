"""
Dataset Quality Cleanup Script
Performs comprehensive data cleaning on dataset_final.json:
1. Remove duplicate-review products (keep only one representative)
2. Remove cross-product duplicate reviews
3. Filter out very short reviews (<50 chars) - too little info for BI analysis
4. Fix broken/missing product descriptions using title+specs  
5. Re-run ingest.py after cleanup

Usage: python cleanup_dataset.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "dataset_final.json"

# Minimum review length (chars) - reviews shorter than this are filtered
MIN_REVIEW_LENGTH = 50


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(data):
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def step1_remove_duplicate_review_products(data):
    """
    Products B0FHCYNG34 and B0FHD45Q1F are config variants 
    (16GB vs 32GB RAM) of the same MSI Thin laptop and share
    ALL 12 reviews with B0F538NQPH. 
    
    Strategy: 
    - Keep B0F538NQPH (Intel i5 variant, has unique description)
    - Keep B0FHD45Q1F (AMD Ryzen 5, 16GB, $899 — mid-tier)  
    - Remove B0FHCYNG34 (AMD Ryzen 5, 32GB, $1039 — same as B0FHD45Q1F but more RAM)
    
    Since B0FHD45Q1F and B0F538NQPH share reviews, we need to 
    clear reviews from one. Keep reviews on B0FHD45Q1F (has more 
    complete description) and B0F538NQPH as separate products.
    Actually — all 3 share the EXACT same 12 reviews.
    Best approach: keep only B0F538NQPH (unique Intel CPU, good description)
    and remove the other two since they're AMD variants with identical reviews.
    """
    remove_asins = {"B0FHCYNG34"}  # Remove 32GB variant (duplicate of B0FHD45Q1F)
    
    before = len(data)
    data = [p for p in data if p["id"] not in remove_asins]
    removed = before - len(data)
    
    # For B0FHD45Q1F and B0F538NQPH that still share reviews:
    # Mark reviews to be deduplicated in step 2
    
    print(f"  Removed {removed} duplicate-review product(s): {remove_asins}")
    return data


def step2_remove_cross_product_duplicate_reviews(data):
    """
    Remove reviews that appear identically across multiple products.
    Keep the review only on the first product alphabetically.
    """
    # Build map: review_text → list of (product_idx, review_idx)
    review_locations = defaultdict(list)
    for pi, p in enumerate(data):
        for ri, r in enumerate(p.get("reviews", [])):
            text = r.get("text", "").strip()
            if len(text) > 30:
                review_locations[text].append((pi, ri))
    
    # Find reviews appearing in multiple products
    to_remove = []  # (product_idx, review_idx) tuples
    for text, locations in review_locations.items():
        product_ids = set(data[pi]["id"] for pi, _ in locations)
        if len(product_ids) > 1:
            # Keep on first product alphabetically, remove from others
            sorted_locs = sorted(locations, key=lambda x: data[x[0]]["id"])
            for pi, ri in sorted_locs[1:]:  # skip first, remove rest
                to_remove.append((pi, ri, data[pi]["id"], text[:60]))
    
    # Remove in reverse order to preserve indices
    removed_per_product = defaultdict(int)
    for pi, ri, asin, _ in sorted(to_remove, key=lambda x: (x[0], x[1]), reverse=True):
        del data[pi]["reviews"][ri]
        removed_per_product[asin] += 1
    
    # Update review counts
    for p in data:
        p["stats"]["review_count"] = len(p.get("reviews", []))
    
    total_removed = sum(removed_per_product.values())
    print(f"  Removed {total_removed} cross-product duplicate reviews")
    for asin, count in sorted(removed_per_product.items()):
        print(f"    {asin}: -{count} reviews")
    
    return data


def step3_filter_short_reviews(data):
    """Remove reviews shorter than MIN_REVIEW_LENGTH characters."""
    total_removed = 0
    for p in data:
        before = len(p.get("reviews", []))
        p["reviews"] = [
            r for r in p.get("reviews", [])
            if len(r.get("text", "").strip()) >= MIN_REVIEW_LENGTH
        ]
        removed = before - len(p["reviews"])
        if removed > 0:
            total_removed += removed
        p["stats"]["review_count"] = len(p["reviews"])
    
    print(f"  Removed {total_removed} reviews shorter than {MIN_REVIEW_LENGTH} chars")
    return data


def step4_fix_broken_descriptions(data):
    """
    Fix products with broken/missing descriptions.
    Use the product title as a base and extract specs from it.
    """
    broken_descs = {
        "B0FG3GDG64": (
            "Lenovo LOQ 15 15ARP9 Gaming Laptop features a 15.6\" FHD 144Hz display, "
            "AMD Ryzen 7 7435HS processor, NVIDIA GeForce RTX 4060 graphics, 32GB DDR5 RAM, "
            "1TB SSD storage, backlit keyboard, and RJ-45 ethernet port. "
            "Designed for high-performance gaming and multitasking with fast DDR5 memory "
            "and a dedicated GPU. Comes bundled with a PCO Laptop Cooler."
        ),
        "B0FH9P1XJV": (
            "HP Victus 15.6\" 144Hz Full HD Gaming Laptop powered by AMD Ryzen 7 7445HS processor "
            "and NVIDIA GeForce RTX 4050 graphics card. Features 16GB DDR5 RAM, 512GB SSD storage, "
            "Copilot integration, backlit keyboard, and Windows 11 Home. "
            "Mica Silver finish with Wi-Fi 6E connectivity. Budget-friendly gaming laptop "
            "suitable for gaming and everyday computing tasks. Bundle includes mouse pad."
        ),
        "B0CCXHXZVP": (
            "HP Victus 15.6\" Full HD 144Hz Gaming Laptop with AMD Ryzen 5 7535HS processor "
            "and NVIDIA GeForce RTX 4050 graphics. Equipped with 8GB DDR5 RAM and 512GB PCIe "
            "Gen 4 NVMe SSD. Features a backlit keyboard, webcam, HDMI, USB-C, Wi-Fi 6, "
            "Bluetooth 5.3, and Windows 11 Home. Natural Silver color with dedicated GPU "
            "for smooth 1080p gaming performance."
        ),
        "B0CSY75ZCN": (
            "MSI Thin 15 B13VE-1697CA 15.6\" 144Hz Gaming Laptop with Intel Core i5-13420H "
            "processor and NVIDIA GeForce RTX 4050 graphics. Features 16GB DDR4 RAM and "
            "512GB NVMe SSD storage. 144Hz Full HD IPS display for smooth gaming visuals. "
            "Thin and portable design suitable for gaming and productivity."
        ),
        "B09X8KFRB4": (
            "HP Victus 15 Gaming Laptop with NVIDIA GeForce RTX 3050 GPU and 12th Gen "
            "Intel Core i5-12500H processor. Features 8GB RAM, 512GB SSD, 15.6\" Full HD "
            "display with enhanced thermals. Includes backlit keyboard, Wi-Fi 6, "
            "and Windows 11 Home. A budget-friendly entry point for 1080p gaming."
        ),
        "B0DN5RWNNC": (
            "HP Victus 15.6\" FHD 144Hz Gaming Laptop powered by Intel Core i5-12450HX "
            "processor and NVIDIA GeForce RTX 4050 graphics. Features 16GB DDR4 RAM, "
            "1TB PCIe NVMe SSD, backlit keyboard, USB-C, HDMI, and Windows 11 Home. "
            "Gray color finish. Designed for gaming and content creation."
        ),
        "B0G4H2K18L": (
            "Acer Nitro V 17\" Gaming Laptop with Intel Core i5-12500H processor and "
            "NVIDIA GeForce RTX 4050 GPU. Features a 17\" FHD IPS display with 144Hz "
            "refresh rate, 16GB DDR5 RAM, 512GB Gen 4 SSD, Wi-Fi 6, and backlit keyboard. "
            "Larger 17-inch screen for immersive gaming and productivity."
        ),
    }
    
    fixed = 0
    for p in data:
        if p["id"] in broken_descs:
            p["metadata"]["description"] = broken_descs[p["id"]]
            fixed += 1
            print(f"    ✅ {p['id']}: description fixed ({len(broken_descs[p['id']])} chars)")
    
    print(f"  Fixed {fixed} broken descriptions")
    return data


def step5_remove_intra_product_duplicate_reviews(data):
    """Remove reviews that appear multiple times within the same product."""
    total_removed = 0
    for p in data:
        seen = set()
        unique_reviews = []
        for r in p.get("reviews", []):
            text = r.get("text", "").strip()
            if text not in seen:
                seen.add(text)
                unique_reviews.append(r)
            else:
                total_removed += 1
        p["reviews"] = unique_reviews
        p["stats"]["review_count"] = len(unique_reviews)
    
    print(f"  Removed {total_removed} intra-product duplicate reviews")
    return data


def print_summary(data):
    """Print final dataset summary."""
    total_products = len(data)
    with_reviews = sum(1 for p in data if len(p.get("reviews", [])) > 0)
    total_reviews = sum(len(p.get("reviews", [])) for p in data)
    
    review_lengths = []
    for p in data:
        for r in p.get("reviews", []):
            review_lengths.append(len(r.get("text", "")))
    
    avg_len = sum(review_lengths) / len(review_lengths) if review_lengths else 0
    min_len = min(review_lengths) if review_lengths else 0
    
    print(f"\n  📦 Total products:        {total_products}")
    print(f"  📝 Products with reviews: {with_reviews}")
    print(f"  📝 Products w/o reviews:  {total_products - with_reviews}")
    print(f"  💬 Total reviews:         {total_reviews}")
    print(f"  📏 Avg review length:     {avg_len:.0f} chars")
    print(f"  📏 Min review length:     {min_len} chars")
    
    # Review count distribution
    review_counts = [len(p.get("reviews", [])) for p in data if len(p.get("reviews", [])) > 0]
    print(f"  📊 Reviews/product:       min={min(review_counts)}, max={max(review_counts)}, avg={sum(review_counts)/len(review_counts):.1f}")


def main():
    print("=" * 60)
    print("DATASET QUALITY CLEANUP")
    print("=" * 60)

    data = load_dataset()
    total_before = len(data)
    reviews_before = sum(len(p.get("reviews", [])) for p in data)
    
    print(f"\n📊 Before: {total_before} products, {reviews_before} reviews")

    print(f"\n[1/5] Removing duplicate-review products...")
    data = step1_remove_duplicate_review_products(data)

    print(f"\n[2/5] Removing cross-product duplicate reviews...")
    data = step2_remove_cross_product_duplicate_reviews(data)

    print(f"\n[3/5] Removing intra-product duplicate reviews...")
    data = step5_remove_intra_product_duplicate_reviews(data)

    print(f"\n[4/5] Filtering short reviews (<{MIN_REVIEW_LENGTH} chars)...")
    data = step3_filter_short_reviews(data)

    print(f"\n[5/5] Fixing broken descriptions...")
    data = step4_fix_broken_descriptions(data)

    save_dataset(data)
    
    reviews_after = sum(len(p.get("reviews", [])) for p in data)
    
    print(f"\n{'=' * 60}")
    print("CLEANUP COMPLETE")
    print("=" * 60)
    print_summary(data)
    print(f"\n  📉 Reviews removed:       {reviews_before - reviews_after}")
    print(f"  📉 Products removed:      {total_before - len(data)}")
    print(f"\n  💾 Saved to: {DATASET_PATH}")
    print(f"\n  ⚠️  Next: Run 'python ingest.py' to rebuild ChromaDB")
    print("=" * 60)


if __name__ == "__main__":
    main()
