"""Dataset hygiene pass for dataset_final.json.

Removes duplicate-review product variants, drops cross- and intra-product
duplicate review texts, filters reviews shorter than MIN_REVIEW_LENGTH, and
patches a known set of broken product descriptions. Re-run ingest.py
afterwards to rebuild ChromaDB.

Usage: python cleanup_dataset.py
"""

import json
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "dataset_final.json"

MIN_REVIEW_LENGTH = 50


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(data):
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def step1_remove_duplicate_review_products(data):
    # B0FHCYNG34 (32GB AMD variant) shares all 12 reviews with B0FHD45Q1F (16GB AMD)
    # and B0F538NQPH (Intel). We keep B0F538NQPH and B0FHD45Q1F as distinct products
    # because their specs differ; B0FHCYNG34 only differs from B0FHD45Q1F by RAM
    # and adds no new review evidence, so it is dropped.
    remove_asins = {"B0FHCYNG34"}
    before = len(data)
    data = [p for p in data if p["id"] not in remove_asins]
    removed = before - len(data)
    print(f"  Removed {removed} duplicate-review product(s): {remove_asins}")
    return data


def step2_remove_cross_product_duplicate_reviews(data):
    """Reviews appearing on multiple products are kept only on the first ASIN alphabetically."""
    review_locations = defaultdict(list)
    for pi, p in enumerate(data):
        for ri, r in enumerate(p.get("reviews", [])):
            text = r.get("text", "").strip()
            if len(text) > 30:
                review_locations[text].append((pi, ri))

    to_remove = []
    for text, locations in review_locations.items():
        product_ids = set(data[pi]["id"] for pi, _ in locations)
        if len(product_ids) > 1:
            sorted_locs = sorted(locations, key=lambda x: data[x[0]]["id"])
            for pi, ri in sorted_locs[1:]:
                to_remove.append((pi, ri, data[pi]["id"], text[:60]))

    # Remove in reverse order so earlier indices stay valid.
    removed_per_product = defaultdict(int)
    for pi, ri, asin, _ in sorted(to_remove, key=lambda x: (x[0], x[1]), reverse=True):
        del data[pi]["reviews"][ri]
        removed_per_product[asin] += 1

    for p in data:
        p["stats"]["review_count"] = len(p.get("reviews", []))

    total_removed = sum(removed_per_product.values())
    print(f"  Removed {total_removed} cross-product duplicate reviews")
    for asin, count in sorted(removed_per_product.items()):
        print(f"    {asin}: -{count} reviews")

    return data


def step3_remove_intra_product_duplicate_reviews(data):
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


def step4_filter_short_reviews(data):
    total_removed = 0
    for p in data:
        before = len(p.get("reviews", []))
        p["reviews"] = [
            r for r in p.get("reviews", [])
            if len(r.get("text", "").strip()) >= MIN_REVIEW_LENGTH
        ]
        total_removed += before - len(p["reviews"])
        p["stats"]["review_count"] = len(p["reviews"])

    print(f"  Removed {total_removed} reviews shorter than {MIN_REVIEW_LENGTH} chars")
    return data


def step5_fix_broken_descriptions(data):
    """Patch a known set of products whose scraped descriptions came back empty or garbled."""
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
            print(f"    {p['id']}: description fixed ({len(broken_descs[p['id']])} chars)")

    print(f"  Fixed {fixed} broken descriptions")
    return data


def print_summary(data):
    total_products = len(data)
    with_reviews = sum(1 for p in data if len(p.get("reviews", [])) > 0)
    total_reviews = sum(len(p.get("reviews", [])) for p in data)

    review_lengths = [len(r.get("text", "")) for p in data for r in p.get("reviews", [])]
    avg_len = sum(review_lengths) / len(review_lengths) if review_lengths else 0
    min_len = min(review_lengths) if review_lengths else 0

    print(f"\n  Total products:        {total_products}")
    print(f"  Products with reviews: {with_reviews}")
    print(f"  Products w/o reviews:  {total_products - with_reviews}")
    print(f"  Total reviews:         {total_reviews}")
    print(f"  Avg review length:     {avg_len:.0f} chars")
    print(f"  Min review length:     {min_len} chars")

    review_counts = [len(p.get("reviews", [])) for p in data if len(p.get("reviews", [])) > 0]
    if review_counts:
        print(f"  Reviews/product:       min={min(review_counts)}, max={max(review_counts)}, avg={sum(review_counts)/len(review_counts):.1f}")


def main():
    print("Dataset cleanup")
    print("=" * 60)

    data = load_dataset()
    total_before = len(data)
    reviews_before = sum(len(p.get("reviews", [])) for p in data)
    print(f"\nBefore: {total_before} products, {reviews_before} reviews")

    print("\n[1/5] Removing duplicate-review products...")
    data = step1_remove_duplicate_review_products(data)

    print("\n[2/5] Removing cross-product duplicate reviews...")
    data = step2_remove_cross_product_duplicate_reviews(data)

    print("\n[3/5] Removing intra-product duplicate reviews...")
    data = step3_remove_intra_product_duplicate_reviews(data)

    print(f"\n[4/5] Filtering short reviews (<{MIN_REVIEW_LENGTH} chars)...")
    data = step4_filter_short_reviews(data)

    print("\n[5/5] Fixing broken descriptions...")
    data = step5_fix_broken_descriptions(data)

    save_dataset(data)
    reviews_after = sum(len(p.get("reviews", [])) for p in data)

    print(f"\n{'=' * 60}\nDone")
    print_summary(data)
    print(f"\n  Reviews removed:       {reviews_before - reviews_after}")
    print(f"  Products removed:      {total_before - len(data)}")
    print(f"\n  Saved to: {DATASET_PATH}")
    print(f"\n  Next: run 'python ingest.py' to rebuild ChromaDB")


if __name__ == "__main__":
    main()
