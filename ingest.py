import json
from pathlib import Path

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings


def load_dataset(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_products_with_reviews(products: list[dict]) -> list[dict]:
    return [p for p in products if p.get("stats", {}).get("review_count", 0) > 0]


def get_sentiment(rating: int) -> str:
    """Star rating to sentiment label: >=4 positive, <=3 negative."""
    return "positive" if rating >= 4 else "negative"


def create_spec_document(product: dict) -> str:
    metadata = product.get("metadata", {})
    title = metadata.get("title", "")
    price = metadata.get("price", "")
    description = metadata.get("description", "")
    return f"Title: {title}\nPrice: ${price}\nDescription: {description}"


def main():
    dataset_path = Path(__file__).parent / "dataset_final.json"
    chroma_path = Path(__file__).parent / "chroma_db"
    embedding_model = "all-MiniLM-L6-v2"

    print("Loading dataset...")
    products = load_dataset(str(dataset_path))
    print(f"  Total products: {len(products)}")

    products_with_reviews = filter_products_with_reviews(products)
    print(f"  Products with reviews: {len(products_with_reviews)}")
    print(f"  Filtered out: {len(products) - len(products_with_reviews)}")

    print(f"Initializing embeddings ({embedding_model})...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    print(f"Initializing ChromaDB at {chroma_path}...")
    client = chromadb.PersistentClient(path=str(chroma_path))

    # Drop existing collections so re-running this script gives a clean state.
    existing = [c.name for c in client.list_collections()]
    if "specs" in existing:
        client.delete_collection("specs")
    if "reviews" in existing:
        client.delete_collection("reviews")

    specs_collection = client.create_collection(
        name="specs",
        metadata={"description": "Product specifications and metadata"},
    )
    reviews_collection = client.create_collection(
        name="reviews",
        metadata={"description": "Individual product reviews"},
    )

    specs_docs, specs_ids, specs_metadatas = [], [], []
    reviews_docs, reviews_ids, reviews_metadatas = [], [], []

    for product in products_with_reviews:
        asin = product.get("id", "")

        spec_doc = create_spec_document(product)
        specs_docs.append(spec_doc)
        specs_ids.append(f"spec_{asin}")
        specs_metadatas.append({"asin": asin})

        for idx, review in enumerate(product.get("reviews", [])):
            review_text = review.get("text", "")
            rating = review.get("rating", 0)
            reviews_docs.append(review_text)
            reviews_ids.append(f"review_{asin}_{idx}")
            reviews_metadatas.append({
                "asin": asin,
                "rating": rating,
                "sentiment": get_sentiment(rating),
            })

    print("Embedding product specifications...")
    specs_embeddings = embeddings.embed_documents(specs_docs)
    specs_collection.add(
        documents=specs_docs,
        embeddings=specs_embeddings,
        ids=specs_ids,
        metadatas=specs_metadatas,
    )

    print("Embedding reviews (this may take a while)...")
    reviews_embeddings = embeddings.embed_documents(reviews_docs)
    reviews_collection.add(
        documents=reviews_docs,
        embeddings=reviews_embeddings,
        ids=reviews_ids,
        metadatas=reviews_metadatas,
    )

    print()
    print(f"Specs:   {len(specs_docs)} documents")
    print(f"Reviews: {len(reviews_docs)} documents")
    print(f"Persisted to {chroma_path}")


if __name__ == "__main__":
    main()
