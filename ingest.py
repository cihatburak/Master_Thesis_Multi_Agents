"""
Data Ingestion Script for Multi-Agent BI Report System
Loads product data and reviews into ChromaDB vector store.
"""

import json
from pathlib import Path

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings


def load_dataset(filepath: str) -> list[dict]:
    """Load the JSON dataset from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_products_with_reviews(products: list[dict]) -> list[dict]:
    """Filter out products with 0 reviews."""
    return [p for p in products if p.get("stats", {}).get("review_count", 0) > 0]


def get_sentiment(rating: int) -> str:
    """Determine sentiment based on rating: >=4 positive, <=3 negative."""
    return "positive" if rating >= 4 else "negative"


def create_spec_document(product: dict) -> str:
    """Create a document string from product metadata."""
    metadata = product.get("metadata", {})
    title = metadata.get("title", "")
    price = metadata.get("price", "")
    description = metadata.get("description", "")
    
    return f"Title: {title}\nPrice: ${price}\nDescription: {description}"


def main():
    # Configuration
    dataset_path = Path(__file__).parent / "dataset_final.json"
    chroma_path = Path(__file__).parent / "chroma_db"
    embedding_model = "all-MiniLM-L6-v2"
    
    print("=" * 60)
    print("Multi-Agent BI Report System - Data Ingestion")
    print("=" * 60)
    
    # Load and filter dataset
    print("\n[1/5] Loading dataset...")
    products = load_dataset(str(dataset_path))
    print(f"      Total products loaded: {len(products)}")
    
    print("\n[2/5] Filtering products with reviews...")
    products_with_reviews = filter_products_with_reviews(products)
    print(f"      Products with reviews: {len(products_with_reviews)}")
    print(f"      Products filtered out: {len(products) - len(products_with_reviews)}")
    
    # Initialize embeddings
    print("\n[3/5] Initializing HuggingFace embeddings...")
    print(f"      Model: {embedding_model}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # Initialize ChromaDB client
    print("\n[4/5] Initializing ChromaDB...")
    print(f"      Path: {chroma_path}")
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    # Get or create collections
    # Delete existing collections if they exist (for clean re-ingestion)
    existing_collections = [c.name for c in client.list_collections()]
    if "specs" in existing_collections:
        client.delete_collection("specs")
    if "reviews" in existing_collections:
        client.delete_collection("reviews")
    
    specs_collection = client.create_collection(
        name="specs",
        metadata={"description": "Product specifications and metadata"}
    )
    reviews_collection = client.create_collection(
        name="reviews",
        metadata={"description": "Individual product reviews"}
    )
    
    # Ingest data
    print("\n[5/5] Ingesting data into ChromaDB...")
    
    specs_docs = []
    specs_ids = []
    specs_metadatas = []
    specs_embeddings = []
    
    reviews_docs = []
    reviews_ids = []
    reviews_metadatas = []
    reviews_embeddings = []
    
    for product in products_with_reviews:
        asin = product.get("id", "")
        
        # Prepare spec document
        spec_doc = create_spec_document(product)
        specs_docs.append(spec_doc)
        specs_ids.append(f"spec_{asin}")
        specs_metadatas.append({"asin": asin})
        
        # Prepare review documents
        reviews = product.get("reviews", [])
        for idx, review in enumerate(reviews):
            review_text = review.get("text", "")
            rating = review.get("rating", 0)
            sentiment = get_sentiment(rating)
            
            reviews_docs.append(review_text)
            reviews_ids.append(f"review_{asin}_{idx}")
            reviews_metadatas.append({
                "asin": asin,
                "rating": rating,
                "sentiment": sentiment
            })
    
    # Batch embed and add specs
    print("      Embedding product specifications...")
    specs_embeddings = embeddings.embed_documents(specs_docs)
    specs_collection.add(
        documents=specs_docs,
        embeddings=specs_embeddings,
        ids=specs_ids,
        metadatas=specs_metadatas
    )
    
    # Batch embed and add reviews
    print("      Embedding reviews (this may take a moment)...")
    reviews_embeddings = embeddings.embed_documents(reviews_docs)
    reviews_collection.add(
        documents=reviews_docs,
        embeddings=reviews_embeddings,
        ids=reviews_ids,
        metadatas=reviews_metadatas
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"\n📦 Specs Collection:   {len(specs_docs)} documents")
    print(f"📝 Reviews Collection: {len(reviews_docs)} documents")
    print(f"\n💾 Data persisted to: {chroma_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
