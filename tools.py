"""
LangChain Tools for Multi-Agent BI Report System
Shared tools for both Flat and Hierarchical agent architectures.
"""

from typing import Optional

import chromadb
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

# =============================================================================
# SETUP: ChromaDB Connection & Embeddings
# =============================================================================

CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize persistent ChromaDB client
_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Initialize HuggingFace embeddings (same model used during ingestion)
_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Get collection references
_specs_collection = _client.get_collection("specs")
_reviews_collection = _client.get_collection("reviews")


# =============================================================================
# TOOL 1: Get Product List
# =============================================================================

@tool
def get_product_list() -> str:
    """
    Returns a list of all available laptop products in the database.
    Each entry includes the product name (title) and its ASIN identifier.
    Use this to discover what products are available for analysis.
    """
    # Get all documents from specs collection
    results = _specs_collection.get(include=["documents", "metadatas"])
    
    if not results["documents"]:
        return "No products found in the database."
    
    products = []
    for doc, metadata in zip(results["documents"], results["metadatas"]):
        asin = metadata.get("asin", "Unknown")
        # Extract title from the document (first line after "Title: ")
        lines = doc.split("\n")
        title = "Unknown"
        for line in lines:
            if line.startswith("Title: "):
                title = line.replace("Title: ", "").strip()
                # Truncate long titles for readability
                if len(title) > 80:
                    title = title[:77] + "..."
                break
        products.append(f"- ASIN: {asin} | {title}")
    
    return f"Available Products ({len(products)} total):\n" + "\n".join(products)


# =============================================================================
# TOOL 2: Get Product Specs
# =============================================================================

class GetProductSpecsInput(BaseModel):
    """Input schema for get_product_specs tool."""
    asin: str = Field(description="The ASIN (Amazon Standard Identification Number) of the product to retrieve specs for.")


@tool(args_schema=GetProductSpecsInput)
def get_product_specs(asin: str) -> str:
    """
    Retrieves the full product specifications for a given ASIN.
    Returns the product's title, price, and description.
    Use this when you need detailed information about a specific product.
    """
    # Query specs collection with ASIN filter
    results = _specs_collection.get(
        where={"asin": asin},
        include=["documents", "metadatas"]
    )
    
    if not results["documents"]:
        return f"No product found with ASIN: {asin}"
    
    # Return the full specification document
    return results["documents"][0]


# =============================================================================
# TOOL 3: Search Reviews
# =============================================================================

class SearchReviewsInput(BaseModel):
    """Input schema for search_reviews tool."""
    query: str = Field(description="The search query to find relevant reviews.")
    asin: Optional[str] = Field(default=None, description="Optional: Filter reviews by a specific product ASIN.")
    sentiment_type: Optional[str] = Field(default=None, description="Optional: Filter by sentiment - 'positive' or 'negative'.")


@tool(args_schema=SearchReviewsInput)
def search_reviews(
    query: str,
    asin: Optional[str] = None,
    sentiment_type: Optional[str] = None
) -> str:
    """
    Searches customer reviews using semantic similarity.
    Can optionally filter by product ASIN and/or sentiment type.
    Returns the top 5 most relevant review excerpts.
    Use this to understand customer opinions and experiences.
    """
    # Build metadata filter
    where_filter = None
    conditions = []
    
    if asin:
        conditions.append({"asin": asin})
    if sentiment_type and sentiment_type.lower() in ["positive", "negative"]:
        conditions.append({"sentiment": sentiment_type.lower()})
    
    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}
    
    # Generate query embedding
    query_embedding = _embeddings.embed_query(query)
    
    # Perform similarity search
    results = _reviews_collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    
    if not results["documents"] or not results["documents"][0]:
        filter_desc = []
        if asin:
            filter_desc.append(f"ASIN={asin}")
        if sentiment_type:
            filter_desc.append(f"sentiment={sentiment_type}")
        filter_str = " with filters: " + ", ".join(filter_desc) if filter_desc else ""
        return f"No reviews found matching query: '{query}'{filter_str}"
    
    # Format results
    output = []
    for i, (doc, metadata, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ), 1):
        # Convert distance to similarity score (ChromaDB uses L2 distance)
        # Normalize: similarity = 1 / (1 + distance)
        similarity = 1 / (1 + distance)
        
        asin_info = metadata.get("asin", "Unknown")
        rating = metadata.get("rating", "N/A")
        sentiment = metadata.get("sentiment", "Unknown")
        
        # Truncate long reviews for readability
        review_text = doc[:500] + "..." if len(doc) > 500 else doc
        
        output.append(
            f"[Review {i}] ASIN: {asin_info} | Rating: {rating}/5 | Sentiment: {sentiment}\n"
            f"Relevance: {similarity:.2%}\n"
            f"Text: {review_text}\n"
        )
    
    return "\n".join(output)


# =============================================================================
# TOOL 4: Verify Claim (CRITIC Tool for Hallucination Detection)
# =============================================================================

class VerifyClaimInput(BaseModel):
    """Input schema for verify_claim tool."""
    claim: str = Field(description="The claim or statement to verify against the database.")


@tool(args_schema=VerifyClaimInput)
def verify_claim(claim: str) -> str:
    """
    CRITIC TOOL: Verifies a claim by searching the database for supporting evidence.
    Returns the most similar document and a similarity score (0-1).
    A score close to 1.0 indicates strong evidence; scores below 0.5 suggest potential hallucination.
    Use this to fact-check generated claims before including them in reports.
    """
    # Generate claim embedding
    claim_embedding = _embeddings.embed_query(claim)
    
    # Search both collections for the claim
    specs_results = _specs_collection.query(
        query_embeddings=[claim_embedding],
        n_results=1,
        include=["documents", "distances"]
    )
    
    reviews_results = _reviews_collection.query(
        query_embeddings=[claim_embedding],
        n_results=1,
        include=["documents", "distances"]
    )
    
    # Calculate similarity scores (convert L2 distance to similarity)
    specs_distance = specs_results["distances"][0][0] if specs_results["distances"][0] else float('inf')
    reviews_distance = reviews_results["distances"][0][0] if reviews_results["distances"][0] else float('inf')
    
    specs_similarity = 1 / (1 + specs_distance)
    reviews_similarity = 1 / (1 + reviews_distance)
    
    # Find the best match
    if specs_similarity >= reviews_similarity:
        best_source = "Product Specifications"
        best_doc = specs_results["documents"][0][0] if specs_results["documents"][0] else "N/A"
        best_similarity = specs_similarity
    else:
        best_source = "Customer Reviews"
        best_doc = reviews_results["documents"][0][0] if reviews_results["documents"][0] else "N/A"
        best_similarity = reviews_similarity
    
    # Truncate document for readability
    best_doc_truncated = best_doc[:600] + "..." if len(best_doc) > 600 else best_doc
    
    # Determine verification status
    if best_similarity >= 0.7:
        status = "✅ VERIFIED - Strong evidence found"
    elif best_similarity >= 0.5:
        status = "⚠️ PARTIAL - Some supporting evidence"
    else:
        status = "❌ UNVERIFIED - Potential hallucination detected"
    
    return (
        f"Claim Verification Result:\n"
        f"{'=' * 40}\n"
        f"Claim: \"{claim}\"\n"
        f"Status: {status}\n"
        f"Similarity Score: {best_similarity:.4f}\n"
        f"Source: {best_source}\n"
        f"{'=' * 40}\n"
        f"Supporting Evidence:\n{best_doc_truncated}"
    )


# =============================================================================
# Export all tools for use in agents
# =============================================================================

ALL_TOOLS = [
    get_product_list,
    get_product_specs,
    search_reviews,
    verify_claim
]

if __name__ == "__main__":
    # Quick test of tools
    print("Testing tools...\n")
    
    print("1. get_product_list():")
    print(get_product_list.invoke({}))
    print("\n" + "=" * 60 + "\n")
    
    print("2. get_product_specs(asin='B0FL85ZPTW'):")
    print(get_product_specs.invoke({"asin": "B0FL85ZPTW"}))
    print("\n" + "=" * 60 + "\n")
    
    print("3. search_reviews(query='battery life', sentiment_type='negative'):")
    print(search_reviews.invoke({"query": "battery life", "sentiment_type": "negative"}))
    print("\n" + "=" * 60 + "\n")
    
    print("4. verify_claim(claim='The Lenovo LOQ 15 has an RTX 4060 graphics card'):")
    print(verify_claim.invoke({"claim": "The Lenovo LOQ 15 has an RTX 4060 graphics card"}))
