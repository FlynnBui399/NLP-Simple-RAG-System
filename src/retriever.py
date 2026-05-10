"""
Retriever module — Task 2
Receives a Vietnamese query and returns the top-k most relevant chunks.
"""

from sentence_transformers import SentenceTransformer
import chromadb
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VECTOR_STORE_DIR: str = os.path.join(os.path.dirname(__file__), "..", "vector_store")
COLLECTION_NAME: str = "fit_hcmute"
EMBEDDING_MODEL: str = "BAAI/bge-m3"

# Lazy-loaded globals so the model is only initialised once per process
_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


def _get_model() -> SentenceTransformer:
    """Return (and cache) the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def load_vectorstore() -> chromadb.Collection:
    """Load ChromaDB collection từ disk.

    Returns:
        chromadb.Collection đã persist.

    Raises:
        FileNotFoundError: Nếu thư mục ``vector_store/`` chưa tồn tại.
    """
    global _collection
    if _collection is not None:
        return _collection

    persist_dir = os.path.abspath(VECTOR_STORE_DIR)
    if not os.path.exists(persist_dir):
        print("[Retriever] ERROR: Run python src/embedder.py first")
        raise FileNotFoundError(
            f"Vector store directory '{persist_dir}' not found. "
            "Run `python src/embedder.py` first."
        )

    client = chromadb.PersistentClient(path=persist_dir)
    _collection = client.get_collection(COLLECTION_NAME)
    print(f"[Retriever] Loaded collection '{COLLECTION_NAME}' with {_collection.count()} items.")
    return _collection


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Tìm top_k chunks liên quan nhất với câu hỏi.

    Args:
        query: Câu hỏi tiếng Việt.
        top_k: Số lượng chunk trả về (mặc định 5).

    Returns:
        List of dicts: ``[{"text": str, "source_url": str, "score": float}, ...]``
    """
    try:
        collection = load_vectorstore()
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[Retriever] ERROR: {e}")
        return []

    # Embed the query with the same model
    model = _get_model()
    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Build response
    documents: list[str] = results["documents"][0]
    metadatas: list[dict] = results["metadatas"][0]
    distances: list[float] = results["distances"][0]

    output: list[dict] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        # ChromaDB cosine distance is in [0, 2]; convert to similarity score
        score = 1.0 - dist
        output.append(
            {
                "text": doc,
                "source_url": meta.get("source_url", ""),
                "score": round(score, 4),
            }
        )

    return output


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    q = "Khoa CNTT có những ngành đào tạo nào?"
    print(f"[Retriever] Testing with query: {q}\n")
    for i, r in enumerate(retrieve(q), 1):
        print(f"  [{i}] score={r['score']:.4f}  url={r['source_url']}")
        print(f"       {r['text'][:120]}...\n")
