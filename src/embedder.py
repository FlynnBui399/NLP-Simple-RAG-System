"""
Embedder module — Task 1
Reads chunks.json, embeds with BAAI/bge-m3, and stores in ChromaDB.
"""

from sentence_transformers import SentenceTransformer
import chromadb
import json
import time
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNKS_PATH: str = os.path.join(os.path.dirname(__file__), "..", "data", "chunks.json")
VECTOR_STORE_DIR: str = os.path.join(os.path.dirname(__file__), "..", "vector_store")
COLLECTION_NAME: str = "fit_hcmute"
EMBEDDING_MODEL: str = "BAAI/bge-m3"


def load_chunks(path: str = CHUNKS_PATH) -> list[dict]:
    """Load chunks từ file JSON.

    Args:
        path: Đường dẫn tới file chunks.json.

    Returns:
        Danh sách các dict chứa ``text`` và ``source_url``.
    """
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        print("[Embedder] ERROR: data/chunks.json not found.")
        print("[Embedder] Please provide chunks.json from Member 1 before running.")
        sys.exit(1)

    with open(abs_path, "r", encoding="utf-8") as f:
        chunks: list[dict] = json.load(f)

    print(f"[Embedder] Loading chunks from {path}...")
    print(f"[Embedder] Found {len(chunks)} chunks. Starting embedding...")
    return chunks


def embed_and_store(chunks: list[dict]) -> None:
    """Embed toàn bộ chunks và lưu vào ChromaDB.

    Args:
        chunks: Danh sách các dict với keys ``text`` và ``source_url``.
    """
    persist_dir = os.path.abspath(VECTOR_STORE_DIR)

    # Check if vector store already exists with data
    if os.path.exists(persist_dir):
        try:
            client = chromadb.PersistentClient(path=persist_dir)
            col = client.get_collection(COLLECTION_NAME)
            if col.count() > 0:
                print(f"[Embedder] Vector store already exists with {col.count()} items. Skipping.")
                return
        except Exception:
            pass  # collection doesn't exist yet — proceed

    # Load embedding model
    print(f"[Embedder] Model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    start = time.time()

    # Compute embeddings
    texts: list[str] = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # Store in ChromaDB
    client = chromadb.PersistentClient(path=persist_dir)

    # Delete existing collection if empty / partially created
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB expects list of lists for embeddings
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=[emb.tolist() for emb in embeddings],
        documents=texts,
        metadatas=[{"source_url": c.get("source_url", "")} for c in chunks],
    )

    elapsed = time.time() - start
    print(f"[Embedder] Done! {len(chunks)} chunks embedded in {elapsed:.1f}s")
    print(f"[Embedder] Vector store saved to {VECTOR_STORE_DIR}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    embed_and_store(load_chunks())
