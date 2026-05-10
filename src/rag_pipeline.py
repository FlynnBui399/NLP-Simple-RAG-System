"""
RAG Pipeline module — Task 4
Combines retriever + generator into a full end-to-end pipeline.
"""

from src.retriever import retrieve
from src.generator import generate


def ask(query: str) -> dict:
    """Nhận câu hỏi, trả về câu trả lời kèm nguồn tài liệu.

    Args:
        query: Câu hỏi tiếng Việt.

    Returns:
        Dict with keys:
            - ``question`` (str): câu hỏi gốc.
            - ``answer`` (str): câu trả lời sinh ra.
            - ``source_documents`` (list[dict]): danh sách ``{"text": str, "source_url": str}``.
    """
    # Step 1 — Retrieve relevant chunks
    context_chunks: list[dict] = retrieve(query, top_k=5)

    # Step 2 — Generate answer
    answer: str = generate(query, context_chunks)

    # Step 3 — Build source documents list
    source_documents: list[dict] = [
        {"text": chunk["text"], "source_url": chunk["source_url"]}
        for chunk in context_chunks
    ]

    return {
        "question": query,
        "answer": answer,
        "source_documents": source_documents,
    }


# ---------------------------------------------------------------------------
# CLI Mode
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== RAG Pipeline FIT HCMUTE ===")
    print("Nhập 'exit' để thoát\n")
    while True:
        query = input("Câu hỏi: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue
        result = ask(query)
        print(f"\nTrả lời: {result['answer']}")
        print(f"\nNguồn tài liệu:")
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"  [{i}] {doc['source_url']}")
        print("-" * 50)
