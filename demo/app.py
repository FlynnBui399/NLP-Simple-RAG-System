"""
Demo UI — Task 5
Gradio web interface for the FIT HCMUTE RAG chatbot.
"""

import sys
import os

# Ensure the project root is on the Python path so `src.*` imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr
from src.rag_pipeline import ask


def handle_query(query: str) -> tuple[str, list[list]]:
    """Process a user query and return answer + source table.

    Args:
        query: Câu hỏi tiếng Việt từ người dùng.

    Returns:
        Tuple of (answer_text, source_table_rows).
    """
    if not query or not query.strip():
        return "Vui lòng nhập câu hỏi.", []

    result = ask(query.strip())

    answer: str = result["answer"]

    # Build source table rows: [STT, Đoạn văn bản, Nguồn URL]
    table_rows: list[list] = []
    for i, doc in enumerate(result["source_documents"], 1):
        text_preview = doc["text"][:200] + ("..." if len(doc["text"]) > 200 else "")
        table_rows.append([i, text_preview, doc["source_url"]])

    return answer, table_rows


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="🎓 Chatbot Tư Vấn - Khoa CNTT FIT HCMUTE",
) as demo:
    gr.Markdown("# 🎓 Chatbot Tư Vấn - Khoa CNTT FIT HCMUTE")
    gr.Markdown("Hệ thống hỏi đáp tự động dựa trên dữ liệu Khoa Công nghệ Thông tin — ĐH Sư phạm Kỹ thuật TP.HCM")

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Câu hỏi",
                placeholder="Nhập câu hỏi của bạn về Khoa CNTT...",
                lines=2,
            )
            submit_btn = gr.Button("Gửi câu hỏi", variant="primary")

        with gr.Column(scale=5):
            answer_output = gr.Textbox(
                label="Câu trả lời",
                lines=8,
                interactive=False,
            )

    source_table = gr.Dataframe(
        headers=["STT", "Đoạn văn bản", "Nguồn URL"],
        label="Nguồn tài liệu tham khảo",
        interactive=False,
        wrap=True,
    )

    # Wire up events
    submit_btn.click(
        fn=handle_query,
        inputs=[query_input],
        outputs=[answer_output, source_table],
    )
    query_input.submit(
        fn=handle_query,
        inputs=[query_input],
        outputs=[answer_output, source_table],
    )

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860, share=True,theme=gr.themes.Soft())
