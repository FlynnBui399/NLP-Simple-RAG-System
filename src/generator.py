"""
Generator module — Task 3
Receives a query + context chunks and generates a Vietnamese answer.
Tries OpenAI (gpt-3.5-turbo) first; falls back to local Qwen2.5-3B-Instruct.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE: str = """\
Bạn là trợ lý tư vấn của Khoa Công nghệ Thông tin - HCMUTE.
Dựa vào các đoạn thông tin sau đây, hãy trả lời câu hỏi một cách chính xác
và ngắn gọn bằng tiếng Việt.
Nếu không tìm thấy thông tin phù hợp, hãy trả lời:
"Tôi chưa có thông tin về vấn đề này."

Thông tin tham khảo:
{context}

Câu hỏi: {query}
Trả lời:"""


def build_prompt(query: str, context_chunks: list[dict]) -> str:
    """Xây dựng prompt tiếng Việt từ câu hỏi và context.

    Args:
        query: Câu hỏi tiếng Việt.
        context_chunks: Danh sách chunk từ retriever.

    Returns:
        Chuỗi prompt hoàn chỉnh.
    """
    context_parts: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        text = chunk.get("text", "")
        source = chunk.get("source_url", "")
        context_parts.append(f"[{i}] {text}\n    Nguồn: {source}")

    context = "\n\n".join(context_parts)
    return PROMPT_TEMPLATE.format(context=context, query=query)


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------
def _generate_deepseek(prompt: str) -> str:
    """Generate answer using Deepseek gpt-3.5-turbo.

    Args:
        prompt: Full prompt string.

    Returns:
        Generated answer text.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Bạn là trợ lý tư vấn Khoa CNTT - HCMUTE. Trả lời bằng tiếng Việt."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Local HuggingFace backend
# ---------------------------------------------------------------------------
def _generate_local(prompt: str) -> str:
    """Generate answer using local Qwen2.5-3B-Instruct model.

    Falls back to float16 if bitsandbytes is unavailable.

    Args:
        prompt: Full prompt string.

    Returns:
        Generated answer text.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    try:
        import sentencepiece  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Local Qwen needs the `sentencepiece` package (and usually `tiktoken`) "
            "for the tokenizer. From the assignment2 folder run: "
            "`pip install -r requirements.txt` or `uv sync`."
        ) from e

    model_name = "./models/Qwen2.5-3B-Instruct"

    print(f"[Generator] Loading local model: {model_name}")

    # Try 4-bit quantization first
    quantization_config = None
    try:
        import bitsandbytes  # noqa: F401~
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        print("[Generator] Using 4-bit quantization (bitsandbytes)")
    except ImportError:
        print("[Generator] bitsandbytes not available — using float16")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
    )

    load_kwargs: dict = {
        "pretrained_model_name_or_path": model_name,
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
        )

    # Decode only the NEW tokens (skip the prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate(query: str, context_chunks: list[dict]) -> str:
    """Sinh câu trả lời tiếng Việt.

    Tries OpenAI first (if OPENAI_API_KEY is set), otherwise falls back to a
    local Qwen2.5-3B-Instruct model.

    Args:
        query: Câu hỏi tiếng Việt.
        context_chunks: Danh sách chunks từ retriever.

    Returns:
        Câu trả lời dạng string.
    """
    prompt = build_prompt(query, context_chunks)

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if api_key and api_key != "your_key_here":
        try:
            print("[Generator] Using DeepSeek (OpenAI API)")
            return _generate_deepseek(prompt)
        except Exception as e:
            error_msg = f"[Generator] ERROR: DeepSeek API failed — {e}"
            print(error_msg)
            return "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."
    else:
        return "Xin lỗi, hệ thống chưa được cấu hình API key hoặc API key không hợp lệ."


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mock_chunks = [
        {
            "text": "Khoa Công nghệ Thông tin - FIT HCMUTE được thành lập năm 2001.",
            "source_url": "https://fit.hcmute.edu.vn/gioi-thieu",
        },
        {
            "text": "Khoa có 4 ngành đào tạo: CNTT, KTPM, HTTT, và MMT&TT.",
            "source_url": "https://fit.hcmute.edu.vn/dao-tao",
        },
    ]
    answer = generate("Khoa CNTT có mấy ngành đào tạo?", mock_chunks)
    print(f"\nAnswer: {answer}")
