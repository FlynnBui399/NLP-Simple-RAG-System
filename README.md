# RAG Pipeline — Chatbot Khoa CNTT FIT HCMUTE

## Yêu cầu hệ thống
- Python 3.10+
- RAM: tối thiểu 8GB (16GB nếu chạy LLM local)
- GPU: khuyến nghị (không bắt buộc)

## Cài đặt
```bash
pip install -r requirements.txt
cp .env .env.local  # điền OPENAI_API_KEY nếu có
```

## Hướng dẫn chạy (sau khi có chunks.json)
```bash
# Bước 1: Index dữ liệu
python src/embedder.py

# Bước 2: Test CLI
python src/rag_pipeline.py

# Bước 3: Chạy Demo UI
python demo/app.py
# Mở trình duyệt: http://localhost:7860
```

## Câu hỏi mẫu để test
- Khoa CNTT được thành lập năm nào?
- Khoa có những ngành đào tạo nào?
- Điểm chuẩn vào Khoa CNTT năm nay là bao nhiêu?
- Danh sách giảng viên của Khoa CNTT?
- Các sự kiện sắp tới của khoa là gì?
