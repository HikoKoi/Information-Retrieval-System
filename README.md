### Information-Retrieval-System | Trợ lý AI kết hợp Google Gemini + LangChain ###

Một hệ thống chatbot thông minh tích hợp mô hình ngôn ngữ lớn Google Gemini với cơ chế Retrieval-Augmented Generation (RAG) để truy xuất và trả lời từ các tài liệu do người dùng cung cấp.

## 🎯 Mục tiêu

Giúp người dùng đặt câu hỏi và nhận câu trả lời chính xác từ các tài liệu PDF, DOCX, hoặc TXT của riêng họ. Hệ thống tận dụng sức mạnh của LLM và retriever (FAISS + BM25) để đưa ra phản hồi chính xác, có dẫn chứng.

## 🧠 Kiến trúc hệ thống

```
Người dùng ↔️ Streamlit UI ↔️ LangChain Agent ↔️ Google Gemini LLM
                                ↘️ VectorStore (FAISS | BM25 từ tài liệu)
```

- LLM: Google Gemini (gemini-2.5-pro hoặc gemini-2.5-flash)

- Retriever: FAISS (embedding) | BM25 (từ khóa)

- VectorStore: FAISS

- UI: Streamlit (hỗ trợ trò chuyện thời gian thực)

- RAG pipeline: Truy xuất tài liệu → Trả lời có dẫn chứng

## 🚀 Tính năng chính

- **Giao diện Web thân thiện**: Xây dựng bằng Streamlit, cho phép người dùng tương tác dễ dàng.
- **Hỗ trợ đa dạng định dạng tài liệu**: Có thể tải lên và xử lý các tệp `.txt`, `.docx`, và `.pdf`.
- **Cơ chế Retrieval tiên tiến**: Kết hợp giữa tìm kiếm dựa trên vector (FAISS) và tìm kiếm dựa trên từ khóa (BM25) để tăng độ chính xác của thông tin được truy xuất.
- **Tích hợp Google Gemini**: Sử dụng các mô hình mạnh mẽ như `gemini-2.5-flash` và `gemini-2.5-pro` để tạo ra câu trả lời chất lượng cao.
- **Lưu trữ và quản lý lịch sử hội thoại**: Giúp chatbot duy trì ngữ cảnh trong suốt cuộc trò chuyện.
- **Cấu hình linh hoạt**: Cho phép người dùng tùy chọn mô hình embedding và mô hình LLM ngay trên giao diện.

## ⚙️ Cài đặt và Chạy dự án

### STEP-00:

Clone the repository

``` bash
git clone https://github.com/HikoKoi/Information-Retrieval-System.git
```
## STEP-01: Tạo môi trường ảo

``` bash
python -m venv venv
```

``` bash
source venv/Scripts/activate
```
## STEP-02: Tải các thư viện cần thiết requirements.txt

``` bash
pip install -r requirements.txt
```
## STEP-03: Tạo file `.env` và nhập GOOGLE_API_KEY:

``` ini
GOOGLE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
## STEP-04: Chạy chương trình

``` bash
streamlit run app.py
```
Link:

``` bash
http://localhost:8501
```

## Công nghệ sử dụng:

- Python
- LangChain
- Streamlit
- Gemini
- FAISS