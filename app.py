import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from src.data import load_pdf, vector_store
from src.llms import get_retriever, get_llm_and_agent


def setup_page():
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    st.set_page_config(
        page_title="Information Retrieval System",
        page_icon="💬",
        layout="wide"
    )

def setup_sidebar():
    with st.sidebar:
        st.title("⚙️Cấu hình")
        
        st.header("📑Embeddings Model")
        embeddings_choice = st.radio(
            "Chọn Embeddings Model:",
            ["models/embedding-001"]
        )

        st.header("🐧Model LLM")
        model_choice = st.radio(
            "Chọn AI Model để trả lời:",
            ["gemini-2.5-flash", "gemini-2.5-pro"]
        )

        st.header("📁Tải dữ liệu")
        uploaded_files = st.file_uploader(
            "Chọn một hoặc nhiều file dữ liệu .txt, docx, pdf",
            type=["txt", "docx", "pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            os.makedirs("data", exist_ok=True)
            for uploaded_file in uploaded_files:
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                st.success(f"✅Đã lưu file vào {file_path}")

        return embeddings_choice, model_choice


def setup_chat_interface(model_choice):
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("Information Retrieval System")
    with col2:
        if st.button("🔄 Làm mới"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"}
            ]
            st.rerun()

    st.caption(f"🚀 Trợ lý AI được hỗ trợ bởi {model_choice}")

    msgs = StreamlitChatMessageHistory(key="langchain_messages")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"}
        ]
        msgs.add_ai_message("Xin chào! Tôi có thể giúp gì cho bạn hôm nay?")

    # 👉 Hiển thị toàn bộ lịch sử hội thoại
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

    return msgs

def handle_user_input(msgs, agent):
    """
    Xử lý khi người dùng gửi tin nhắn:
    1. Hiển thị tin nhắn người dùng
    2. Gọi AI xử lý và trả lời
    3. Lưu vào lịch sử chat
    """
    if prompt := st.chat_input("Hãy nhập câu hỏi của bạn:"):
        # Lưu và hiển thị tin nhắn người dùng
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

        # Xử lý và hiển thị câu trả lời
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            
            # Lấy lịch sử chat
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]
            ]

            # Gọi AI xử lý
            response = agent.invoke(
                {
                    "input": prompt,
                    "chat_history": chat_history
                },
                callbacks= [st_callback]
            )


            # Lưu và hiển thị câu trả lời
            output = response["output"]
            st.session_state.messages.append({"role": "assistant", "content": output})
            msgs.add_ai_message(output)
            st.write(output)

def main():
    setup_page()
    embeddings_choice, model_choice = setup_sidebar()

    # Giao diện chat
    msgs = setup_chat_interface(model_choice)
    
    os.makedirs("data", exist_ok=True)

    # Vector hóa dữ liệu
    all_chunks = load_pdf()

    if not all_chunks:
        st.warning("📂Chưa có dữ liệu để tạo embeddings. Vui lòng upload file .txt vào sidebar.")
        return  # Dừng lại, không chạy tiếp main
    else:
        vector = vector_store(all_chunks, embeddings_choice)
        retriever = get_retriever(vector)

    # Chat
    agent = get_llm_and_agent(model_choice, retriever)
    handle_user_input(msgs, agent)

if __name__ == "__main__":
    main()
