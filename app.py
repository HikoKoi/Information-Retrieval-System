import streamlit as st 
import src.helper as helper
from langchain_core.messages import HumanMessage, AIMessage

def main():
    st.set_page_config(
        page_title="Information Retrieval System",
    )
    st.header("Information Retrieval System with Gemini")
    
    # --- 1. Khởi tạo State (Trạng thái) cho Session ---
    
    # Khởi tạo chain xử lý hội thoại và lưu vào session
    # Việc này chỉ chạy 1 lần mỗi session
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = helper.get_conversational_chain()

    # Khởi tạo lịch sử chat (nếu chưa có)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [] # Bắt đầu với list rỗng

    # --- 2. Sidebar (Code của bạn, đã sửa lỗi nhỏ) ---
    with st.sidebar:
        st.title("Menu:")
        pdf_files = st.file_uploader("Upload your PDF", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Submit"):
            with st.spinner("Processing..."):
                if pdf_files:
                    documents = helper.load_pdf(pdf_files)
                    chunks = helper.split_chunks(documents)
                    vector_store = helper.create_vector_store(chunks)
                    
                    # Lưu vector store vào session
                    st.session_state.vector_store = vector_store
                    
                    # QUAN TRỌNG: Reset lịch sử chat khi upload file mới
                    st.session_state.chat_history = [] 
                    
                    st.success(f"Uploaded {len(pdf_files)} PDF file(s) successfully!")
                else:
                    st.error("Please upload at least one PDF file.")

    # --- 3. Giao diện Chat chính ---
    
    # Hiển thị lịch sử chat cũ (từ session_state)
    for message in st.session_state.chat_history:
        with st.chat_message(message.type):
            st.markdown(message.content)

    # Lấy câu hỏi mới từ người dùng (dùng st.chat_input)
    user_question = st.chat_input("Ask a question about your documents:")
    
    if user_question:
        # 5. Kiểm tra xem đã upload file và tạo vector_store chưa
        if "vector_store" not in st.session_state:
            st.error("Bạn phải upload tài liệu PDF và nhấn 'Submit' trước khi đặt câu hỏi.")
        else:
            # --- Đây là logic chính khi người dùng hỏi ---
            
            # Hiển thị câu hỏi của user
            with st.chat_message("human"):
                st.markdown(user_question)
            
            # Thêm câu hỏi của user vào lịch sử chat
            st.session_state.chat_history.append(HumanMessage(content=user_question))

            # Lấy retriever từ vector_store trong session
            retriever = st.session_state.vector_store.as_retriever()
            
            # Chuẩn bị input cho chain
            inputs = {
                "question": user_question,
                "chat_history": st.session_state.chat_history,
                "retriever": retriever 
            }

            with st.spinner("Đang tìm kiếm và trả lời..."):
                # Lấy chain từ session và thực thi
                response = st.session_state.conversation_chain.invoke(inputs)
            
            # Hiển thị câu trả lời của AI
            with st.chat_message("ai"):
                st.markdown(response)
            
            # Thêm câu trả lời của AI vào lịch sử chat
            st.session_state.chat_history.append(AIMessage(content=response))

if __name__ == "__main__":
    main()