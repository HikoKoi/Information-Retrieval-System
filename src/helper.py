import os
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv() 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", 
                             temperature=0.1,
                             google_api_key=GEMINI_API_KEY)

def load_pdf(pdf_files):
    documents = []
    for pdf in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.read())
            tmp_file.flush()

        loader = PyPDFLoader(tmp_file.name)
        documents.extend(loader.load())
        
        os.remove(tmp_file.name)

    return documents


def split_chunks(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks):
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    return vector_store


def format_docs(docs):
    """
    Hàm này nhận một danh sách các đối tượng Document
    và biến chúng thành MỘT chuỗi (string) duy nhất.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def get_conversational_chain():
    """
    Tạo ra một RAG chain có khả năng trò chuyện (conversational).
    Chain này được thiết kế để nhận một dictionary bao gồm:
    - question: Câu hỏi của người dùng (str)
    - chat_history: Lịch sử trò chuyện (list[BaseMessage])
    - retriever: Một đối tượng retriever (từ vector store)
    """

    # 1. Prompt để "cô đọng" câu hỏi (Condense Question Prompt)
    # Mục đích: Biến câu hỏi follow-up và lịch sử chat -> một câu hỏi độc lập
    condense_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and a follow-up question, rephrase the follow-up question to be a standalone question, in its original language."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
        ("ai", "Standalone question:"),
    ])
    
    # 2. Chain con 1: Dùng để tạo câu hỏi độc lập
    condense_q_chain = condense_q_prompt | llm | StrOutputParser()

    # 3. Prompt để trả lời (SỬ DỤNG PROMPT BẠN CUNG CẤP)
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là một trợ lý AI hữu ích chuyên trả lời các câu hỏi dựa trên tài liệu được cung cấp.
Hãy trả lời câu hỏi của người dùng một cách chi tiết và rõ ràng.
Chỉ sử dụng lịch sử chat (Chat History) và ngữ cảnh (Context) sau đây để có câu trả lời đầy đủ nhất.
Hãy trả lời dựa trên ngữ cảnh và lịch sử chat.
Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy nói "Tôi không tìm thấy thông tin trong tài liệu."

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # 4. Hàm trợ giúp để lấy tài liệu (context)
    # Hàm này sẽ chạy chain "cô đọng câu hỏi", sau đó dùng retriever để tìm tài liệu,
    # và cuối cùng dùng hàm format_docs của bạn để định dạng chúng.
    def retrieve_docs(input_dict):
        # Lấy câu hỏi độc lập
        condensed_question = condense_q_chain.invoke({
            "chat_history": input_dict["chat_history"],
            "question": input_dict["question"]
        })
        # Lấy retriever từ input
        retriever = input_dict["retriever"]
        # Tìm tài liệu
        docs = retriever.invoke(condensed_question)
        # Format tài liệu
        return format_docs(docs)

    # 5. Xây dựng Chain chính
    conversational_chain = (
        RunnablePassthrough.assign(
            # Chạy hàm retrieve_docs và gán kết quả vào key "context"
            context=retrieve_docs 
        )
        | answer_prompt     # Chuyển (question, chat_history, retriever, context) vào prompt
        | llm
        | StrOutputParser()
    )

    return conversational_chain