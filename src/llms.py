import os
from dotenv import load_dotenv
# from langchain_community.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 

load_dotenv() 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")



def get_retriever(vector_store):
    """
    Kết hợp FAISS retriever (dựa trên embedding) và BM25 retriever (dựa trên từ khóa)
    theo tỷ trọng 7:3.
    """
    ## FAISS-based retriever (embedding)
    retriever_faiss = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    ## BM25-based retriever (keyword)

        #Lấy 100 documents gần nhất từ FAISS
    top_docs = vector_store.similarity_search("", k=100)
        # Tạo BM25 retriever từ 100 tài liệu đã lấy
    retriever_bm25 = BM25Retriever.from_documents(top_docs)
    retriever_bm25.k = 5

    # ## Kết hợp hai retriever
    # retriever = EnsembleRetriever(
    #     retrievers=[retriever_faiss, retriever_bm25],
    #     weights=[0.7, 0.3]
    # )

    return retriever_faiss



def get_llm_and_agent(model_choice, retriever):
    """
    Khởi tạo LLM của Google Gemini
    """
    # Khởi tạo ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model= model_choice,
        temperature=0.01,
        google_api_key=GOOGLE_API_KEY,
        model_kwargs={"streaming": True},
    )
    retriever_tool = tool(
        name_or_callable=retriever.get_relevant_documents,
        description="Search for information of a question in the knowledge base",
        return_direct=True  # trả trực tiếp kết quả thay vì dict
    )

    # Prompt template cho agent
    system = """Bạn là một trợ lý AI hữu ích chuyên trả lời các câu hỏi dựa trên tài liệu được cung cấp.
                Tên bạn là Koi.
                Hãy trả lời câu hỏi của người dùng một cách chi tiết và rõ ràng.
                Hãy trả lời dựa trên thông tin được cung cấp. 
                Nếu không đủ thông tin, hãy nói: "Tôi không chắc chắn về câu trả lời do thiếu thông tin mà tài liệu cung cấp"."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("system", "You have access to the following tools: {tool_names}. Use them if needed."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


    # Tạo agent
    agent = create_agent(
        llm=llm,
        tools=[retriever_tool],
        system_prompt=prompt,
    )

    return agent

