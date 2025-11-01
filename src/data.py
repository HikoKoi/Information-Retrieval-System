import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv() 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)


def load_pdf(folder_path="data"):
    all_chunks = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            chunks = split_chunks(docs)
            all_chunks.extend(chunks)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            chunks = split_chunks(docs)
            all_chunks.extend(chunks)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            chunks = split_chunks(docs)
            all_chunks.extend(chunks)

    return all_chunks


def split_chunks(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def vector_store(all_chunks, embeddings_choice):
    
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model=embeddings_choice,
        google_api_key=GOOGLE_API_KEY,
        batch_size=5
    )
    vector_store = FAISS.from_documents(all_chunks, embeddings)

    return vector_store

