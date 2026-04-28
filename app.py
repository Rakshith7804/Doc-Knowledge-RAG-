import streamlit as st
import os
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Load env
load_dotenv()

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📄 RAG-Based AI Chatbot")
st.write("Upload PDFs and chat with them!")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Sidebar for file upload
st.sidebar.header("Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# Process PDFs
if st.sidebar.button("Process Documents"):
    if uploaded_files:
        all_docs = []

        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(uploaded_file.name)
            docs = loader.load()
            all_docs.extend(docs)

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(all_docs)

        # Embeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Store in Chroma
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )

        st.session_state.vector_store = vector_store

        st.success("Documents processed successfully!")

    else:
        st.warning("Please upload at least one PDF.")

# Chat section
st.subheader("💬 Chat")

query = st.text_input("Ask a question:")

if query:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process documents first.")
    else:
        retriever = st.session_state.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        docs = retriever.invoke(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        # Prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are a helpful AI assistant.
                Use only the provided context to answer the question.
                If you don't know, say: "I couldn't find the answer in the context."
                """),
                ("human", """Context: {context}

                Question: {question}
                """)
            ]
        )

        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        # LLM
        llm = ChatMistralAI(model="mistral-small-2506")

        response = llm.invoke(final_prompt)

        st.write("### 🤖 Answer:")
        st.write(response.content)