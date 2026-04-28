from dotenv import load_dotenv


load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate


embedding_model = HuggingFaceEmbeddings()

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4, 
        "fetch_k": 10,
        "lambda_mult":0.5
    }
)

llm = ChatMistralAI(model = "mistral-small-2506")

#prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful AI assistant.
         
         Use only the provided context to answer the question. If you don't know the answer, 
         say:"i couldn't find the answer in the context."
         """),
        ("human", """Context: {context}
         
         Question:{question}
         """)
    ]
)
print("Rag Based Ai Chat bot")
print("0 to exit")

while True:
    
    query = input("You : ")
    if query == "0":
        break
    
    docs = retriever.invoke(query)
    
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )
    
    response = llm.invoke(final_prompt)
    
    print("\n AI : ", response.content)
