# 📄 RAG-Based AI Chatbot (Streamlit + Mistral)

An interactive **Retrieval-Augmented Generation (RAG)** chatbot built using **Streamlit**, **LangChain**, and **Mistral AI**.
Upload PDF documents and ask questions — the chatbot answers based only on the document content.

---

## 🚀 Features

* 📂 Upload multiple PDF files
* ✂️ Automatic document chunking
* 🔎 Semantic search using embeddings
* 🧠 Context-aware answers using Mistral LLM
* 💬 Interactive chat interface (Streamlit)
* 📁 Persistent vector database using Chroma

---

## 🛠️ Tech Stack

* Python
* Streamlit
* LangChain
* Mistral AI (`langchain_mistralai`)
* HuggingFace Embeddings
* Chroma Vector DB

---

## 📁 Project Structure

```
.
├── app.py                 # Main Streamlit app
├── project.py             # Additional logic (if used)
├── requirements.txt       # Dependencies
├── .gitignore             # Ignored files
├── chroma_db/             # Vector database (auto-created)
├── documents/             # Optional document storage
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

---

### 2️⃣ Create virtual environment

```
python -m venv .venv
source .venv/Scripts/activate   # Windows
```

---

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Set environment variables

Create a `.env` file:

```
MISTRAL_API_KEY=your_api_key_here
```

⚠️ **Important:** Never upload `.env` to GitHub.

---

## ▶️ Run the App

```
streamlit run app.py
```

---

## 💡 How It Works

1. Upload PDF files via sidebar
2. PDFs are loaded using `PyPDFLoader`
3. Text is split into chunks
4. Embeddings generated using HuggingFace model
5. Stored in Chroma vector database
6. User query → similarity search (MMR)
7. Relevant chunks passed to Mistral LLM
8. Response generated using context

---

## 📌 Example Use Cases

* 📚 Study assistant for textbooks
* 📄 Document Q&A system
* 🧠 Knowledge base chatbot
* 🏢 Internal company document search

---

## ⚠️ Notes

* Do not upload large PDFs (>50MB) to GitHub
* `chroma_db/` should not be committed
* API keys must be kept secret
* Works best with clean, text-based PDFs

---

## 🔐 Security

* API keys are stored using `.env`
* Sensitive files are excluded via `.gitignore`

---

## 📈 Future Improvements

* Add chat history memory
* Support for DOCX / TXT files
* Deploy on cloud (Streamlit Cloud / AWS)
* UI enhancements

---

## 🙌 Author

**Rakshith K N**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!

