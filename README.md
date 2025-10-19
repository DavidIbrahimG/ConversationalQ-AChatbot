# Conversational RAG Chatbot with PDF Uploads

## 🚀 Overview
This project is a **Conversational Retrieval-Augmented Generation (RAG) Chatbot** built using **Streamlit**, **LangChain**, and **Groq's Llama 3 model**.  
It allows users to **upload PDF documents** (such as research papers) and engage in **context-aware conversations** with their content.  
By leveraging embeddings, vector databases, and chat memory, the chatbot provides accurate, grounded, and contextual answers to user queries.

---

## 🧠 Key Features
- 📚 **PDF Uploads:** Upload and query multiple PDF documents.  
- 🧩 **Retrieval-Augmented Generation:** Combines semantic search with LLM reasoning.  
- 💬 **Conversational Memory:** Maintains chat history for context across turns.  
- ⚙️ **Embeddings + Vector Search:** Uses HuggingFace embeddings and Chroma for fast retrieval.  
- ⚡ **Groq Llama 3 Model:** High-performance, low-latency language model for answering questions.  
- 🧠 **Session Management:** Supports multiple sessions with unique retrievers and chat histories.

---
<img width="1470" height="956" alt="Screenshot 2025-10-19 at 12 45 40" src="https://github.com/user-attachments/assets/a78c6aed-6f05-42fe-a694-50ab041fa6e9" />
---
<img width="1470" height="956" alt="Screenshot 2025-10-19 at 12 47 16" src="https://github.com/user-attachments/assets/b6f72776-fd9e-432e-8c3d-a6e710d18f8c" />
---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Streamlit** – Interactive UI
- **LangChain** – Core orchestration framework
- **Groq (Llama 3.1-8b-instant)** – Main LLM for text generation
- **HuggingFace Sentence Embeddings** – For semantic vector representation
- **Chroma Vector Database** – Vector storage and retrieval
- **PyPDFLoader** – Extracts text from PDF files

---

## 📂 Project Structure
```
.
├── app.py                 # Main Streamlit app
├── .env                   # Environment variables (API keys)
├── requirements.txt       # Dependencies
├── research_papers/       # (optional) Local directory for PDFs
└── README.md              # Documentation
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/conversational-rag-chatbot.git
cd conversational-rag-chatbot
```

### 2️⃣ Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate    # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables
Create a `.env` file and add:
```
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

### 5️⃣ Run the App
```bash
streamlit run app.py
```

---

## 💻 Usage
1. Enter your **Groq API key** in the sidebar.  
2. **Upload one or more PDF files**.  
3. Wait for embeddings to build.  
4. Ask a question related to the uploaded documents.  
5. View real-time answers and chat history in the interface.

---

## 🔍 How It Works
1. **PDF Ingestion:** Text extracted from uploaded PDFs.  
2. **Chunking:** Long texts are split into manageable segments.  
3. **Embeddings:** Text chunks are converted into vector embeddings.  
4. **Storage:** Embeddings are stored in a Chroma vector database.  
5. **Retrieval:** When queried, the most relevant document chunks are retrieved.  
6. **LLM Response:** Groq’s Llama 3 generates an answer using the retrieved context.  
7. **Chat Memory:** History is saved for multi-turn, context-aware conversations.

---

## 🧩 Use Cases
- 🧑‍🎓 **Academic Research:** Chat with your papers to summarize and explore concepts.  
- 🏢 **Corporate Knowledge Base:** Query internal documents for instant insights.  
- 📚 **Education:** Interactive learning from textbooks or notes.  

---

## 🌱 Future Improvements
- 🔁 Persistent Chroma storage across sessions  
- 🔊 Voice-based input and output  
- 🌍 Multi-language document support  
- 🧠 Enhanced summarization and question generation  

---

## 👩‍💻 Author
Developed by **[Your Name]**  
📧 [your.email@example.com]  
🔗 [LinkedIn / GitHub Profile]

---

## 🪪 License
This project is licensed under the **MIT License**.  
Feel free to use and modify it with attribution.

---

### 💬 Short Description
> A Streamlit-based conversational RAG chatbot that lets users upload PDFs and interact with them using Groq’s Llama 3 model. It integrates retrieval, embeddings, and chat memory for accurate, context-aware responses.
