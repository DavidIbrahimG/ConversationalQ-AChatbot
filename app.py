# app.py — Conversational RAG with PDF uploads + chat history (no undefined helpers)

import os, uuid, tempfile, time
from operator import itemgetter

import streamlit as st
from dotenv import load_dotenv

# LangChain (modern, modular)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory


# ---------- Setup ----------
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload PDFs and chat with their content")

api_key = st.text_input("Enter your Groq API key:", type="password")
session_id = st.text_input("Session ID", value="default_session")

# Per-session state
if "store" not in st.session_state:
    st.session_state.store = {}
if "retrievers" not in st.session_state:
    st.session_state.retrievers = {}


def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def save_temp_pdf(uploaded_file) -> str:
    tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pdf")
    with open(tmp, "wb") as f:
        f.write(uploaded_file.getvalue())
    return tmp


if api_key:
    # Use a valid Groq model
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

    uploaded_files = st.file_uploader(
        "Choose PDF file(s)", type="pdf", accept_multiple_files=True
    )

    # Build/refresh retriever when PDFs are uploaded
    if uploaded_files:
        docs = []
        for up in uploaded_files:
            tmp_pdf = save_temp_pdf(up)
            loader = PyPDFLoader(tmp_pdf)
            docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        splits = splitter.split_documents(docs)

        # Make both collection and document IDs explicit STRINGS
        collection_name = f"rag_{session_id}".replace(" ", "_")
        ids = [str(uuid.uuid4()) for _ in range(len(splits))]

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=collection_name,
            ids=ids,
           
        )
        retriever = vectorstore.as_retriever()
        st.session_state.retrievers[session_id] = retriever
        st.success("Vector store built. You can now ask questions.")

    retriever = st.session_state.retrievers.get(session_id)
    if not retriever:
        st.info("Upload one or more PDFs to build the vector store, then ask a question.")
    else:
        # -------- History-aware question rewrite --------
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given a chat history and the latest user question which might reference "
             "context in the chat history, rewrite it as a standalone question. "
             "Do NOT answer it, only rewrite."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        def pick_history_and_input(x):
            return {"chat_history": x["chat_history"], "input": x["input"]}

        history_aware_retriever = (
            RunnableLambda(pick_history_and_input)
            | contextualize_q_prompt
            | llm
            | StrOutputParser()
            | retriever
        )

        # -------- QA over retrieved context --------
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the retrieved context to answer the question. "
            "If you don't know, say you don't know. "
            "Use at most three sentences.\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        rag_chain = (
            {
                "context": history_aware_retriever | RunnableLambda(format_docs),
                "input": itemgetter("input"),                # string only
                "chat_history": itemgetter("chat_history"),  # list[BaseMessage]
            }
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)

            start = time.process_time()
            response = conversational_rag_chain.invoke(
                {"input": user_input, "chat_history": session_history.messages},
                config={"configurable": {"session_id": session_id}},
            )
            dur = time.process_time() - start

            st.write(f"**Response time:** {dur:.2f}s")
            st.write("**Assistant:**", response)

            with st.expander("Debug: Chat History"):
                st.write(session_history.messages)
else:
    st.warning("Please enter your Groq API key to begin.")
