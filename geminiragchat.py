import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import tempfile
import os

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="Gemini PDF Chat", layout="wide")
st.title("🤖 Real-time Chat + PDF Q&A (Gemini)")

# Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a helpful AI assistant.")]

# Create two columns (left for file upload, right for chat)
left_col, right_col = st.columns([1, 3])

# PDF-related logic
qa_chain = None

with left_col:
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        st.success("✅ PDF uploaded and processed!")

        # Load and process PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        # Create embeddings and vectorstore
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",  # Corrected model name for embeddings
            google_api_key=google_api_key
        )
        vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

        # LLM and QA Chain
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=google_api_key)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

with right_col:
    user_input = st.text_input("Ask anything (or upload a PDF first):")

    if user_input:
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        if qa_chain:
            with st.spinner("🔎 Searching in uploaded PDF..."):
                result = qa_chain.invoke(user_input)
                ai_response = result["result"]
                st.markdown("### 📄 Answer from PDF:")
                st.write(ai_response)
        else:
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",  # Using Gemini 1.5 Pro for general chat
                google_api_key=google_api_key
            )
            with st.spinner("💬 Thinking..."):
                result = model.invoke(st.session_state.chat_history)
                ai_response = result.content
                st.markdown("### 💬 AI Response:")
                st.write(ai_response)

        st.session_state.chat_history.append(AIMessage(content=ai_response))
