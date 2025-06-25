import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import tempfile

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="PDF QA Chat", layout="wide")
st.title("📄 Chat with your PDF using LangChain + Gemini")

# Create two columns
left_col, right_col = st.columns([1, 3])

qa_chain = None  # Initialize globally

with left_col:
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        st.success("✅ PDF uploaded and processed!")

        try:
            # Load and process PDF
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()

            # Split text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)

            # Create embeddings and vector store
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",  # ✅ No "models/" prefix
                google_api_key=google_api_key
            )
            vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

            # LLM and QA chain setup
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro-latest",  # ✅ Updated model
                google_api_key=google_api_key
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )
        finally:
            # Clean up the temp file
            os.remove(tmp_file_path)

# Q/A Section - Always Visible
with right_col:
    question = st.text_input("Ask a question about the uploaded PDF:")

    if question:
        if qa_chain is None:
            st.warning("⚠️ Please upload a PDF first.")
        else:
            with st.spinner("Thinking..."):
                result = qa_chain.invoke(question)
                st.markdown("### 🤖 Answer:")
                st.write(result["result"])
                
                
            
