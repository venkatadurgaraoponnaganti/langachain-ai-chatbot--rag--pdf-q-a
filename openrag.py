import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import tempfile

# Load environment variables
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="PDF QA Chat", layout="wide")
st.title("📄 Chat with your PDF using LangChain + OpenAI")

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

        st.success("PDF uploaded and processed!")

        # Load and process PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(disallowed_special=())
        vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

        # LLM and QA chain
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

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
