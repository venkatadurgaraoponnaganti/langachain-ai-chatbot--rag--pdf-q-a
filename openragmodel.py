import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import tempfile
import os

# Load environment variables
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="AI Assistant Chat", layout="wide")
st.title("🤖 Real-time Chat + PDF Q&A Assistant")

# Create two columns (left for file upload, right for chat)
left_col, right_col = st.columns([1, 3])

# Chat history and system message
chat_history = [SystemMessage(content="You are a helpful AI assistant.")]

# PDF-related logic
qa_chain = None

with left_col:
    # PDF Upload Section
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

        # LLM and QA chain setup
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

# Real-time Chat Section
with right_col:
    # Chat input
    user_input = st.text_input("Ask anything (or upload a PDF first):")

    if user_input:
        if user_input.lower() == "exit":
            st.stop()  # Stop if user types 'exit'

        # Add user input to chat history
        chat_history.append(HumanMessage(content=user_input))

        # If a PDF has been uploaded, try to answer based on PDF knowledge
        if qa_chain:
            # Use the PDF knowledge base for answers
            with st.spinner("Thinking..."):
                result = qa_chain.invoke(user_input)
                ai_response = result["result"]
                st.markdown("### 🤖 Answer from PDF knowledge:")
                st.write(ai_response)
        else:
            # Otherwise, provide a general AI assistant response
            model = ChatOpenAI(
                model="gpt-3.5-turbo",
                openai_api_key=os.getenv("OPENAI_API_KEY"),  # Replace with your OpenAI API key
                  # Adjust the temperature for randomness
            )

            # Get a response from the general assistant model
            result = model.invoke(chat_history)
            ai_response = result.content

            # Display the AI's response
            st.markdown("### 🤖 AI Response:")
            st.write(ai_response)

        # Add AI response to chat history
        chat_history.append(AIMessage(content=ai_response))

      
