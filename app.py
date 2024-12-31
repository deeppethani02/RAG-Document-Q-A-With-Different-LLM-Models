import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

# Load the GROQ API Key
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# If you do not have OpenAI key, use the below Huggingface embedding
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to create LLM instance based on the selected model
def get_llm_model(selected_model):
    if selected_model == "distil-whisper-large-v3-en":
        return ChatGroq(groq_api_key=groq_api_key, model_name="distil-whisper-large-v3-en")
    elif selected_model == "gemma2-9b-it":
        return ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")
    elif selected_model == "gemma-7b-it":
        return ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it")
    elif selected_model == "llama-3.1-70b-versatile":
        return ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")
    elif selected_model == "llama-3.1-8b-instant":
        return ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
    elif selected_model == "llama-guard-3-8b":
        return ChatGroq(groq_api_key=groq_api_key, model_name="llama-guard-3-8b")
    elif selected_model == "llama3-70b-8192":
        return ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    elif selected_model == "mixtral-8x7b-32768":
        return ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
    else:
        return None

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question from pdf or llm models.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def create_vector_embedding(documents):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("RAG Document Q&A With Different LLM Models")

# Sidebar for file upload and model selection
st.sidebar.header("Upload PDF Files")
uploaded_files = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# Model selection dropdown
model_options = ["llama3-70b-8192",
    "gemma2-9b-it",
    "gemma-7b-it",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-guard-3-8b",
    "mixtral-8x7b-32768",
    "distil-whisper-large-v3-en",
]
selected_model = st.sidebar.selectbox("Select LLM Model", model_options)

# Process uploaded PDFs
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        with open(f"./temp_{uploaded_file.name}", "wb") as file:
            file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(f"./temp_{uploaded_file.name}")
            docs = loader.load()
            documents.extend(docs)

    create_vector_embedding(documents)
    st.sidebar.write("Vector Database is ready")

user_prompt = st.text_input("Enter your query from the documents")

if st.button("Submit Query"):
    if "vectors" in st.session_state:
        llm = get_llm_model(selected_model)
        if llm is None:
            st.error("Selected model is not available.")
        else:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({'input': user_prompt})
            st.write(f"Response time: {time.process_time() - start}")

            st.write(response['answer'])
    else:
        st.warning("Please upload PDF files and prepare the vector database first.")