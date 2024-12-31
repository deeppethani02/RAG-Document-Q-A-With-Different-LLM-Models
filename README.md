# RAG-Document-Q-A-With-Different-LLM-Models

## **Overview**
This project demonstrates how to implement a **Retrieval-Augmented Generation (RAG)** system using **Streamlit** and **ChatGroq LLMs**. The system processes PDF documents, creates embeddings for document retrieval, and uses various Large Language Models (LLMs) for answering user queries based on the content of the uploaded documents.  

Key Features:
- Supports multiple LLMs such as `llama3-70b-8192`, `gemma2-9b-it`, and `distil-whisper-large-v3-en`.
- PDF ingestion and vector embedding creation using **HuggingFace embeddings**.
- User-friendly interface built with **Streamlit**.
- Context-aware Q&A based on uploaded documents.


## **Process Steps**

### **1. Setup and Dependencies**
1. Install the required dependencies:
   ```bash
   pip install streamlit langchain langchain_community langchain_huggingface langchain_groq python-dotenv
   ```
2. Create a `.env` file and add the following keys:
   ```env
   GROQ_API_KEY=<your_groq_api_key>
   HF_TOKEN=<your_huggingface_api_key>
   ```


### **2. How It Works**

#### **Step 1: Upload PDF Files**
- Users upload multiple PDF files via the sidebar in the Streamlit app.
- The PDFs are processed using the `PyPDFLoader`, and the content is split into smaller chunks using `RecursiveCharacterTextSplitter`.

#### **Step 2: Create Vector Embeddings**
- The content of the PDF documents is embedded using **HuggingFace Embeddings** (`all-MiniLM-L6-v2`) to facilitate similarity-based retrieval.
- The embeddings are stored in a vector database using **FAISS**.

#### **Step 3: Select an LLM**
- Users select one of the available **ChatGroq LLMs** from the dropdown menu. Supported models include:
  - `llama3-70b-8192`
  - `gemma2-9b-it`
  - `distil-whisper-large-v3-en`
  - And more.

#### **Step 4: Enter a Query**
- Users provide a natural language query, which the system processes using the selected LLM and context from the uploaded documents.

#### **Step 5: Retrieve and Answer**
- A **retrieval chain** is created by combining the document retriever and the selected LLM.
- The system retrieves relevant chunks of text from the vector database and generates an answer using the selected LLM.

#### **Step 6: View Results**
- The generated answer is displayed along with the response time.
- Users can expand the results to view document similarity searches.


## **Summary**
This project integrates modern tools like **LangChain**, **FAISS**, and **HuggingFace embeddings** to create a highly flexible and efficient RAG system. It allows users to interact with uploaded PDF documents through natural language queries and provides seamless integration with various LLMs for improved context-aware responses.


## **Future Enhancements**
- Add support for additional document formats (e.g., Word, Excel).
- Integrate other vector databases like **Pinecone** or **Weaviate**.
- Enhance UI for better interactivity and visualization.
