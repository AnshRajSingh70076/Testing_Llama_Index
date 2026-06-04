RAG Pipeline using LlamaCloud, FAISS, Groq, and LangSmith
Overview

This project implements a Retrieval-Augmented Generation (RAG) system that processes PDF documents using LlamaCloud, stores embeddings using FAISS, and generates answers using Groq LLM. The system also includes evaluation using LangSmith to measure retrieval and answer quality.

The goal of this project is to build a complete document question-answering pipeline that can scale to real-world enterprise documents.

Features
PDF parsing using LlamaCloud (layout-aware extraction)
Chunking using LangChain text splitters
Embedding generation using SentenceTransformers
Vector storage using FAISS
Question answering using Groq LLM (Llama 3)
RAG pipeline with context-based responses
Evaluation using LangSmith datasets and evaluators
Custom dataset support for benchmarking
Project Structure
Testing_Llama_Index/
│
├── main.py                # RAG pipeline implementation
├── llama.py               # LlamaCloud PDF parsing
├── dataset.json           # Evaluation dataset
├── README.md              # Project documentation
├── requirements.txt      # Python dependencies
├── .gitignore            # Ignored files configuration
Installation
1. Clone the repository
git clone https://github.com/AnshRajSingh70076/Testing_Llama_Index.git
cd Testing_Llama_Index
2. Create virtual environment using uv
uv venv
3. Activate environment

Windows:

.venv\Scripts\activate
4. Install dependencies
uv pip install -r requirements.txt
Environment Variables

Create a .env file in the root directory and add the following:

LLAMA_CLOUD_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
LANGSMITH_API_KEY=your_key_here
How It Works
1. Document Parsing

PDF files are parsed using LlamaCloud. The output is converted into structured markdown.

2. Chunking

The markdown is split into smaller chunks using recursive text splitters.

3. Embeddings

Each chunk is converted into vector embeddings using a sentence-transformer model.

4. Vector Storage

Embeddings are stored in FAISS for fast similarity search.

5. Retrieval + Generation

For each query:

Relevant chunks are retrieved from FAISS
Context is passed to Groq LLM
Final answer is generated using only retrieved context
RAG Function

The system uses a simple function:

ask(query)

Steps:

Retrieve top-k relevant chunks
Build context
Send to LLM
Return final response
Evaluation using LangSmith

The system includes evaluation setup:

Dataset of questions and expected answers
Custom RAG pipeline wrapper
Evaluation using LangSmith evaluate() function

Results are logged in LangSmith dashboard for analysis.

Example Usage
from main import ask

response = ask("What is MCP?")
print(response)
Dataset Format

The evaluation dataset follows this structure:

[
  {
    "input": "Question here",
    "expected_output": "Expected answer here"
  }
]
Requirements
Python 3.10+
LlamaCloud API access
Groq API key
LangSmith API key
Key Libraries
langchain
langchain-groq
langchain-community
langchain-huggingface
faiss-cpu
sentence-transformers
python-dotenv
llama-cloud
Future Improvements
Add reranking for better retrieval accuracy
Use hybrid search (BM25 + vector search)
Add FastAPI backend for deployment
Improve evaluation metrics
Add streaming response support
License

This project is for educational and research purposes.
