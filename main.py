# from dotenv import load_dotenv

# load_dotenv()

# from llama import get_markdown
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_groq import ChatGroq


# # Parse document
# markdown = get_markdown(
#     r"C:\Users\rajs1\Downloads\Testing_Llama_Index\finalmcp.pdf"
# )

# print(f"Markdown length: {len(markdown)}")


# # Split into chunks
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# chunks = splitter.split_text(markdown)

# print(f"Chunks created: {len(chunks)}")


# # Create embeddings
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )


# # Store in FAISS
# vectorstore = FAISS.from_texts(
#     chunks,
#     embeddings
# )


# # Create retriever
# retriever = vectorstore.as_retriever(
#     search_kwargs={"k": 3}
# )


# # Groq LLM
# llm = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     temperature=0
# )


# # # Ask question
# # query = "What is this document about?"


# # # Retrieve relevant chunks
# # docs = retriever.invoke(query)

# # print(f"Retrieved {len(docs)} chunks")


# # # Build context
# # context = "\n\n".join(
# #     doc.page_content
# #     for doc in docs
# # )


# # # Send context to Groq
# # response = llm.invoke(
# #     f"""
# # You are a helpful assistant.

# # Answer the question using only the provided context.

# # Context:
# # {context}

# # Question:
# # {query}
# # """
# # )

# # print("\nAnswer:")
# # print(response.content)

# def ask(query):
#     docs = retriever.invoke(query)

#     context = "\n\n".join(doc.page_content for doc in docs)

#     response = llm.invoke(
#         f"""
# Answer using only context.

# Context:
# {context}

# Question:
# {query}
# """
#     )

#     return response.content

from dotenv import load_dotenv
import os
import json

load_dotenv()

# =========================
# LANGSMITH SETUP
# =========================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "mcp-rag-eval"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")


from llama import get_markdown
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langsmith import Client
from langsmith.evaluation import evaluate


# =========================
# PARSE DOCUMENT
# =========================
markdown = get_markdown(
    r"C:\Users\rajs1\Downloads\Testing_Llama_Index\finalmcp.pdf"
)

print(f"Markdown length: {len(markdown)}")


# =========================
# CHUNKING
# =========================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(markdown)

print(f"Chunks created: {len(chunks)}")


# =========================
# EMBEDDINGS
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# VECTOR STORE
# =========================
vectorstore = FAISS.from_texts(
    chunks,
    embeddings
)


# =========================
# RETRIEVER
# =========================
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)


# =========================
# GROQ LLM
# =========================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


# =========================
# RAG FUNCTION
# =========================
def ask(query):
    docs = retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in docs)

    response = llm.invoke(
        f"""
Answer ONLY using the context below.
If answer is not present, say "Not found in document".

Context:
{context}

Question:
{query}
"""
    )

    return response.content


# =========================
# LOAD DATASET
# =========================
with open("dataset.json", "r", encoding="utf-8") as f:
    dataset_json = json.load(f)

client = Client()

# Create dataset (RUN ONCE SAFE)
dataset = client.create_dataset(
    dataset_name="mcp-rag-eval2"
)

examples = [
    {
        "inputs": {"question": item["input"]},
        "outputs": {"expected": item["expected_output"]}
    }
    for item in dataset_json
]

client.create_examples(
    dataset_id=dataset.id,
    examples=examples
)


# =========================
# TARGET FUNCTION
# =========================
def target(inputs):
    return ask(inputs["question"])


# =========================
# CUSTOM EVALUATOR
# =========================
# def exact_match_evaluator(run, example):
#     prediction = run.outputs["output"].lower()
#     expected = example.outputs["expected"].lower()

#     score = 1 if expected in prediction else 0

#     return {"score": score}
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_evaluator(run, example):
    pred = run.outputs["output"]
    exp = example.outputs["expected"]

    emb1 = model.encode(pred)
    emb2 = model.encode(exp)

    score = np.dot(emb1, emb2) / (
        np.linalg.norm(emb1) * np.linalg.norm(emb2)
    )

    return {"score": float(score)}

# =========================
# RUN EVALUATION (FIXED)
# =========================
results = evaluate(
    target,
    data=dataset,   # ✅ IMPORTANT FIX (NOT formatted_data)
    evaluators=[semantic_evaluator],
    experiment_prefix="mcp-rag-eval2"
)

print(results)