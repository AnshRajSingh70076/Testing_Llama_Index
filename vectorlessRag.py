import os
import json
import time
import numpy as np

from dotenv import load_dotenv
from pageindex import PageIndexClient
from groq import Groq

from langsmith import Client
from langsmith.evaluation import evaluate
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================

PDF_PATH = r"C:\Users\rajs1\Downloads\Testing_Llama_Index\finalmcp.pdf"
DATASET_PATH = r"C:\Users\rajs1\Downloads\Testing_Llama_Index\dataset.json"

load_dotenv()

# =========================
# CLIENTS
# =========================

page_client = PageIndexClient(
    api_key=os.getenv("PAGEINDEX_API_KEY")
)

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

ls_client = Client()

# =========================
# UPLOAD DOCUMENT
# =========================

print("Uploading PDF...")

upload_result = page_client.submit_document(PDF_PATH)
print(upload_result)

doc_id = upload_result["doc_id"]
print(f"Document ID: {doc_id}")

# =========================
# WAIT FOR INDEXING
# =========================

print("Waiting for document processing...")

while True:
    try:
        ready = page_client.is_retrieval_ready(doc_id)
        print("Ready:", ready)

        if ready:
            break

    except Exception as e:
        print("Waiting...", e)

    time.sleep(5)

print("Document ready!")

# =========================
# LOAD DATASET
# =========================

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_dataset = json.load(f)

print(f"Loaded {len(raw_dataset)} examples")

# =========================
# CREATE LANGSMITH DATASET
# =========================

dataset_name = "pageindex-groq-rag-dataset5"

try:
    ls_client.delete_dataset(dataset_name=dataset_name)
except:
    pass

dataset_obj = ls_client.create_dataset(
    dataset_name=dataset_name,
    description="PageIndex + Groq RAG evaluation5"
)

for row in raw_dataset:
    ls_client.create_example(
        dataset_id=dataset_obj.id,
        inputs={"question": row["input"]},
        outputs={"expected": row["expected_output"]}
    )

# =========================
# FIXED ASK FUNCTION (IMPORTANT CHANGE)
# =========================
# ❗ We FIX your main issue here:
# Instead of submit_query/get_retrieval (deprecated & broken),
# we use PageIndex Chat API directly.

# def ask(question: str) -> str:

#     print("\nQuestion:", question)

#     response = page_client.chat_completions(
#         doc_id=doc_id,
#         messages=[
#             {"role": "user", "content": question}
#         ],
#         enable_citations=True
#     )

#     print("PageIndex Response:", response)

#     # Extract answer safely
#     if isinstance(response, dict):
#         return (
#             response.get("answer")
#             or response.get("output")
#             or str(response)
#         )

#     return str(response)

def ask(question: str) -> str:

    print("\nQuestion:", question)

    response = page_client.chat_completions(
        doc_id=doc_id,
        messages=[
            {"role": "user", "content": question}
        ],
        enable_citations=True
    )

    print("PageIndex Response:", response)

    # =========================
    # SAFE EXTRACTION (IMPORTANT FIX)
    # =========================

    try:
        # OpenAI-style response (MOST COMMON)
        return response["choices"][0]["message"]["content"]

    except Exception:
        pass

    try:
        # fallback 1
        return response.get("answer")

    except Exception:
        pass

    try:
        # fallback 2
        return response.get("output")

    except Exception:
        pass

    # final fallback
    return str(response)

# =========================
# TARGET
# =========================

def target(inputs):
    return {
        "output": ask(inputs["question"])
    }

# =========================
# EVALUATOR (SEMANTIC SIMILARITY)
# =========================

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_evaluator(run, example):

    pred = run.outputs["output"]
    exp = example.outputs["expected"]

    emb1 = model.encode(pred)
    emb2 = model.encode(exp)

    score = np.dot(emb1, emb2) / (
        np.linalg.norm(emb1) *
        np.linalg.norm(emb2)
    )

    return {"score": float(score)}

# =========================
# LANGSMITH EVALUATION
# =========================

results = evaluate(
    target,
    data=dataset_obj,
    evaluators=[semantic_evaluator],
    experiment_prefix="pageindex-groq-rag5"
)

# print(results)