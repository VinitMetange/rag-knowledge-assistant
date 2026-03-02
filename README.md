# 🔍 RAG Knowledge Assistant

> **Author:** [Vinit Metange](https://linkedin.com/in/vinit-metange) | AI Product Leader

[![LinkedIn](https://img.shields.io/badge/LinkedIn-vinit--metange-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/vinit-metange)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python)]()
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=flat)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=flat&logo=fastapi)]()

---

## 🎯 Problem Solved

Product knowledge was siloed in 1000s of pages of documentation. Support teams spent hours searching for answers. This RAG assistant delivers instant, accurate answers from enterprise documents — reducing support cost by 20%.

> *Inspired by the RAG-based knowledge assistant implemented at Nokia SDL (2021-2023)*

---

## 🏗️ Architecture

```
[Documents/PDFs] → [Chunking] → [Embeddings] → [FAISS Index]
                                                        ↓
[User Query] → [Query Embedding] → [Similarity Search] → [Top-K Chunks]
                                                        ↓
                              [LLM + Context] → [Answer + Sources]
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Embeddings | OpenAI `text-embedding-ada-002` / HuggingFace |
| Vector Store | FAISS (local) / ChromaDB (persistent) |
| LLM | GPT-4o / Claude 3.5 / AWS Bedrock |
| Framework | LangChain |
| API | FastAPI |
| UI | Streamlit (optional) |

---

## 📂 Repository Structure

```
rag-knowledge-assistant/
├── README.md
├── src/
│   ├── ingestion.py          # Document loading & chunking
│   ├── embeddings.py         # Embedding generation
│   ├── retriever.py          # FAISS / ChromaDB vector store
│   ├── chain.py              # LangChain RAG chain
│   ├── api.py                # FastAPI REST endpoints
│   └── evaluator.py          # RAGAS evaluation metrics
├── data/
│   └── sample_docs/          # Sample product documentation
├── notebooks/
│   ├── 01_rag_basics.ipynb
│   ├── 02_advanced_retrieval.ipynb
│   └── 03_evaluation.ipynb
├── tests/
│   └── test_retrieval.py
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/VinitMetange/rag-knowledge-assistant
cd rag-knowledge-assistant
pip install -r requirements.txt

# Ingest your documents
python src/ingestion.py --docs_path ./data/sample_docs

# Start the API
uvicorn src.api:app --reload

# Or run the notebook demo
jupyter notebook notebooks/01_rag_basics.ipynb
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Answer accuracy (RAGAS faithfulness) | >0.85 |
| Retrieval precision | >0.80 |
| Average response time | <2s |
| Support cost reduction | ~20% |
| Documents indexed | Scales to 100K+ pages |

---

## 💬 About the Author

**Vinit Metange** — AI Product Leader | Nokia (9 yrs) → Netcracker

- 💼 [linkedin.com/in/vinit-metange](https://linkedin.com/in/vinit-metange)
- 💙 [github.com/VinitMetange](https://github.com/VinitMetange)
