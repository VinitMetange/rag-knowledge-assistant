"""
RAG Knowledge Assistant - FastAPI REST API
Author: Vinit Metange | AI Product Leader
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os

from rag_pipeline import RAGPipeline

app = FastAPI(
    title="RAG Knowledge Assistant API",
    description="Enterprise knowledge retrieval powered by RAG + LLMs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG pipeline instance
rag = RAGPipeline(persist_directory="./chroma_db")


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    question: str


class IngestTextRequest(BaseModel):
    texts: List[dict]  # [{"content": "...", "source": "...", "title": "..."}]


@app.get("/health")
def health_check():
    """Health check endpoint."""
    stats = rag.get_stats()
    return {"status": "healthy", "vectorstore": stats}


@app.post("/query", response_model=QueryResponse)
def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base with a question."""
    try:
        result = rag.query(request.question)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            question=request.question
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/text")
def ingest_text(request: IngestTextRequest):
    """Ingest plain text documents."""
    try:
        docs = rag.ingestion.load_texts(request.texts)
        count = rag.ingest(docs)
        return {"status": "success", "chunks_indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Ingest a PDF or text file."""
    if not file.filename.endswith((".pdf", ".txt", ".md")):
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and MD files supported")

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename)[1]
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        docs = rag.ingestion.load_directory(os.path.dirname(tmp_path))
        count = rag.ingest(docs)
        os.unlink(tmp_path)

        return {"status": "success", "filename": file.filename, "chunks_indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats():
    """Get vectorstore statistics."""
    return rag.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
