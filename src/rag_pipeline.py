"""
RAG Knowledge Assistant - Retrieval-Augmented Generation Pipeline
Author: Vinit Metange | AI Product Leader
GitHub: https://github.com/VinitMetange
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


RAG_PROMPT = PromptTemplate(
    template="""You are an expert knowledge assistant. Use the provided context to answer 
questions accurately and concisely. If the context doesn't contain enough information, 
say so clearly rather than guessing.

Context:
{context}

Question: {question}

Answer (be specific and cite sources when possible):""",
    input_variables=["context", "question"]
)


class DocumentIngestion:
    """Handles document loading, splitting, and indexing."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_directory(self, directory: str, glob: str = "**/*.{pdf,txt,md}") -> List[Document]:
        """Load all documents from a directory."""
        docs = []
        directory_path = Path(directory)

        for pattern, loader_cls in [
            ("**/*.pdf", PyPDFLoader),
            ("**/*.txt", TextLoader),
            ("**/*.md", TextLoader),
        ]:
            for file_path in directory_path.glob(pattern):
                try:
                    loader = loader_cls(str(file_path))
                    docs.extend(loader.load())
                    logger.info(f"Loaded: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")

        logger.info(f"Total documents loaded: {len(docs)}")
        return docs

    def load_texts(self, texts: List[Dict[str, str]]) -> List[Document]:
        """Create documents from plain text dictionaries."""
        return [
            Document(
                page_content=item["content"],
                metadata={"source": item.get("source", "manual"), "title": item.get("title", "")}
            )
            for item in texts
        ]

    def split(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks


class RAGPipeline:
    """
    Production RAG pipeline with ChromaDB vector store.
    Supports document ingestion, retrieval, and QA.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        model: str = "gpt-4o",
        k_results: int = 4,
    ):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.persist_directory = persist_directory
        self.k_results = k_results
        self.vectorstore: Optional[Chroma] = None
        self.qa_chain: Optional[RetrievalQA] = None
        self.ingestion = DocumentIngestion()

        # Load existing vectorstore if available
        if os.path.exists(persist_directory):
            self._load_vectorstore()

    def _load_vectorstore(self):
        """Load existing Chroma vectorstore from disk."""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        self._build_qa_chain()
        logger.info(f"Loaded vectorstore from {self.persist_directory}")

    def _build_qa_chain(self):
        """Build the RetrievalQA chain."""
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={"k": self.k_results, "fetch_k": 20}
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": RAG_PROMPT},
            return_source_documents=True,
        )

    def ingest(self, documents: List[Document]) -> int:
        """Ingest documents into the vector store."""
        chunks = self.ingestion.split(documents)

        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vectorstore.add_documents(chunks)

        self._build_qa_chain()
        logger.info(f"Ingested {len(chunks)} chunks into vectorstore")
        return len(chunks)

    def ingest_directory(self, directory: str) -> int:
        """Ingest all documents from a directory."""
        docs = self.ingestion.load_directory(directory)
        return self.ingest(docs)

    def query(self, question: str) -> Dict[str, Any]:
        """Query the knowledge base."""
        if self.qa_chain is None:
            return {"answer": "No documents ingested yet.", "sources": []}

        result = self.qa_chain.invoke({"query": question})
        sources = [
            {
                "content": doc.page_content[:200],
                "source": doc.metadata.get("source", "unknown")
            }
            for doc in result.get("source_documents", [])
        ]
        return {"answer": result["result"], "sources": sources}

    def get_stats(self) -> Dict[str, Any]:
        """Return vectorstore statistics."""
        if self.vectorstore is None:
            return {"documents": 0, "status": "empty"}
        count = self.vectorstore._collection.count()
        return {"documents": count, "status": "ready", "persist_dir": self.persist_directory}


if __name__ == "__main__":
    rag = RAGPipeline()

    # Demo: ingest sample texts
    sample_docs = rag.ingestion.load_texts([
        {"content": "LangChain is a framework for developing applications powered by language models.", "source": "demo"},
        {"content": "RAG (Retrieval Augmented Generation) combines retrieval with generation for better accuracy.", "source": "demo"},
    ])
    rag.ingest(sample_docs)

    result = rag.query("What is LangChain?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
