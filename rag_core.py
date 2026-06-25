"""
Core RAG logic: load vector store, retrieve relevant chunks,
filter out references/citations, and generate a grounded answer
using Groq's free-tier API (Llama-3-8B) instead of a local model.
"""
import os
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
CHROMA_DIR = "./chroma_db"


def looks_like_reference_or_title(text):
    """Filter out citation lists, page headers, and author/affiliation blocks."""
    signals = [
        "doi.org" in text,
        "https://" in text[:80],
        bool(re.search(r"\[\d+\]", text)),
        "@acm.org" in text or "@dcc" in text,
        text.count(",") > 6 and "University" in text,
    ]
    return any(signals)


class RagEngine:
    def __init__(self):
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        print("Loading vector store...")
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings
        )

        print("Connecting to Groq API...")
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Check your .env file.")
        self.client = Groq(api_key=api_key)
        print("RAG engine ready.")

    def retrieve(self, question, k=15, max_chunks=4):
        """Retrieve relevant chunks and filter out reference/citation noise."""
        results = self.vectorstore.similarity_search(question, k=k)
        body_chunks = [
            doc for doc in results
            if not looks_like_reference_or_title(doc.page_content)
        ]

        if not body_chunks:
            body_chunks = results

        return body_chunks[:max_chunks]

    def generate_answer(self, question, chunks):
        """Generate a grounded answer from the given context chunks using Groq."""
        context = "\n\n".join([doc.page_content for doc in chunks])

        prompt = f"""Answer the question using only the context below. Be precise and concise (2-3 sentences). If the context doesn't fully answer the question, say what it does cover.

Context:
{context}

Question: {question}

Answer:"""

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=250,
        )
        return response.choices[0].message.content.strip()

    def ask(self, question):
        """Full pipeline: retrieve + generate. Returns (answer, sources)."""
        chunks = self.retrieve(question)
        if not chunks:
            return "I couldn't find relevant information in the documents to answer that.", []

        answer = self.generate_answer(question, chunks)
        sources = [
            {"file": doc.metadata.get("source", "unknown"), "page": doc.metadata.get("page", "?")}
            for doc in chunks
        ]
        return answer, sources