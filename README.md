# Course Notes RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in academic papers on recommender systems, with sources cited. Built end-to-end: PDF ingestion, vector search, and LLM-based generation, deployed as a live Streamlit app.

**Live app:** _add your Streamlit Cloud link here after deploying_

---

## Why this project

Reading through dense academic papers to find specific facts is slow. This assistant lets you ask a question in plain English and get an answer grounded in the actual paper text, with the source page cited — built as a hands-on exploration of Retrieval-Augmented Generation, a core technique behind modern AI assistants.

---

## Architecture

```
Ingestion (build once):
  PDFs -> strip references sections -> chunk (700 chars, 100 overlap)
       -> embed (HuggingFace sentence-transformers) -> Chroma vector store

Query (runs per question):
  Question -> embed -> similarity search (top 15, filtered to top 4 body chunks)
           -> Groq API (Llama-3-8B) generates answer from retrieved context
           -> Answer + cited sources shown in Streamlit UI
```

---

## Tech stack

- **LangChain** — document loading, chunking, retrieval
- **Chroma** — local, persistent vector database
- **HuggingFace `sentence-transformers`** — local embedding model (`all-MiniLM-L6-v2`)
- **Groq API** — fast, free-tier hosted inference (`llama-3.1-8b-instant`) for answer generation
- **Streamlit** — chat interface, deployed on Streamlit Cloud

---

## Why Groq instead of a fully local LLM

The project initially ran fully local, including generation, using `flan-t5-large` (770M parameters). On evaluation, this model frequently:
- Copied raw text (including garbled LaTeX-derived math) instead of synthesizing an answer
- Gave short, unhelpful non-answers on harder questions

Upgrading to a larger local model (Mistral-7B) wasn't feasible on the development machine's 8GB RAM. Switching the **generation** step to Groq's free-tier hosted API (while keeping retrieval, embeddings, and the vector store fully local) solved this without requiring new hardware:

| Metric | Local `flan-t5-large` | Groq `Llama-3-8B` |
|---|---|---|
| Retrieval accuracy | 100% | 100% |
| Strict keyword match (eval set) | 16.7% | **41.7%** |
| Semantic similarity (eval set) | 28.4% | 34.0% |
| Answer coherence | Often garbled/copied | Consistently fluent |

This is a deliberate, documented architecture decision rather than a default choice — a common real-world tradeoff between fully local inference and hosted APIs.

---

## Evaluation

A hand-built set of 6 question/answer pairs (`qa_set.json`), grounded in the actual source papers, is used to measure:
- **Retrieval accuracy** — did the correct source PDF appear in the retrieved chunks?
- **Keyword match** — does the generated answer contain the expected key facts?
- **Semantic similarity** — cosine similarity between the answer and expected keywords, to credit correct answers phrased differently than the literal expected text

Run the evaluation yourself:
```bash
python evaluate.py
```

### Known limitations
- Two of six test questions (asking for an exact dataset name and a list of metrics) are only partially answered — the model is honest about this rather than fabricating an answer, which is the preferred failure mode for a grounded assistant.
- Academic PDFs convert LaTeX math notation into garbled symbol strings during text extraction; a dedicated math-aware chunking filter was attempted but rejected after it removed legitimate technical content.
- The reference-stripping step in `ingest.py` removes citation lists to prevent retrieval from matching on citation titles rather than body text discussion.

---

## Project structure

```
course-rag/
├── docs/                  # source PDFs (not committed)
├── chroma_db/             # vector store (not committed, rebuild with ingest.py)
├── ingest.py              # load, clean, chunk, embed, build vector store
├── rag_core.py            # RagEngine: retrieval + Groq-based generation
├── app.py                 # Streamlit chat UI
├── evaluate.py            # retrieval + answer-quality evaluation
├── qa_set.json            # evaluation question/answer set
├── .env                   # GROQ_API_KEY (not committed)
├── .gitignore
└── README.md
```

---

## Setup

```bash
git clone <your-repo-url>
cd course-rag
pip install -r requirements.txt
```

Create a `.env` file with your free Groq API key (get one at console.groq.com):
```
GROQ_API_KEY=your_key_here
```

Add your own PDFs to `docs/`, then build the vector store:
```bash
python ingest.py
```

Run the app:
```bash
streamlit run app.py
```

---

## Author

**Riya Suneesh**
M.Sc. Digital Engineering — Otto-von-Guericke University Magdeburg
[LinkedIn](https://www.linkedin.com/in/riya-suneesh-6095931a2) · [GitHub](https://github.com/Riya14486)