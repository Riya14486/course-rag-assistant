from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

# Load vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Retrieve relevant chunks
question = "What problems exist with offline evaluation methods for recommenders?"
results = vectorstore.similarity_search(question, k=8)


def looks_like_reference_or_title(text):
    """Filter out citation lists, page headers, and author/affiliation blocks."""
    signals = [
        "doi.org" in text,
        "https://" in text[:80],
        bool(re.search(r"\[\d+\]", text)),          # [1], [23] style citations
        "@acm.org" in text or "@dcc" in text,         # email addresses
        text.count(",") > 6 and "University" in text,  # author/affiliation blocks
    ]
    return any(signals)


body_chunks = [doc for doc in results if not looks_like_reference_or_title(doc.page_content)]
context = "\n\n".join([doc.page_content for doc in body_chunks[:3]])

print(f"Filtered {len(results)} retrieved chunks down to {len(body_chunks)} body-text chunks\n")

# Load a stronger free local LLM
print("Loading flan-t5-large (downloads ~3GB first time, please wait)...")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

prompt = f"""Read the text below and answer the question in your own words, in 2-3 sentences. Do not copy sentences directly from the text.

Text:
{context}

Question: {question}

Answer:"""

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
outputs = model.generate(
    **inputs,
    max_length=200,
    num_beams=4,
    repetition_penalty=2.0
)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n--- ANSWER ---")
print(answer)

print("\n--- SOURCES USED ---")
for doc in body_chunks[:3]:
    print(f"- {doc.metadata['source']}, page {doc.metadata['page']}")