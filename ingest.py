"""
Ingestion pipeline: load PDFs, strip references sections (so citation
text never pollutes the vector store), chunk, embed, and save to Chroma.
"""
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DOCS_FOLDER = "docs"
CHROMA_DIR = "./chroma_db"


def strip_references_section(pages):
    """
    Once we hit a page whose text starts with a 'References' heading,
    drop that page and everything after it for this document.
    This prevents citation-list text from ever being embedded.
    """
    kept_pages = []
    references_found = False

    for page in pages:
        text = page.page_content

        heading_match = re.search(r"\bREFERENCES\b", text[:300], re.IGNORECASE)

        if heading_match and not references_found:
            cutoff = heading_match.start()
            if cutoff > 50:
                trimmed_page = page
                trimmed_page.page_content = text[:cutoff]
                kept_pages.append(trimmed_page)
            references_found = True
            continue

        if references_found:
            continue

        kept_pages.append(page)

    return kept_pages


print("Loading PDFs...")
all_documents = []

for filename in os.listdir(DOCS_FOLDER):
    if filename.endswith(".pdf"):
        filepath = os.path.join(DOCS_FOLDER, filename)
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        original_count = len(pages)

        pages = strip_references_section(pages)

        print(f"  {filename}: {original_count} pages loaded, {len(pages)} kept after stripping references")
        all_documents.extend(pages)

print(f"\nTotal pages kept: {len(all_documents)}")

# Wider chunks and overlap for better context continuity
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
chunks = splitter.split_documents(all_documents)
print(f"Created {len(chunks)} chunks (chunk_size=700, overlap=100)")

print("\nLoading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Building vector store... this takes 1-2 minutes")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)

print(f"\nVector store rebuilt and saved to {CHROMA_DIR}")
print(f"Total chunks stored: {len(chunks)}")