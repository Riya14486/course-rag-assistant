from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

docs_folder = "docs"
all_documents = []

print("Loading PDFs...")
for filename in os.listdir(docs_folder):
    if filename.endswith(".pdf"):
        filepath = os.path.join(docs_folder, filename)
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        all_documents.extend(pages)

print(f"Loaded {len(all_documents)} pages")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_documents)
print(f"Created {len(chunks)} chunks")

# Create embeddings (this downloads a free model first time, ~90MB)
print("\nLoading embedding model (downloads once)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build and save the vector store
print("Building vector store... this takes 1-2 minutes")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("\nVector store built and saved to ./chroma_db ✅")
print(f"Total chunks stored: {len(chunks)}")