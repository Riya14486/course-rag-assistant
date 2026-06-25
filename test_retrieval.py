from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load the existing vector store (no need to rebuild)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Ask a test question
question = "What problems exist with offline evaluation methods for recommenders?"
results = vectorstore.similarity_search(question, k=6)

print(f"Question: {question}\n")
print(f"Top {len(results)} matching chunks:\n")
for i, doc in enumerate(results):
    print(f"--- Result {i+1} (from {doc.metadata['source']}, page {doc.metadata['page']}) ---")
    print(doc.page_content[:300])
    print()