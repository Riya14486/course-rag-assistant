from rag_core import RagEngine

engine = RagEngine()

question = "What problems exist with offline evaluation methods for recommenders?"
answer, sources = engine.ask(question)

print("\n--- ANSWER ---")
print(answer)

print("\n--- SOURCES ---")
for s in sources:
    print(f"- {s['file']}, page {s['page']}")