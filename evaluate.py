"""
Evaluation script for the RAG assistant — v2.
Measures:
  1. Retrieval accuracy - did the correct source file get retrieved?
  2. Answer quality - semantic similarity between generated answer and
     expected keywords/reference, instead of strict substring matching.
     This avoids penalizing correct answers that are just phrased differently.
"""
import json
from rag_core import RagEngine
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

with open("qa_set.json", "r") as f:
    qa_set = json.load(f)

print(f"Loaded {len(qa_set)} test questions\n")
print("Loading RAG engine (takes a minute)...\n")
engine = RagEngine()

print("Loading semantic similarity model...")
sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

results = []

for i, item in enumerate(qa_set):
    question = item["question"]
    expected_keywords = item.get("expected_answer_contains", [])
    expected_source = item.get("expected_source_file", "")

    print(f"--- Question {i+1}/{len(qa_set)} ---")
    print(f"Q: {question}")

    answer, sources = engine.ask(question)
    print(f"A: {answer}")

    # Retrieval check
    retrieved_files = [s["file"] for s in sources]
    retrieval_hit = False
    if expected_source:
        retrieval_hit = any(expected_source in f for f in retrieved_files)

    # Semantic similarity: compare answer to a "reference sentence" built
    # from the expected keywords, rather than requiring exact substrings
    semantic_score = None
    if expected_keywords:
        reference_text = " ".join(expected_keywords)
        embeddings = sim_model.encode([answer, reference_text])
        semantic_score = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

    # Keep the old strict keyword match too, for comparison
    answer_lower = answer.lower()
    keyword_hits = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    keyword_score = len(keyword_hits) / len(expected_keywords) if expected_keywords else None

    print(f"Retrieval hit: {retrieval_hit}")
    if semantic_score is not None:
        print(f"Semantic similarity: {semantic_score:.2f}  (strict keyword match: {keyword_score:.2f})")
    print()

    results.append({
        "question": question,
        "answer": answer,
        "retrieval_hit": retrieval_hit,
        "semantic_score": semantic_score,
        "keyword_score": keyword_score,
        "has_expected_data": bool(expected_keywords or expected_source)
    })

# ---------- SUMMARY ----------
evaluated = [r for r in results if r["has_expected_data"]]

if evaluated:
    retrieval_scores = [r["retrieval_hit"] for r in evaluated if r["retrieval_hit"] is not None]
    semantic_scores = [r["semantic_score"] for r in evaluated if r["semantic_score"] is not None]
    keyword_scores = [r["keyword_score"] for r in evaluated if r["keyword_score"] is not None]

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    if retrieval_scores:
        retrieval_accuracy = sum(retrieval_scores) / len(retrieval_scores) * 100
        print(f"Retrieval accuracy:       {retrieval_accuracy:.1f}% ({sum(retrieval_scores)}/{len(retrieval_scores)})")
    if semantic_scores:
        avg_semantic = np.mean(semantic_scores) * 100
        print(f"Avg semantic similarity:  {avg_semantic:.1f}%")
    if keyword_scores:
        avg_keyword = sum(keyword_scores) / len(keyword_scores) * 100
        print(f"Avg strict keyword match: {avg_keyword:.1f}%  (for comparison)")
else:
    print("No questions had expected data filled in yet.")

with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nFull results saved to eval_results.json")