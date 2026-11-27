"""Compare two local chunking configurations on the fixed public corpus."""

import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from app.ingestion.chunking import chunk_document
from app.ingestion.dispatcher import extract_document
from app.ingestion.embedding import get_embedding_service
from app.ingestion.tokenizer import get_tokenizer
from app.config import settings


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
GOLDEN = ROOT / "evals" / "golden_questions.jsonl"
RESULTS = ROOT / "evals" / "results"
CONFIGS = {"A": (320, 45), "B": (450, 65), "current": (512, 75)}
# Each tuple is (tokens per chunk, overlapping tokens) for comparison.


def main() -> None:
    """Write source-location retrieval metrics and raw rankings."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    questions = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]
    if len(questions) < 25 or sum(not q["answerable"] for q in questions) < 5:
        raise ValueError("Golden set must have 25 questions and five unanswerable cases.")
    tokenizer = get_tokenizer()
    embedding = get_embedding_service()
    sources = [p for p in CORPUS.iterdir() if p.suffix.lower() in {".pdf", ".docx"}
               and p.name in {"Energy_Literacy_text_only_3.0.docx", "energy-saver-guide-2022.pdf"}]
    if len(sources) != 2:
        raise ValueError("Evaluation corpus is incomplete.")
    extracted = [(p.name, extract_document(p, uuid4(), p.name)) for p in sources]
    query_vectors = np.stack([embedding.embed_query(q["question"]) for q in questions])
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, (size, overlap) in CONFIGS.items():
        chunks = [(filename, chunk) for filename, document in extracted
                  for chunk in chunk_document(document, tokenizer,
                                              chunk_size=size, overlap=overlap)]
        vectors = embedding.embed_documents([part.text for _, part in chunks])
        rankings = []
        hits = []
        reciprocal_ranks = []
        answerable_above_cutoff = []
        unanswerable_below_cutoff = []
        for question, query in zip(questions, query_vectors):
            # Normalized embeddings make a dot product a cosine similarity.
            scores = vectors @ query
            order = np.argsort(-scores)[:5]
            top = [{"source": chunks[i][0], "page": chunks[i][1].page_number,
                    "chunk_index": chunks[i][1].chunk_index,
                    "source_block_indices": chunks[i][1].source_block_indices,
                    "score": float(scores[i])} for i in order]
            rankings.append({"question": question["question"], "top_k": top})
            if question["answerable"]:
                answerable_above_cutoff.append(
                    int(top[0]["score"] >= settings.min_similarity_score)
                )
                rank = next((i + 1 for i, item in enumerate(top)
                             # Score a hit only when the expected source
                             # location appears among the first five results.
                             if item["source"] == question["source"] and (
                                 item["page"] == question.get("source_page")
                                 if "source_page" in question else
                                 question["source_block_index"] in item["source_block_indices"]
                             )), None)
                hits.append(int(rank is not None))
                reciprocal_ranks.append(1 / rank if rank else 0)
            else:
                unanswerable_below_cutoff.append(
                    int(top[0]["score"] < settings.min_similarity_score)
                )
        summary[name] = {"chunk_size": size, "overlap": overlap,
                         "chunk_count": len(chunks), "evaluated_answerable": len(hits),
                         "source_hit_at_5": sum(hits) / len(hits),
                         "source_recall_at_5": sum(hits) / len(hits),
                         "source_mrr_at_5": sum(reciprocal_ranks) / len(hits),
                         "answerable_above_cutoff": sum(answerable_above_cutoff),
                         "unanswerable_below_cutoff": sum(unanswerable_below_cutoff),
                         "cutoff": settings.min_similarity_score}
        (RESULTS / f"retrieval_{name}.json").write_text(
            json.dumps(rankings, indent=2), encoding="utf-8")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
