# Evaluation

The fixed public corpus is the Energy Literacy DOCX and the 2022 Energy Saver PDF listed in `data/corpus/README.md`. The 25 question set in `evals/golden_questions.jsonl` has 20 answerable questions drafted from extracted source text and five questions intended to be unanswerable from this corpus. Each answerable question has a labelled PDF page or DOCX source block. Labels and reference answers still need independent review.

Run the local comparison with:

```powershell
$env:HF_HUB_OFFLINE='1'
.venv\Scripts\python.exe -m evals.run_evaluation
```

This run uses the BGE embedding model and cosine similarity locally. It does not call Pinecone or Gemini. Raw top-five rankings for A, B, and the previous default are saved in `evals/results/`.

| Configuration | Tokens / overlap | Chunks | Source Hit@5 | Source Recall@5 | Source MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 320 / 45 | 167 | 0.90 | 0.90 | 0.725 |
| B | 450 / 65 | 126 | 0.85 | 0.85 | 0.767 |
| Previous default | 512 / 75 | 114 | 0.80 | 0.80 | 0.717 |

These are source-location measures on two distinct documents. A hit requires the labelled page or source block to be included in a retrieved chunk. Because each question currently has one labelled location, Recall@5 equals Hit@5; additional relevant locations may be missed. A finds one more source within the top five than B and two more than the previous default. B ranks its first relevant source slightly higher than A on average. We chose A for new uploads because finding the source is the first requirement for a grounded answer. Documents already indexed with 512 / 75 retain their old vectors until reindexed. The configurations have not been run through Pinecone, and answer correctness, faithfulness, abstention quality, latency, and cost have not been measured.

For the A ranking, all 20 answerable questions had a highest cosine score above 0.58; four of five unanswerable questions had a highest score below 0.58. The fifth unanswerable question, about a website password, scored above that cutoff and still needs answer-level refusal. We therefore use 0.58 as a provisional low-evidence cutoff. This is calibrated on the same small set used to report it and is not an independent validation result; a larger held-out set may require changing it.

Both settings missed the labelled passage for the question about the heating/cooling share and for the heat-pump filter schedule. B also missed the labelled whole-home systems passage. The first two misses suggest competition from nearby or similarly worded passages; inspecting the raw rankings and extracted text is necessary before attributing a specific cause.

For a running service with both evaluation documents indexed, run `.venv\Scripts\python.exe -m evals.run_answer_evaluation --api-url http://127.0.0.1:8000`. It checkpoints every response in `evals/results/answer_evaluation.jsonl` and writes citation, abstention, and reference-token-overlap checks to `answer_summary.json`. A repeat run resumes successful rows and retries failed rows. Three consecutive API failures pause the run, preserving partial evidence; the summary records failed and unattempted counts and sets `complete` only when all 25 calls succeed. Token overlap is only a proxy for correctness. The `manual_faithfulness` and `manual_correctness` fields remain `null` until a human reviews each answer against its retrieved chunks and source reference. This answer evaluation has not been executed successfully yet, so no answer-quality numbers are claimed.

Live smoke checks completed separately: the Energy Literacy DOCX indexed into 29 chunks, the Energy Literacy PDF into 37 chunks, and the Energy Saver PDF into 85 chunks. A question about energy literacy returned a grounded answer with two citations from five retrieved chunks. The test PDF deletion returned 204, its subsequent lookup returned 404, and a filtered Pinecone query returned zero remaining matches. On 2026-11-26, a question about a website password returned HTTP 200 with abstention, no citations, and no retrieved chunks. An answerable question returned HTTP 503 from the application after answer generation failed; the 25-question live run was stopped after repeated 503 responses, so answer-quality metrics are still unavailable. A one-call check of another free Gemini model was not permitted by the execution environment and was not performed.
