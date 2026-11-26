# Demo recording guide

This is a recording checklist, not a completed video or screenshot set. Use only rotated provider keys and public corpus files.

1. Show `data/corpus/README.md`, the layered `app/` tree, and the architecture in `DESIGN.md`.
2. Start the service with the README command. Open `/docs` and `/app`; capture Swagger screens for health, upload, ask, and delete after successful requests.
3. In `/app`, upload `Energy_Literacy_text_only_3.0.docx` and `energy-saver-guide-2022.pdf` if they are not already present. Confirm both show `completed` and a chunk count. If already indexed with the old 512/75 setting, use Reindex first.
4. Ask at least three answerable questions from `evals/golden_questions.jsonl`. For each, show the answer, cited source rows, retrieved chunk count, text, and cosine scores.
5. Ask an unanswerable question. Show a clear abstention with no citations. If Groq returns 503, label the attempt as a provider failure and retry; do not present the failure as a successful guardrail.
6. Show a duplicate upload returning 409, an unsupported extension returning 415, an empty question returning 422, and a prompt-injection question returning 422.
7. Delete one document. Confirm its API lookup returns 404 and a follow-up answer has no citations from it. Show the source-location results and limitations in `EVAL.md`.
8. Run `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp data/pytest_tmp`; set `$env:HF_HUB_OFFLINE='1'` and run `.venv\Scripts\python.exe -m evals.run_evaluation` on screen.

Record the actual screen and outcomes. Do not add fabricated screenshots or edit response statuses into the recording.
