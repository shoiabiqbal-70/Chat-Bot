# Project status and completion plan

Last reviewed on 2026-11-26 against the eight-page portfolio brief, `Docs/production_rag_service_implementation_blueprint.html`, and the current source tree.

## Verified baseline

- FastAPI has health, document upload, list, read, process, delete, and ask routes. Upload validation, SHA-256 duplicate detection, original-file storage, and metadata registration are implemented.
- PDF, DOCX, and TXT extractors, token-aware chunking, an embedding wrapper, SQLAlchemy models and repositories, and the initial Alembic migration exist as separate components.
- The primary upload route now returns 201 after extraction, chunking, embeddings, Pinecone upsert, and Neon chunk persistence. The older registration and process routes remain for compatibility and retry workflows. New uploads use the evaluated 320/45 chunk setting; existing documents can use the reindex route. The underlying path passed live smoke checks for one DOCX and two PDFs under the earlier setting; the new single-call route has an API test with mocked providers.
- Retrieval, Groq answer generation, citation-ID validation, and a browser interface are implemented. The earlier Gemini configuration passed live DOCX and PDF indexing, one grounded answer, PDF deletion, and a low-evidence abstention. A later answerable query returned 503. Groq has passed mocked tests but has not been live tested; full answer evaluation and deployment remain missing.
- `.env` is ignored by Git and is not tracked. The working tree was clean at the start of this audit.
- Local validation: `350 passed, 19 skipped` using the README test command. JavaScript syntax passed and `/app`, `/static/app.js`, and `/openapi.json` returned 200. Four notebook files were generated; the first three have saved outputs from the local Python runner, while the live answer notebook remains unexecuted. The source-location comparison of A, B, and the previous default was run and saved under `evals/results/`. The skipped tests include the opt-in live Neon suite. A separate live smoke test succeeded for upload, indexing, one answer, and deletion; deployment and an interactive browser flow remain unverified.

## Completion sequence and gates

1. **Configuration and baseline:** use `APP_DEBUG` to avoid host `DEBUG` conflicts, verify a fresh install, and run migrations on an isolated database. Gate: imports and local tests pass without host-specific configuration errors.
2. **Corpus and experiments:** select a representative PDF and DOCX corpus, inspect extracted text and metadata, create notebooks, and prepare source-grounded questions. Gate: manually verified sample extraction and recorded corpus inventory.
3. **Ingestion:** connect upload registration to extraction, chunking, embedding, Neon chunk persistence, and Pinecone upsert; track processing failures; implement list and coordinated delete. Gate: both formats upload and become searchable, duplicate yields 409, and deletion removes all vectors.
4. **Retrieval and generation:** embed queries, search Pinecone, load matching Neon chunks, call Groq with bounded context, validate every citation against retrieved IDs, and abstain when evidence is insufficient. Gate: answerable and unanswerable questions behave as specified.
5. **Frontend:** build upload, library, question, answer, citation, and retrieval-inspector views. Gate: full user journey works through the UI.
6. **Reliability and security:** add provider timeouts, retries, logging, input bounds, injection-resistant prompts, failure recovery, and tests for external failures. Gate: offline tests pass and live failure behavior is demonstrated.
7. **Evaluation:** run at least 25 golden questions, including five unanswerable questions, compare two chunk configurations, and record retrieval/answer metrics and failures in `EVAL.md`; document design choices in `DESIGN.md`. Gate: reproducible measured result and justified configuration.
8. **Submission:** pin compatible dependencies, finish setup and API docs, add screenshots and AI-use record, deploy with persistent storage, and record the demo. Gate: fresh setup and deployed workflow are verified.

The highest-priority remaining gates are full live answer-quality evaluation, browser interaction checks, durable hosted storage, and submission evidence. A Dockerfile and deployment assessment are in `DEPLOYMENT.md`, but no public host has been configured or validated. The user has no Oracle Cloud account; a free temporary demo route using a Cloudflare Quick Tunnel is documented, but it is not durable hosting. The existing provider settings were sufficient for earlier live smoke checks, but the two keys pasted into chat should be rotated before any deployment.
