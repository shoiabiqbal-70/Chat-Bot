# Production RAG Service

A document-based retrieval-augmented generation service with a browser workspace, indexed PDF/DOCX uploads, Pinecone retrieval, Neon metadata, and cited Groq answers. Local tests and a two-configuration retrieval evaluation are complete; live answer-quality evaluation and durable public deployment remain outstanding. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for verified evidence and [the implementation blueprint](Docs/production_rag_service_implementation_blueprint.html) for the full requirements.

## Local setup

Use Python 3.12 and PostgreSQL for the document API.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `NEON_DATABASE_URL`, `NEON_MIGRATION_DATABASE_URL`, `PINECONE_API_KEY`, and `GROQ_API_KEY` in `.env`. The Pinecone index must already exist with 384 dimensions and cosine similarity. Use `APP_DEBUG=true` only for local debugging. Keep `.env` private.

Apply the database migration before using document uploads:

```powershell
.venv\Scripts\python.exe -m alembic upgrade head
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

The browser workspace is at `http://127.0.0.1:8000/app` and API documentation at `http://127.0.0.1:8000/docs`. The question form selects the newest completed document by default; choose another file or All documents as needed. Selected-document questions include all extracted passages for documents of up to 12 chunks. Longer documents search a wider set of vector matches and add keyword matches from indexed text. Journal, title, author, DOI, and publisher questions include the opening passage. Figure- and table-count questions scan numbered references across the selected document. These counts describe labels in extracted text, not visually detected objects. Answers require citations to retrieved passages; scanned pages without extractable text and information present only in images are not currently supported. `POST /api/documents` uploads and indexes in one request; it returns 201 with a chunk count only after indexing succeeds. The earlier `POST /api/documents/upload` and `POST /api/documents/{id}/process` routes remain available for registration and retry workflows. Other routes are `GET /api/documents`, `GET /api/documents/{id}`, `DELETE /api/documents/{id}`, and `POST /api/ask`. Original files are stored in `DOCUMENT_STORAGE_DIR` (`data/uploads` by default); production hosting needs a persistent volume for that directory.

## Tests

Run the isolated suite on Windows with a writable pytest temp directory:

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp data/pytest_tmp
```

Live Neon integration tests are opt-in with `RUN_DB_INTEGRATION_TESTS=1` and require a migrated test database. The default suite mocks Pinecone and Groq and does not establish a working provider configuration.

## Evaluation and notebooks

The public sample corpus is documented in [data/corpus/README.md](data/corpus/README.md). Four notebooks under `notebooks/` walk through extraction, chunking, embeddings/retrieval, and answering. The first three use local source files; the answer notebook requires working provider services and indexed documents.

Run the reproducible local comparison of two candidates and the previous default from the repository root:

```powershell
$env:HF_HUB_OFFLINE='1'
.venv\Scripts\python.exe -m evals.run_evaluation
```

Read [EVAL.md](EVAL.md) for measured results and limitations. The model must be cached locally for offline mode; otherwise download it once before setting `HF_HUB_OFFLINE`.

## Deployment

Source Atlas serves the frontend and API from one FastAPI container. `render.yaml` prepares a temporary Render Free demo, while a Docker Compose and Caddy stack is prepared for an Oracle Cloud Always Free VM with persistent uploads and HTTPS. See [DEPLOYMENT.md](DEPLOYMENT.md) for setup and Render's memory and storage limits. Production requires `APP_ACCESS_PASSWORD` to protect the shared document library. No public deployment has been validated yet. Existing documents can be moved to the evaluated chunk setting through `POST /api/documents/{id}/reindex` or the Reindex button in the browser workspace.
