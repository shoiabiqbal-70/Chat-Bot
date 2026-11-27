# Design decisions

The backend is a modular FastAPI service. Neon PostgreSQL holds canonical document and chunk records; Pinecone holds 384-dimensional cosine vectors keyed by Neon chunk IDs. Original files are stored locally under `data/uploads`. Upload registration is followed by a separate process call; the browser performs both. This split leaves a `pending` record if indexing is interrupted, which can be retried or diagnosed.

PDF, DOCX, and UTF-8 TXT extraction preserve page or section metadata. Chunking uses the BGE tokenizer and prevents overlap across page or section boundaries. The current default for new uploads is 320 tokens with 45 overlap, selected for the highest source Hit@5 on the fixed local comparison. Previously indexed documents retain their earlier vectors until reindexed. Local BGE-small-en-v1.5 embeddings are normalized; query text receives the model's retrieval instruction.

The answer service resolves Pinecone matches to Neon rows, drops missing or non-completed documents, and sends only retrieved text to Groq. It requires every returned citation ID to be among the retrieved chunk IDs. Uncited or explicitly abstained responses become a fixed abstention message. Groq uses the openai/gpt-oss-20b model by default. Temporary 502, 503, and 504 responses get two bounded retries. The Groq configuration has not yet been live tested.

Cross-store deletion marks a document as `deleting`, removes Pinecone vectors, then removes Neon records and the original file. This ordering avoids reporting successful deletion while known vectors remain. There is no distributed transaction between Neon and Pinecone, so a failed step may leave a `deleting` record for recovery. Retry and reconciliation tooling still need to be built and tested.

The browser uses plain HTML, CSS, and JavaScript served by FastAPI. It shows upload and indexing status, document records, answers, cited passages, source metadata, and cosine scores. No authentication is implemented; deployment must account for access control and durable original-file storage.

## Choices and trade-offs

| Area | Selected | Alternative considered | Why and trade-off |
| --- | --- | --- | --- |
| Corpus | Two distinct public DOE sources, PDF and DOCX | Private company files | Public files are reproducible and safe to share; two documents are too small to establish broad retrieval quality. The second PDF is the same Energy Literacy content in another format and is used only for ingestion checks. |
| Extraction | `pypdf` and `python-docx` | OCR or PyMuPDF | Both run locally without paid APIs and preserve basic page or paragraph structure. Scanned PDFs need OCR and some Energy Saver fonts produce garbled characters; tables and reading order still need manual inspection. |
| Chunking | Token-aware, page or section bounded, currently 320 / 45 | Character splitting or zero overlap | Token bounds match the embedder better and overlap carries context. More chunks raise embedding work and vector usage. The local comparison found higher source Hit@5 than 450 / 65 and the former 512 / 75 default on this corpus. |
| Embeddings | Local BGE-small-en-v1.5, 384 dimensions | Paid embedding API or multilingual BGE-M3 | Local inference avoids per-query API charges and keeps source text on the machine until generation. CPU startup and inference add latency; English focus limits multilingual retrieval. |
| Vector store | Pinecone serverless dense index, cosine metric | Local FAISS | Pinecone persists vectors separately from the API host and supports metadata filtering. It adds remote latency, quotas, and cross-store deletion risk; FAISS would require a persistent host volume and its own concurrency strategy. |
| Retrieval | Top five cosine matches, a provisional 0.58 cutoff, Neon lookup, completed-document filter | Hybrid search or reranking | The simple method is easy to inspect and inexpensive. It can miss exact terms; the cutoff was selected on a small set and requires held-out validation. |
| LLM | Groq openai/gpt-oss-20b | Gemini or local LLM | Groq provides JSON object output through its chat completions API. Provider quotas and data-use terms should be checked for the chosen account. A local LLM avoids external transfer but needs more host memory and compute. |
| Prompt | Retrieved text only, strict citation IDs, abstention instruction | Unrestricted generation | Restricting context and validating IDs reduces fabricated citations. It cannot prove every sentence is faithful; answer evaluation remains required. |
| Tests | pytest unit/API tests plus separate live smoke checks | Live provider calls in every test | Offline tests are fast and reproducible. Live tests catch provider configuration and timing faults but require credentials and can be flaky. |

The API uses `POST /api/documents` for upload and indexing with 201 only after successful ingestion. `GET /api/documents` lists records, `GET /api/documents/{id}` reads one, `DELETE /api/documents/{id}` removes it, and `POST /api/ask` returns answer and evidence. Registration and processing routes remain for compatibility and recovery.
