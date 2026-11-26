# AI use record

Codex assisted with repository review, implementation planning, backend and frontend code, unit tests, public corpus selection, local evaluation tooling, and documentation on 2026-11-25. The project owner supplied provider configuration and approved use of a public corpus and a free Pinecone Starter index.

Human review remains required for the golden question labels, source page or chunk relevance, answer quality, deployment settings, and the final demonstration. Generated code was checked with pytest and JavaScript syntax validation; selected provider flows were checked with live smoke requests. Evaluation numbers in `EVAL.md` are from the executed local runner and are explicitly limited to document-level retrieval.

The available project prompts asked Codex to read the repository and blueprint, implement the portfolio project, use free sources, avoid repeated questions, and continue from the eight-page portfolio brief. This summary is not a complete exported chat transcript; the portfolio brief requests the full prompt list and chat history as separate submission evidence.

No API key values are included in tracked project files. Keys shared in chat should be rotated before deployment.
