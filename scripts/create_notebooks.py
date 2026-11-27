"""Create the four reproducible portfolio demonstration notebooks."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


def save(name: str, cells: list[dict]) -> None:
    notebook = {"cells": cells, "metadata": {"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5}
    (OUT / name).write_text(json.dumps(notebook, indent=2), encoding="utf-8")


setup = """from pathlib import Path
import os
from uuid import uuid4
ROOT = Path.cwd().resolve()
if not (ROOT / 'app').exists():
    ROOT = ROOT.parent
os.chdir(ROOT)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
"""

save("01_extraction.ipynb", [
    markdown("# 1. Extract PDF and DOCX\nRun from the repository or `notebooks/` directory. The corpus is public DOE material. Inspect the printed source metadata and text before grading questions.\n"),
    code(setup),
    code("""from app.ingestion.dispatcher import extract_document
for path in (ROOT / 'data' / 'corpus').glob('*'):
    if path.suffix not in {'.pdf', '.docx'}:
        continue
    document = extract_document(path, uuid4(), path.name)
    print(path.name, 'blocks=', len(document.blocks), 'characters=', len(document.full_text))
    for block in document.blocks[:3]:
        print('  block', block.block_index, 'page', block.page_number,
              'section', block.section, repr(block.text[:160]))
"""),
])

save("02_chunking.ipynb", [
    markdown("# 2. Inspect tokenizer-aware chunks\nCompare two chunk sizes and inspect page or section provenance.\n"),
    code(setup),
    code("""from app.ingestion.dispatcher import extract_document
from app.ingestion.chunking import chunk_document
from app.ingestion.tokenizer import get_tokenizer
path = ROOT / 'data/corpus/Energy_Literacy_text_only_3.0.docx'
document = extract_document(path, uuid4(), path.name)
tokenizer = get_tokenizer()
for size, overlap in ((320, 45), (450, 65)):
    chunks = chunk_document(document, tokenizer, chunk_size=size, overlap=overlap)
    print(size, overlap, 'chunks=', len(chunks))
    for chunk in chunks[:2]:
        print(chunk.chunk_index, chunk.token_count,
              chunk.source_block_indices, repr(chunk.text[:160]))
"""),
])

save("03_embeddings_retrieval.ipynb", [
    markdown("# 3. Local embedding and retrieval comparison\nThis notebook uses local BGE embeddings and saves measured rankings without provider keys. The comparison runner uses 25 fixed questions.\n"),
    code(setup),
    code("""from app.ingestion.embedding import get_embedding_service
embedding = get_embedding_service()
query = embedding.embed_query('What is energy literacy?')
print('dimension=', query.shape[0], 'norm=', round(float((query @ query) ** 0.5), 5))
"""),
    code("""from evals.run_evaluation import main
main()
print((ROOT / 'evals/results/summary.json').read_text())
"""),
])

save("04_answering.ipynb", [
    markdown("# 4. API answer and evidence inspection\nRequires configured Neon, Pinecone, and Groq services, a migrated database, and indexed documents. Do not print API keys or save private answers in notebook outputs.\n"),
    code(setup),
    code("""from fastapi.testclient import TestClient
from app.main import app
with TestClient(app) as client:
    response = client.post('/api/ask', json={'question': 'What is energy literacy?'})
    print('status=', response.status_code)
    if response.status_code == 200:
        result = response.json()
        print('answer=', result['answer'])
        print('citations=', result['citations'])
        print('retrieved=', result['chunk_count'])
        for chunk in result['chunks']:
            print(chunk['filename'], chunk['page_number'], chunk['similarity_score'])
"""),
])
