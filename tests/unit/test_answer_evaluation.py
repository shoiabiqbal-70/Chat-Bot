"""Mechanical answer evaluation metric checks."""

import json

import httpx

from evals import run_answer_evaluation as evaluation


reference_f1 = evaluation.reference_f1


def test_reference_f1_exact_match():
    assert reference_f1("Energy is conserved", "Energy is conserved") == 1.0


def test_reference_f1_no_match():
    assert reference_f1("energy conservation", "football match") == 0.0


def test_repeated_failures_checkpoint_and_resume(monkeypatch, tmp_path):
    questions = [
        {"question": f"Question {i}", "answerable": True,
         "reference_answer": "energy"} for i in range(3)
    ]
    source = tmp_path / "questions.jsonl"
    source.write_text("".join(json.dumps(item) + "\n" for item in questions),
                      encoding="utf-8")
    monkeypatch.setattr(evaluation, "QUESTIONS", source)
    monkeypatch.setattr(evaluation, "RESULTS", tmp_path)

    class FailedClient:
        def __init__(self, **kwargs):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json):
            self.calls += 1
            request = httpx.Request("POST", "http://example.test/api/ask")
            return httpx.Response(503, request=request)

    monkeypatch.setattr(evaluation.httpx, "Client", FailedClient)
    first = evaluation.main(max_consecutive_errors=2)
    assert first["failed_calls"] == 2
    assert first["not_attempted"] == 1
    assert first["complete"] is False
    saved = (tmp_path / "answer_evaluation.jsonl").read_text(encoding="utf-8")
    assert len(saved.splitlines()) == 2
    assert all(row["status_code"] == 503 for row in map(json.loads, saved.splitlines()))

    class WorkingClient(FailedClient):
        def post(self, url, json):
            request = httpx.Request("POST", "http://example.test/api/ask")
            return httpx.Response(200, request=request, json={
                "answer": "energy", "abstained": False,
                "citations": ["one"], "chunks": [{"chunk_id": "one"}],
            })

    monkeypatch.setattr(evaluation.httpx, "Client", WorkingClient)
    second = evaluation.main()
    assert second["successful_calls"] == 3
    assert second["complete"] is True
    assert second["citation_valid_rate"] == 1.0
