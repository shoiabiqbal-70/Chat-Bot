"""Run the fixed question set through a running API and save answer checks."""

import json
import re
import argparse
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "evals" / "golden_questions.jsonl"
RESULTS = ROOT / "evals" / "results"


def words(value: str) -> set[str]:
    """Normalize text for a limited reference overlap proxy."""
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def reference_f1(reference: str, answer: str) -> float:
    """Return lexical F1; this is not a semantic correctness judgment."""
    # Word overlap is a quick proxy; the saved rows support later manual review.
    expected, actual = words(reference), words(answer)
    if not expected or not actual:
        return 0.0
    shared = len(expected & actual)
    precision = shared / len(actual)
    recall = shared / len(expected)
    return 2 * precision * recall / (precision + recall) if shared else 0.0


def summarize(questions: list[dict], rows: dict[str, dict]) -> dict:
    """Keep incomplete runs explicit so partial scores are not mistaken for final ones."""
    successful = [row for row in rows.values() if "error" not in row]
    answered = [row for row in successful if row["answerable"]]
    unanswerable = [row for row in successful if not row["answerable"]]
    return {
        "question_count": len(questions),
        "successful_calls": len(successful),
        "failed_calls": sum("error" in row for row in rows.values()),
        "not_attempted": len(questions) - len(rows),
        "complete": len(successful) == len(questions),
        "citation_valid_rate": (
            sum(row["citations_valid"] for row in successful) / len(successful)
            if successful else None
        ),
        "unanswerable_accuracy": (
            sum(row["unanswerable_handled"] for row in unanswerable) / len(unanswerable)
            if unanswerable else None
        ),
        "unanswerable_coverage": len(unanswerable),
        "mean_reference_token_f1": (
            sum(row["reference_token_f1"] for row in answered) / len(answered)
            if answered else None
        ),
    }


def main(api_url: str = "http://127.0.0.1:8000",
         max_consecutive_errors: int = 3) -> dict:
    """Checkpoint every call, resume successes, and pause on repeated failures."""
    questions = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines()]
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "answer_evaluation.jsonl"
    existing = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            existing[row["question"]] = row
    rows = {item["question"]: existing[item["question"]]
            for item in questions if item["question"] in existing}
    # Preserve prior successes so a failed provider call does not restart the run.

    def save() -> dict:
        path.write_text("".join(json.dumps(rows[item["question"]]) + "\n"
                                for item in questions if item["question"] in rows),
                        encoding="utf-8")
        summary = summarize(questions, rows)
        (RESULTS / "answer_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    consecutive_errors = 0
    with httpx.Client(base_url=api_url, timeout=90) as client:
        for number, item in enumerate(questions, 1):
            if item["question"] in rows and "error" not in rows[item["question"]]:
                continue
            try:
                response = client.post("/api/ask", json={"question": item["question"]})
                response.raise_for_status()
                data = response.json()
                retrieved = {chunk["chunk_id"] for chunk in data["chunks"]}
                # A citation is valid only if its chunk appears in this response.
                row = {"question": item["question"], "answerable": item["answerable"],
                       "answer": data["answer"], "abstained": data["abstained"],
                       "citations": data["citations"],
                       "citations_valid": all(c in retrieved for c in data["citations"]),
                       "unanswerable_handled": (data["abstained"] and not data["citations"])
                           if not item["answerable"] else None,
                       "reference_token_f1": reference_f1(
                           item["reference_answer"], data["answer"])
                           if item["answerable"] else None,
                       "retrieved_chunks": data["chunks"],
                       "manual_faithfulness": None, "manual_correctness": None}
                consecutive_errors = 0
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                row = {"question": item["question"], "answerable": item["answerable"],
                       "error": type(exc).__name__}
                if isinstance(exc, httpx.HTTPStatusError):
                    row["status_code"] = exc.response.status_code
                consecutive_errors += 1
            rows[item["question"]] = row
            save()
            print(f"{number}/{len(questions)}: " +
                  (f"HTTP {row.get('status_code', row['error'])}" if "error" in row
                   else "answered"), flush=True)
            if consecutive_errors >= max_consecutive_errors:
                print("Paused after repeated API failures; rerun to resume.", flush=True)
                break
    summary = save()
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-consecutive-errors", type=int, default=3)
    args = parser.parse_args()
    if args.max_consecutive_errors < 1:
        parser.error("--max-consecutive-errors must be at least 1")
    main(args.api_url, args.max_consecutive_errors)
