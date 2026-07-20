"""Evaluate retrieval, refusal behavior, sources, answer terms, and latency."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import NO_INFORMATION_ANSWER, ask
from src.retriever import find_relevant_chunks


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CASES = os.path.join(BASE_DIR, "evaluation", "questions.json")


def evaluate_case(case: dict, mode: str) -> dict:
    started = time.perf_counter()
    expected_source = case["expected_source"]

    if mode == "full":
        result = ask(case["question"])
        answer = result["answer"]
        chunks = result["chunks"]
        sources = result["sources"]
    else:
        chunks = find_relevant_chunks(case["question"])
        sources = list(dict.fromkeys(chunk["source"] for chunk in chunks))
        answer = NO_INFORMATION_ANSWER if not chunks else ""

    source_ok = not sources if expected_source is None else expected_source in sources
    refusal_ok = expected_source is not None or answer == NO_INFORMATION_ANSWER
    expected_terms = [term.lower() for term in case.get("expected_any", [])]
    terms_ok = (
        mode != "full"
        or not expected_terms
        or any(term in answer.lower() for term in expected_terms)
    )

    return {
        "id": case["id"],
        "question": case["question"],
        "passed": source_ok and refusal_ok and terms_ok,
        "source_ok": source_ok,
        "refusal_ok": refusal_ok,
        "terms_ok": terms_ok,
        "expected_source": expected_source,
        "sources": sources,
        "top_score": round(max((chunk["score"] for chunk in chunks), default=0.0), 4),
        "answer": answer if mode == "full" else None,
        "latency_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("retrieval", "full"), default="retrieval")
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output")
    parser.add_argument(
        "--resume",
        help="Reuse passing results from a previous report when the question is unchanged.",
    )
    args = parser.parse_args()

    with open(args.cases, "r", encoding="utf-8") as file:
        cases = json.load(file)
    if args.limit is not None:
        cases = cases[: args.limit]

    cached_results = {}
    if args.resume and os.path.exists(args.resume):
        with open(args.resume, "r", encoding="utf-8") as file:
            previous = json.load(file)
        cached_results = {
            result["id"]: result
            for result in previous.get("results", [])
            if result.get("passed")
        }

    results = []
    for index, case in enumerate(cases, start=1):
        cached = cached_results.get(case["id"])
        if cached and cached.get("question") == case["question"]:
            result = cached
            result_state = "CACHED"
        else:
            result = evaluate_case(case, args.mode)
            result_state = "PASS" if result["passed"] else "FAIL"
        results.append(result)
        print(
            f"[{index:02}/{len(cases):02}] {result_state} "
            f"{case['id']} ({result['latency_seconds']:.2f}s)"
        )

    passed = sum(result["passed"] for result in results)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "passed": passed,
        "total": len(results),
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "average_latency_seconds": round(
            sum(result["latency_seconds"] for result in results) / len(results), 3
            if results
            else 0.0,
        ),
        "results": results,
    }

    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)
        print(f"Report: {output_path}")

    print(f"Result: {passed}/{len(results)} ({report['pass_rate']:.1%})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
