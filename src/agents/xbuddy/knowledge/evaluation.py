"""Retrieval evaluation for the learning-resource library.

Each case is a query written the way requirements appear in real postings, with the
library ids a reviewer judged relevant. Negative cases (soft skills, years of
experience) must retrieve nothing. Metrics, at k = 3 (what the prompts show per
requirement is at most 2, so 3 is a fair ceiling):

  hit@k      share of cases with at least one relevant resource in the top k
  recall@k   mean share of a case's relevant resources found in the top k
  mrr        mean reciprocal rank of the first relevant resource
  negatives  share of negative cases that correctly retrieve nothing

Two case files:
  retrieval_eval.json          50 cases written alongside the skill vocabulary
                               (development set; optimistic)
  retrieval_eval_heldout.json  23 requirements copied from three real postings
                               (Thumbtack, Asana, Insight), labeled before the
                               retriever was run on them

Run it:  cd src && python -m agents.xbuddy.knowledge.evaluation [cases.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .retriever import retrieve

DEFAULT_CASES = (
    Path(__file__).resolve().parents[4] / "tests" / "xbuddy" / "fixtures" / "retrieval_eval.json"
)
K = 3


def evaluate(cases: dict, k: int = K) -> dict:
    hits, recalls, ranks, misses = 0, [], [], []
    for case in cases["positive"]:
        relevant = set(case["relevant"])
        got = [h.resource.id for h in retrieve(case["query"], k=k)]
        found = [i for i in got if i in relevant]
        if found:
            hits += 1
        else:
            misses.append({"query": case["query"], "got": got, "relevant": sorted(relevant)})
        recalls.append(len(found) / min(len(relevant), k))
        first = next((n for n, i in enumerate(got, start=1) if i in relevant), None)
        ranks.append(1 / first if first else 0.0)
    false_hits = []
    for query in cases["negative"]:
        got = [h.resource.id for h in retrieve(query, k=k)]
        if got:
            false_hits.append({"query": query, "got": got})
    n = len(cases["positive"])
    return {
        "cases": n,
        "negative_cases": len(cases["negative"]),
        f"hit@{k}": round(hits / n, 3),
        f"recall@{k}": round(sum(recalls) / n, 3),
        "mrr": round(sum(ranks) / n, 3),
        "negatives": round(1 - len(false_hits) / max(len(cases["negative"]), 1), 3),
        "misses": misses,
        "false_hits": false_hits,
    }


def main(path: str | None = None) -> None:
    cases = json.loads(Path(path or DEFAULT_CASES).read_text("utf-8"))
    report = evaluate(cases)
    for key in ("cases", "negative_cases", f"hit@{K}", f"recall@{K}", "mrr", "negatives"):
        print(f"{key:>15}: {report[key]}")
    for miss in report["misses"]:
        print(f"MISS  {miss['query']!r}: got {miss['got']}, wanted one of {miss['relevant']}")
    for fh in report["false_hits"]:
        print(f"FALSE {fh['query']!r}: got {fh['got']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
