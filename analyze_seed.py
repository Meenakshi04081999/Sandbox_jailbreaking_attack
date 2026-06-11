"""
Analyze how informative the seed pool is for distillation.

Joins seed_meta.jsonl (category + our expected label) with labels_debug.jsonl
(teacher's actual label + score) and reports:

  1. per-category pass/not-pass breakdown
  2. confidence distribution -- how saturated is the teacher?
  3. boundary cases   -- teacher score nearest 0.5 (most informative)
  4. disagreements    -- teacher label != our expected (surprising boundary cases)

A "block confidence" p_block is reported in [0,1] regardless of which raw label
won, so scores are comparable across both classes (0.5 = maximally uncertain).
"""
import json
from pathlib import Path
from collections import defaultdict


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    meta = {d["input"]: d for d in load("seed_meta.jsonl")}
    rows = []
    for d in load("labels_debug.jsonl"):
        m = meta.get(d["input"], {})
        # p_block: probability the teacher BLOCKS, on a common scale.
        p_block = d["score"] if not d["passed"] else 1.0 - d["score"]
        rows.append({**d, "category": m.get("category", "?"),
                     "expected": m.get("expected"), "p_block": p_block})

    # 1. per-category breakdown
    print("=== per-category (passed / total) ===")
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    for cat in sorted(by_cat):
        items = by_cat[cat]
        npass = sum(1 for r in items if r["passed"])
        print(f"  {cat:22s} pass {npass:2d}/{len(items):<2d}")

    # 2. confidence distribution
    print("\n=== teacher confidence (|p_block - 0.5| binned) ===")
    bins = {"saturated (>.45)": 0, "mid (.25-.45)": 0, "near-boundary (<.25)": 0}
    for r in rows:
        d = abs(r["p_block"] - 0.5)
        key = "saturated (>.45)" if d > 0.45 else "mid (.25-.45)" if d >= 0.25 else "near-boundary (<.25)"
        bins[key] += 1
    for k, v in bins.items():
        print(f"  {k:24s} {v:3d}  {'#' * v}")

    # 3. boundary cases
    print("\n=== most informative (p_block closest to 0.5) ===")
    for r in sorted(rows, key=lambda r: abs(r["p_block"] - 0.5))[:10]:
        print(f"  p_block={r['p_block']:.3f}  [{r['category']}]  {r['input'][:55]}")

    # 4. disagreements with our expectation
    print("\n=== teacher disagrees with our expected label ===")
    disagree = [r for r in rows if r["expected"] is not None and r["passed"] != r["expected"]]
    if not disagree:
        print("  (none)")
    for r in disagree:
        exp = "pass" if r["expected"] else "block"
        got = "pass" if r["passed"] else "block"
        print(f"  expected {exp:5s} got {got:5s} (p_block={r['p_block']:.2f})  [{r['category']}]  {r['input'][:50]}")
    print(f"\n  total disagreements: {len(disagree)}/{len(rows)}")


if __name__ == "__main__":
    main()
