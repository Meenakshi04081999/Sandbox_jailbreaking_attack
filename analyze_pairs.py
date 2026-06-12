"""
Measure the minimal-pair flip rate to test the lexical-trigger hypothesis.

Joins pairs_meta.jsonl with the teacher labels and, for each pair, compares the
trigger variant vs the neutral variant. A pair "flips" when the trigger variant
is blocked AND the neutral variant passes -- i.e. swapping only the verb changed
the decision while intent stayed fixed. A high flip rate => teacher is lexical.

Run after:
  python build_pairs.py
  python collect_labels.py pairs_prompts.txt --out pairs_labels.jsonl
"""
import json
from pathlib import Path
from collections import defaultdict


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    meta = {d["input"]: d for d in load("pairs_meta.jsonl")}
    label = {}
    for d in load("pairs_labels_debug.jsonl"):
        p_block = d["score"] if not d["passed"] else 1.0 - d["score"]
        label[d["input"]] = {"passed": d["passed"], "p_block": p_block}

    pairs = defaultdict(dict)
    for text, m in meta.items():
        pairs[(m["pair_id"], m["intent"])][m["variant"]] = {**label[text], "text": text}

    flips, intent_flips = 0, defaultdict(lambda: [0, 0])
    print(f"{'pair':16s} {'intent':9s} {'trig p_block':>12s} {'neut p_block':>12s}  flip?")
    print("-" * 64)
    for (pid, intent), v in pairs.items():
        t, n = v["trigger"], v["neutral"]
        flip = (not t["passed"]) and n["passed"]
        flips += flip
        intent_flips[intent][0] += flip
        intent_flips[intent][1] += 1
        mark = "FLIP" if flip else ("--" if t["passed"] == n["passed"] else "partial")
        print(f"{pid:16s} {intent:9s} {t['p_block']:12.3f} {n['p_block']:12.3f}  {mark}")

    total = len(pairs)
    print("-" * 64)
    print(f"\nfull flips: {flips}/{total} ({100*flips/total:.0f}%)")
    for intent, (f, tot) in intent_flips.items():
        print(f"  {intent:9s} {f}/{tot}")
    # mean within-pair p_block swing
    swing = sum(v['trigger']['p_block'] - v['neutral']['p_block'] for v in pairs.values()) / total
    print(f"\nmean p_block(trigger) - p_block(neutral): {swing:+.3f}  (near +1.0 => strongly lexical)")


if __name__ == "__main__":
    main()
