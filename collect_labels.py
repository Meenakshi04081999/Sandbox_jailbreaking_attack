"""
Label-collection harness.

Queries the TEACHER over a set of inputs and records (input, label) pairs.
This output IS the distillation training set: the student will later be trained
to reproduce the teacher's binary decisions from exactly this data.

Two files are written:
  - labels.jsonl        binary only {input, passed}  -- attacker-faithful signal
  - labels_debug.jsonl  + raw label/score            -- OUR analysis, off-limits
                                                          to the student/attack code

Usage:
  python collect_labels.py seed_prompts.txt
  python collect_labels.py my_prompts.txt --out run1.jsonl --budget 500
"""
import json
import argparse
from pathlib import Path

from teacher import Teacher


def collect(prompts, out="labels.jsonl", debug_out=None, budget=None):
    if debug_out is None:
        p = Path(out)
        debug_out = str(p.with_name(p.stem + "_debug" + p.suffix))
    teacher = Teacher()
    out_p, dbg_p = Path(out), Path(debug_out)
    n = 0
    with out_p.open("w", encoding="utf-8") as f, dbg_p.open("w", encoding="utf-8") as df:
        for raw in prompts:
            text = raw.strip()
            if not text:
                continue
            if budget is not None and teacher.query_count >= budget:
                print(f"Query budget {budget} reached; stopping early.")
                break
            v = teacher.classify_verbose(text)
            f.write(json.dumps({"input": text, "passed": v["passed"]}) + "\n")
            df.write(json.dumps({"input": text, **v}) + "\n")
            n += 1

    passed = sum(1 for line in out_p.read_text(encoding="utf-8").splitlines()
                 if json.loads(line)["passed"])
    print(f"Collected {n} labels in {teacher.query_count} teacher queries -> {out}")
    print(f"  pass: {passed}   not-pass: {n - passed}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts_file", help="newline-separated prompts")
    ap.add_argument("--out", default="labels.jsonl")
    ap.add_argument("--budget", type=int, default=None,
                    help="max teacher queries (None = no limit)")
    args = ap.parse_args()
    prompts = Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
    collect(prompts, out=args.out, budget=args.budget)


if __name__ == "__main__":
    main()
