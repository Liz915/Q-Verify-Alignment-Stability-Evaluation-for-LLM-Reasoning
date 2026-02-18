import argparse
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.append(os.getcwd())
from src.data.task_benchmark import collect_task_entries, stratified_split, write_jsonl


def summarize(entries):
    by_family = Counter(e["family"] for e in entries)
    by_bin = Counter(e["size_bin"] for e in entries)
    return by_family, by_bin


def main():
    parser = argparse.ArgumentParser(description="Build train/val/test manifests from benchmark tasks.")
    parser.add_argument("--task_root", type=str, default="benchmarks/tasks")
    parser.add_argument("--output_dir", type=str, default="data/task_manifests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_two_qubit_gates", type=int, default=1)
    parser.add_argument("--max_eval_pairs_per_task", type=int, default=64)
    parser.add_argument("--target_min_tasks", type=int, default=20)
    parser.add_argument("--no_recursive", action="store_true")
    args = parser.parse_args()

    entries = collect_task_entries(
        task_root=args.task_root,
        recursive=not args.no_recursive,
        keep_single_qubit=False,
        max_eval_pairs_per_task=args.max_eval_pairs_per_task,
        min_two_qubit_gates=args.min_two_qubit_gates,
    )

    if not entries:
        print("[ERR] No valid tasks found. Check task_root and parser compatibility.")
        return

    by_family, by_bin = summarize(entries)
    print(f"[INFO] Total valid tasks: {len(entries)}")
    print(f"[INFO] Family distribution: {dict(by_family)}")
    print(f"[INFO] Size-bin distribution: {dict(by_bin)}")
    if len(entries) < args.target_min_tasks:
        print(
            f"[WARN] Only {len(entries)} tasks found (<{args.target_min_tasks}). "
            "Add more QASM files for publication-grade coverage."
        )

    splits = stratified_split(entries, seed=args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    write_jsonl(entries, os.path.join(args.output_dir, "all_tasks.jsonl"))
    write_jsonl(splits["train"], os.path.join(args.output_dir, "train.jsonl"))
    write_jsonl(splits["val"], os.path.join(args.output_dir, "val.jsonl"))
    write_jsonl(splits["test"], os.path.join(args.output_dir, "test.jsonl"))

    rows = []
    for split_name, split_entries in [("all", entries), ("train", splits["train"]), ("val", splits["val"]), ("test", splits["test"])]:
        fam_cnt = Counter(e["family"] for e in split_entries)
        bin_cnt = Counter(e["size_bin"] for e in split_entries)
        for family, cnt in fam_cnt.items():
            rows.append({"split": split_name, "axis": "family", "key": family, "count": cnt})
        for bname, cnt in bin_cnt.items():
            rows.append({"split": split_name, "axis": "size_bin", "key": bname, "count": cnt})
        rows.append({"split": split_name, "axis": "total", "key": "tasks", "count": len(split_entries)})

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[DONE] Wrote manifests to {args.output_dir}")
    print(f"[DONE] Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
