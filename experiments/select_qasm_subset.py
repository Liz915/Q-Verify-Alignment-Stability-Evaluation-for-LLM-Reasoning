import argparse
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


def family_from_name(name: str) -> str:
    lower = name.lower()
    if "ghz" in lower:
        return "ghz"
    if "graphstate" in lower:
        return "graphstate"
    if "qft" in lower:
        return "qft"
    return "other"


def qubits_from_name(name: str) -> int:
    m = re.search(r"_qiskit_(\d+)\.qasm$", name.lower())
    if m:
        return int(m.group(1)) + 1
    return -1


def evenly_pick(files, k):
    if k <= 0 or not files:
        return []
    if len(files) <= k:
        return list(files)
    idxs = [round(i * (len(files) - 1) / (k - 1)) for i in range(k)]
    return [files[i] for i in sorted(set(idxs))]


def main():
    parser = argparse.ArgumentParser(description="Select a balanced QASM subset (20+) across families/scales.")
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing many MQT-Bench qasm files.")
    parser.add_argument("--dest_dir", type=str, default="benchmarks/tasks/qasm_files")
    parser.add_argument("--target_total", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy files. If unset, use symlinks.")
    args = parser.parse_args()

    src = Path(args.source_dir).expanduser()
    if not src.exists() and not src.is_absolute():
        # Convenience fallback: treat relative source_dir as HOME-relative.
        home_rel = Path.home() / src
        if home_rel.exists():
            src = home_rel
    if not src.exists() and args.source_dir.startswith("下载/"):
        # Common Chinese alias on macOS maps to Downloads in filesystem.
        suffix = args.source_dir.split("/", 1)[1] if "/" in args.source_dir else ""
        zh_fallback = (Path.home() / "Downloads" / suffix) if suffix else (Path.home() / "Downloads")
        if zh_fallback.exists():
            src = zh_fallback

    if not src.exists():
        raise FileNotFoundError(
            f"source_dir not found: {src}. "
            f"Try an absolute path like: {Path.home() / 'Downloads' / 'MQTBench'}"
        )

    all_qasm = sorted(src.rglob("*.qasm"))
    if not all_qasm:
        raise RuntimeError(f"No .qasm files found in {src}")

    by_family = defaultdict(list)
    for qasm in all_qasm:
        fam = family_from_name(qasm.name)
        qcount = qubits_from_name(qasm.name)
        by_family[fam].append((qcount, qasm))

    for fam in by_family:
        by_family[fam] = sorted(by_family[fam], key=lambda x: x[0])

    rng = random.Random(args.seed)
    families = [f for f in ["ghz", "graphstate", "qft"] if by_family.get(f)]
    if not families:
        families = sorted(by_family.keys())

    per_family = max(1, args.target_total // max(1, len(families)))
    selected = []
    leftovers = []
    for fam in families:
        files = [p for _, p in by_family[fam]]
        pick = evenly_pick(files, per_family)
        selected.extend(pick)
        leftovers.extend([p for p in files if p not in set(pick)])

    if len(selected) < args.target_total:
        rng.shuffle(leftovers)
        needed = args.target_total - len(selected)
        selected.extend(leftovers[:needed])

    selected = sorted(set(selected))
    os.makedirs(args.dest_dir, exist_ok=True)

    # Keep destination synchronized with the selected subset.
    selected_names = {p.name for p in selected}
    for existing in Path(args.dest_dir).glob("*.qasm"):
        if existing.name not in selected_names:
            existing.unlink()

    for p in selected:
        dest = Path(args.dest_dir) / p.name
        if dest.exists():
            dest.unlink()
        if args.copy:
            shutil.copy2(p, dest)
        else:
            os.symlink(p.resolve(), dest)

    print(f"[DONE] Selected {len(selected)} files into {args.dest_dir}")
    by_family_sel = defaultdict(int)
    for p in selected:
        by_family_sel[family_from_name(p.name)] += 1
    print(f"[INFO] Family counts: {dict(by_family_sel)}")


if __name__ == "__main__":
    main()
