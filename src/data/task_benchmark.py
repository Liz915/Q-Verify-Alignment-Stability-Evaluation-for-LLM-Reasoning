import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import yaml

TWO_QUBIT_OP_MAP = {
    "cx": "cx",
    "cnot": "cx",
    "cz": "cx",
    "swap": "swap",
    "cp": "cx",
    "crz": "cx",
    "rzz": "cx",
    "ecr": "cx",
    "iswap": "swap",
}

ONE_QUBIT_OP_MAP = {
    "h": "h",
    "x": "x",
    "y": "x",
    "z": "x",
    "s": "x",
    "sdg": "x",
    "t": "x",
    "tdg": "x",
    "sx": "x",
    "sxdg": "x",
    "rx": "h",
    "ry": "h",
    "rz": "h",
    "u": "h",
    "u1": "h",
    "u2": "h",
    "u3": "h",
    "id": "h",
}

IGNORED_OPS = {
    "openqasm",
    "include",
    "qreg",
    "creg",
    "barrier",
    "measure",
    "reset",
}

QASM_LINE_PATTERN = re.compile(r"^([a-z][a-z0-9_]*)\s*(\([^)]*\))?\s+(.+);$", re.IGNORECASE)


def infer_family(path_like: str) -> str:
    name = Path(path_like).name.lower()
    if "ghz" in name:
        return "ghz"
    if "graphstate" in name:
        return "graphstate"
    if "qft" in name:
        return "qft"
    if "vqe" in name:
        return "vqe"
    if "qaoa" in name:
        return "qaoa"
    return "other"


def size_bin(n_qubits: int) -> str:
    if n_qubits <= 20:
        return "small"
    if n_qubits <= 60:
        return "medium"
    return "large"


def _normalize_gate(op: str, qubits: Sequence[int], keep_single_qubit: bool) -> Dict | None:
    op = op.lower()
    if len(qubits) == 2:
        mapped = TWO_QUBIT_OP_MAP.get(op)
        if mapped is None:
            return None
        return {"op": mapped, "qubits": [int(qubits[0]), int(qubits[1])], "source_op": op}

    if len(qubits) == 1 and keep_single_qubit:
        mapped = ONE_QUBIT_OP_MAP.get(op)
        if mapped is None:
            return None
        return {"op": mapped, "qubits": [int(qubits[0])], "source_op": op}

    return None


def parse_qasm_file(qasm_path: str, keep_single_qubit: bool = False) -> List[Dict]:
    gates: List[Dict] = []
    with open(qasm_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.split("//", 1)[0].strip()
            if not line or ";" not in line:
                continue

            match = QASM_LINE_PATTERN.match(line)
            if not match:
                continue

            op = match.group(1).lower()
            if op in IGNORED_OPS:
                continue

            qubits = [int(x) for x in re.findall(r"q\[(\d+)\]", line)]
            normalized = _normalize_gate(op, qubits, keep_single_qubit=keep_single_qubit)
            if normalized is not None:
                gates.append(normalized)

    return gates


def parse_yaml_task(yaml_path: str, keep_single_qubit: bool = False) -> Tuple[str, List[Dict]]:
    with open(yaml_path, "r", encoding="utf-8", errors="ignore") as f:
        record = yaml.safe_load(f)

    task_name = record.get("task_name", Path(yaml_path).stem)
    gates: List[Dict] = []
    for gate in record.get("gates", []):
        op = str(gate.get("op", "")).lower()
        qubits = gate.get("qubits", [])
        if not isinstance(qubits, list):
            continue
        normalized = _normalize_gate(op, qubits, keep_single_qubit=keep_single_qubit)
        if normalized is not None:
            gates.append(normalized)
    return task_name, gates


def collect_task_files(task_root: str, recursive: bool = True) -> List[Path]:
    root = Path(task_root)
    if not root.exists():
        return []

    patterns = ("*.qasm", "*.yaml", "*.yml")
    files: List[Path] = []
    if recursive:
        for pattern in patterns:
            files.extend(root.rglob(pattern))
    else:
        for pattern in patterns:
            files.extend(root.glob(pattern))

    files = sorted(set(files))
    return files


def _build_eval_pairs(gates: Iterable[Dict], max_eval_pairs_per_task: int = 64) -> List[List[int]]:
    pairs: List[List[int]] = []
    seen = set()
    for gate in gates:
        qubits = gate.get("qubits", [])
        if len(qubits) != 2:
            continue
        pair = tuple(int(q) for q in qubits)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append([pair[0], pair[1]])
        if len(pairs) >= max_eval_pairs_per_task:
            break
    return pairs


def collect_task_entries(
    task_root: str = "benchmarks/tasks",
    recursive: bool = True,
    keep_single_qubit: bool = False,
    max_eval_pairs_per_task: int = 64,
    min_two_qubit_gates: int = 1,
) -> List[Dict]:
    root = Path(task_root)
    entries: List[Dict] = []
    for path in collect_task_files(task_root, recursive=recursive):
        rel_path = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        family = infer_family(rel_path)

        if path.suffix.lower() == ".qasm":
            gates = parse_qasm_file(str(path), keep_single_qubit=keep_single_qubit)
            task_name = path.stem
        else:
            task_name, gates = parse_yaml_task(str(path), keep_single_qubit=keep_single_qubit)

        two_qubit = [g for g in gates if len(g.get("qubits", [])) == 2]
        if len(two_qubit) < min_two_qubit_gates:
            continue

        max_qubit = max((max(g["qubits"]) for g in gates), default=-1)
        n_qubits = max_qubit + 1 if max_qubit >= 0 else 0
        task_id = re.sub(r"[^a-zA-Z0-9_]+", "_", f"{family}_{task_name}")
        eval_pairs = _build_eval_pairs(gates, max_eval_pairs_per_task=max_eval_pairs_per_task)

        entries.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "family": family,
                "source_path": str(path),
                "format": path.suffix.lower().lstrip("."),
                "n_qubits": n_qubits,
                "size_bin": size_bin(n_qubits),
                "n_gates": len(gates),
                "n_two_qubit_gates": len(two_qubit),
                "eval_pairs": eval_pairs,
            }
        )
    return entries


def stratified_split(
    entries: Sequence[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for entry in entries:
        key = (entry["family"], entry["size_bin"])
        groups.setdefault(key, []).append(entry)

    split = {"train": [], "val": [], "test": []}
    for key, group in groups.items():
        bucket = list(group)
        rng.shuffle(bucket)
        n = len(bucket)

        if n == 1:
            split["train"].extend(bucket)
            continue
        if n == 2:
            split["train"].append(bucket[0])
            split["test"].append(bucket[1])
            continue

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1

        split["train"].extend(bucket[:n_train])
        split["val"].extend(bucket[n_train : n_train + n_val])
        split["test"].extend(bucket[n_train + n_val :])

    return split


def write_jsonl(rows: Sequence[Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_eval_task_pool(manifest_rows: Sequence[Dict], max_pairs_per_task: int = 32) -> List[Dict]:
    pool = []
    for row in manifest_rows:
        task_id = row.get("task_id", "unknown")
        family = row.get("family", "other")
        for pair in row.get("eval_pairs", [])[:max_pairs_per_task]:
            if len(pair) != 2:
                continue
            q1, q2 = int(pair[0]), int(pair[1])
            if q1 == q2:
                continue
            pool.append({"op": "cx", "qubits": [q1, q2], "task_id": task_id, "family": family})
    return pool


def sample_eval_tasks(pool: Sequence[Dict], n_tasks: int, seed: int) -> List[Dict]:
    if not pool:
        return []
    rng = random.Random(seed)
    if n_tasks <= len(pool):
        return rng.sample(list(pool), n_tasks)
    return [rng.choice(pool) for _ in range(n_tasks)]

