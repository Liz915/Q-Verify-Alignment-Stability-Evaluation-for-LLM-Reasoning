# Q-Verify Benchmarks

This directory defines the **"Laws of Physics"** (Topologies) and **"Goal States"** (Tasks) for the VPA framework.

## 1. Topologies (Hardware Constraints)
We benchmark against two leading superconducting architectures:
- **IBM Eagle (127 qubits)**: Uses a *Heavy-Hex* lattice, characterized by low connectivity and challenging routing bottlenecks.
- **Google Sycamore (53 qubits)**: Uses a *2D Grid* lattice, offering nearest-neighbor connectivity.

## 2. Tasks (Reasoning Challenges)
- **GHZ State**: Tests global entanglement capability (Star graph routing).
- **QAOA (MaxCut)**: Tests combinatorial optimization on 3-regular graphs (Complex arbitrary routing).
- **VQE (LiH)**: Tests variational ansatz preparation (Ladder-like routing).
