# ICA-JH: Iterated Congestion-Aware Joint Heuristic for WAN Traffic Engineering

## Overview

This repository contains the implementation and evaluation code for the university project
**"ICA-JH: Iterated Congestion-Aware Joint Heuristic"**, developed as an extension to the
baseline traffic engineering framework for Wide Area Networks (WANs).

The project evaluates joint link-weight and segment (waypoint) optimisation strategies using
real-world network topologies from [SNDLib](http://sndlib.zib.de/home.action).  
Our proposed algorithm, **ICA-JH** (`ica_joint_heuristic`), iteratively identifies congested
links and re-routes traffic through alternative waypoints, combining OSPF weight optimisation
with greedy waypoint search. It is benchmarked against the following baselines:

| Algorithm | Description |
| --- | --- |
| `demand_first_waypoints` | Greedy waypoint optimisation (Demands-First) |
| `heur_ospf_weights` | OSPF link-weight heuristic (Fortz & Thorup) |
| `inverse_capacity` | Inverse-capacity link weights |
| `uniform_weights` | Uniform link weights |
| `segment_ilp` | Exact ILP solver (WEIGHTS / WAYPOINTS / JOINT variants) |

---

## Prerequisites

### Python & Conda

Python **3.7.10** is required. We use [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
as the package manager.

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate wan_sr
```

Key dependencies installed by the environment:

| Package | Version | Purpose |
| --- | --- | --- |
| `networkit` | 8.1 | ECMP shortest-path routing (ICA-JH, HeurOSPF, GWO) |
| `networkx` | 2.5.1 | Topology file parsing (`.graphml`, `.xml`) |
| `numpy` | 1.20.3 | Numerical computations (flow maps, fraction maps) |
| `gurobipy` | via `gurobi=9.1.2` | ILP solver and MCF demand generation |
| `matplotlib` / `seaborn` / `pandas` | — | Result plotting |

> **Note:** `networkit 8.1` does **not** support Microsoft Windows natively.
> The evaluation was conducted on **Ubuntu 18.04.5 LTS**.
> On Windows, use WSL2 (Ubuntu) to run this project.

### Gurobi License (Required)

A valid Gurobi licence is **strictly required** to run:
- The ILP algorithms (`segment_ilp` — WEIGHTS, WAYPOINTS, JOINT variants)
- The MCF demand generator (`maximal_multi_commodity_flow_dp`)

Academic licences are freely available:
[https://www.gurobi.com/academia/academic-program-and-licenses/](https://www.gurobi.com/academia/academic-program-and-licenses/)

Download Gurobi Optimizer 9.1.2:
[https://www.gurobi.com/downloads/](https://www.gurobi.com/downloads/)

---

## Project Structure

| Path | Description |
| --- | --- |
| `src/main.py` | Entry point — configures and runs all evaluations |
| `src/algorithm/segment_routing/ica_joint_heuristic.py` | **Our proposed ICA-JH algorithm** |
| `src/algorithm/segment_routing/` | All routing algorithms (heuristics + ILP) |
| `src/demand/` | Real-world and synthetic traffic demand generators |
| `src/topology/` | Topology providers (SNDLib, TopologyZoo) |
| `src/utility/` | Shared constants, JSON result handler |
| `src/plot_results.py` | Plots Fig. 3–5 from JSON result files |
| `data/` | SNDLib topology and demand data |
| `data/topologies/topology_zoo/archive/` | TopologyZoo `.graphml` files (see below) |
| `out/` | Output directory for JSON results and plots |

---

## Topologies Evaluated

All experiments use the following six **SNDLib** topologies, selected for their range of
network sizes:

| Topology | \|V\| (nodes) | \|E\| (links) |
| --- | --- | --- |
| `abilene` | 12 | 30 |
| `geant` | 22 | 72 |
| `cost266` | 37 | 114 |
| `zib54` | 54 | 160 |
| `germany50` | 50 | 176 |
| `ta2` | 65 | 216 |

> **ILP safety guard:** The `segment_ilp` variants are computationally expensive and are
> restricted to run only on `abilene` and `geant` (small topologies). Larger topologies
> automatically skip ILP execution via a guard in `abilene_all_algorithms()`.

---

## Real-World Data Setup

### SNDLib (included)

Topology and demand data from SNDLib is already included under `data/` and is redistributed
under the [ZIB Academic License](data/LICENSE_SNDLib).

### TopologyZoo (manual step)

TopologyZoo data is **not** included and must be added manually:

1. Download the full dataset: [http://www.topology-zoo.org/files/archive.zip](http://www.topology-zoo.org/files/archive.zip)
2. Extract the archive
3. Place all `.graphml` files into:
   ```
   data/topologies/topology_zoo/archive/
   ```

---

## Run Instructions

Navigate to the source directory:

```bash
cd src
```

Run all evaluations:

```bash
python3 main.py
```

Results are written as JSON files to the `out/` directory:

| Output file | Contents |
| --- | --- |
| `out/results_all_topologies.json` | Fig. 3 — all 6 topologies, synthetic demands |
| `out/results_all_algorithms.json` | Fig. 4 — all algorithms on all 6 topologies |
| `out/results_real_demands.json` | Fig. 5 — real SNDLib demands (abilene, geant, germany50) |

### Plot Results

```bash
python3 plot_results.py "../out/"
```

---

## Global Configuration (`main.py`)

| Parameter | Value | Description |
| --- | --- | --- |
| `DEMANDS_SAMPLES` | `10` | Number of demand matrix samples per topology |
| `ALGORITHM_TIME_OUT` | `4 * 60 * 60` | Per-algorithm time limit (4 hours) |
| `ACTIVE_PAIRS_FRACTION` | `0.2` | Fraction of node pairs with active demands |
| `SEED` | `318924135` | Random seed for reproducibility |

---

## Important Note on Baseline Comparison (`sequential_combination`)

The professor's reference baseline algorithm, **`sequential_combination`** (HeurOSPF followed
by Greedy Waypoint Optimisation), is **currently commented out** in all algorithm lists in
`main.py`. This was done deliberately for the final submission run to reduce total execution
time, since the algorithm runs as a strict sequential sub-routine already embedded inside
ICA-JH's initialisation phase.

**To re-enable the baseline for a side-by-side comparison**, uncomment the following lines in
`main.py`:

In `all_topologies_synthetic_demands()` and `snd_real_demands()`:
```python
# "sequential_combination",   ← remove the #
```

In `abilene_all_algorithms()`:
```python
# ("sequential_combination", ""),   ← remove the #
```

When uncommented, the output will show results for both algorithms. The comparison will
demonstrate that **ICA-JH achieves the same or lower Maximum Link Utilisation (MLU) as
`sequential_combination`, while producing a significantly shorter Weighted Average Path
Length (WAPL)**. This is because ICA-JH's iterative congestion-aware refinement actively
minimises WAPL as a secondary objective whenever the MLU is not strictly reduced, resulting
in more efficient traffic routing paths.

The WAPL for every algorithm is printed in the console output and stored in the JSON results:
```
objective: 0.7823 | WAPL: 4.1250
```

---

## Contact

University project — ICA-JH algorithm implementation.  
Base framework by Thomas Fenz — [University of Vienna, Communication Technologies](https://ct.cs.univie.ac.at/).

*This project is licensed under the [MIT License](LICENSE).*
