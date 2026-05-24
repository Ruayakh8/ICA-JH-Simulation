"""
ICA-JH: Iterated Congestion-Aware Joint Heuristic

Algorithm outline
-----------------
1. Initialization
   Run HeurOSPF to get initial link weights, then run DemandsFirstWaypoints
   (greedy waypoint optimisation, GWO) on those weights to obtain the starting
   waypoint assignment.  Record the resulting MLU and Weighted Average Path
   Length (WAPL).

2. Iterative improvement loop  (up to max_iterations)
   a. Congestion identification
      Collect every link whose utilisation >= beta * current_MLU.
   b. Candidate demands
      Find all demands that route any fraction of traffic through a congested
      link.  Sort them descending by their total load contribution to congested
      links.
   c. Waypoint search & acceptance
      For each candidate demand test the direct route (s->t) and every single-
      node waypoint (s->w->t).  Accept a change only if it
        - strictly reduces the MLU, OR
        - keeps the MLU approximately equal but reduces the WAPL.
      Accepted changes are applied greedily so later demands in the same
      iteration already see an updated flow map.
   d. Re-optimisation
      If at least one change was accepted, re-run HeurOSPF with the updated
      waypoints.  Adopt the new weights only when OSPF finds a lower MLU.
   e. Early termination
      If no demand improved in a full pass, stop.

3. Return the best solution seen across all iterations.
"""

import time

import networkit as nk
import numpy as np

from algorithm.generic_sr import GenericSR
from algorithm.segment_routing.heur_ospf_weights import HeurOSPFWeights
from algorithm.segment_routing.demand_first_waypoints import DemandsFirstWaypoints
from utility import utility


class ICAJointHeuristic(GenericSR):

    def __init__(self, nodes: list, links: list, demands: list,
                 weights: dict = None, waypoints: dict = None,
                 max_iterations: int = 10, beta: float = 0.9,
                 seed: int = 42, time_out: int = None, **kwargs):
        super().__init__(nodes, links, demands, weights, waypoints)

        self.__nodes = nodes
        self.__links = links
        self.__demands = demands
        self.__n = len(nodes)
        self.__capacities = {(u, v): c for u, v, c in links}
        self.__link_list = list(self.__capacities.keys())
        self.__max_iterations = max_iterations
        self.__beta = beta
        self.__seed = seed
        self.__time_out = time_out if time_out else utility.TIME_LIMIT
        self.__start_time = None

    # ──────────────────────────────────────────────────────────────────────────
    # Time helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _elapsed(self) -> float:
        return time.time() - self.__start_time

    def _remaining(self) -> float:
        return max(1.0, self.__time_out - self._elapsed())

    def _timed_out(self) -> bool:
        return self._elapsed() >= self.__time_out

    # ──────────────────────────────────────────────────────────────────────────
    # Graph / routing helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_graph(self, weights: dict):
        """Build a weighted directed networkit graph and run APSP."""
        g = nk.Graph(weighted=True, directed=True, n=self.__n)
        for u, v in self.__link_list:
            g.addEdge(u, v, weights[(u, v)])
        apsp = nk.distance.APSP(g)
        apsp.run()
        return g, apsp.getDistances()

    def _sp_fraction_map(self, g, distances, weights) -> np.ndarray:
        """
        Compute fraction_map[s][t][u][v] = fraction of (s->t) demand routed
        over link (u,v) under ECMP shortest-path routing with the given weights.

        Identical logic to DemandsFirstWaypoints.__get_shortest_path_fraction_map.
        """
        n = self.__n
        frac = np.zeros((n, n, n, n), float)
        for s in range(n):
            # Process nodes in increasing distance-from-s order
            u_map = dict(zip(range(n), np.array(distances[s]).argsort()))
            for t in range(n):
                if s == t:
                    continue
                node_fracs = np.zeros(n, float)
                node_fracs[s] = 1.0
                for u_idx in range(n - 1):
                    u = u_map[u_idx]
                    f = node_fracs[u]
                    if f == 0.0:
                        continue
                    # Link (u,v) is on a shortest path to t iff
                    #   weight(u,v) == dist(u,t) - dist(v,t)
                    succ = [v for v in g.iterNeighbors(u)
                            if weights[(u, v)] == distances[u][t] - distances[v][t]]
                    if not succ:
                        continue
                    share = f / len(succ)
                    for v in succ:
                        frac[s][t][u][v] += share
                        if v != t:
                            node_fracs[v] += share
        return frac

    def _flow_map(self, frac: np.ndarray, waypoints: dict) -> np.ndarray:
        """
        Aggregate traffic onto links given the current waypoint assignment.
        Each demand (s,t,d) is routed along its waypoint segments; each segment
        (p,q) carries d units of traffic distributed by ECMP fractions.
        """
        flow = np.zeros((self.__n, self.__n), float)
        for d_idx, (s, t, d) in enumerate(self.__demands):
            for p, q in waypoints.get(d_idx, [(s, t)]):
                flow += frac[p][q] * d
        return flow

    def _loads_and_mlu(self, flow: np.ndarray):
        """Return per-link utilisation dict and the maximum link utilisation."""
        loads = {(u, v): flow[u][v] / self.__capacities[(u, v)]
                 for u, v in self.__link_list}
        return loads, max(loads.values())

    def _wapl(self, distances, waypoints: dict) -> float:
        """Weighted Average Path Length over all demands."""
        total_d = sum(d for _, _, d in self.__demands)
        if total_d == 0.0:
            return 0.0
        wsum = sum(
            d * sum(distances[p][q] for p, q in waypoints.get(idx, [(s, t)]))
            for idx, (s, t, d) in enumerate(self.__demands)
        )
        return wsum / total_d

    # ──────────────────────────────────────────────────────────────────────────
    # Sub-algorithm wrappers
    # ──────────────────────────────────────────────────────────────────────────

    def _run_ospf(self, waypoints):
        """Run HeurOSPF with the given waypoints; return (weights, loads, mlu)."""
        ospf = HeurOSPFWeights(
            self.__nodes, self.__links, self.__demands,
            weights=None, waypoints=waypoints,
            seed=self.__seed, time_out=self._remaining()
        )
        sol = ospf.solve()
        return sol['weights'], sol['loads'], sol['objective']

    def _run_gwo(self, weights):
        """Run greedy waypoint optimisation (DemandsFirstWaypoints)."""
        gwo = DemandsFirstWaypoints(
            self.__nodes, self.__links, self.__demands,
            weights=weights, waypoints=None
        )
        return gwo.solve()['waypoints']

    # ──────────────────────────────────────────────────────────────────────────
    # Main solve
    # ──────────────────────────────────────────────────────────────────────────

    def solve(self) -> dict:
        self.__start_time = t_start = time.time()
        pt_start = time.process_time()

        # ── Step 1: Initialisation ────────────────────────────────────────────
        # 1a. HeurOSPF for initial weights
        weights, _, _ = self._run_ospf(waypoints=None)

        # 1b. Greedy waypoint optimisation on those weights
        waypoints = self._run_gwo(weights)

        # Build graph state for the initial solution
        g, distances = self._build_graph(weights)
        frac = self._sp_fraction_map(g, distances, weights)
        flow = self._flow_map(frac, waypoints)
        loads, mlu = self._loads_and_mlu(flow)
        wapl = self._wapl(distances, waypoints)
        total_demand = sum(d for _, _, d in self.__demands)

        best_mlu = mlu
        best_wapl = wapl
        best_solution = dict(weights=dict(weights), waypoints=dict(waypoints),
                             loads=dict(loads), objective=mlu, wapl=wapl)

        # ── Step 2: Iterative improvement loop ───────────────────────────────
        for _it in range(self.__max_iterations):
            if self._timed_out():
                break

            improved_this_iter = False

            # ── Step 3: Congested links ───────────────────────────────────────
            threshold = self.__beta * mlu
            congested = {(u, v) for u, v in self.__link_list
                         if loads[(u, v)] >= threshold}
            if not congested:
                break

            # ── Step 4: Candidate demands sorted by load contribution ─────────
            contrib = {}
            for d_idx, (s, t, d) in enumerate(self.__demands):
                segs = waypoints.get(d_idx, [(s, t)])
                c = sum(d * frac[p][q][u][v]
                        for u, v in congested
                        for p, q in segs)
                if c > 0.0:
                    contrib[d_idx] = c
            sorted_demands = sorted(contrib, key=lambda i: -contrib[i])

            # ── Step 5: Waypoint search & acceptance ──────────────────────────
            for d_idx in sorted_demands:
                if self._timed_out():
                    break

                s, t, d = self.__demands[d_idx]
                cur_segs = waypoints.get(d_idx, [(s, t)])
                cur_path_len = sum(distances[p][q] for p, q in cur_segs)

                # Remove this demand's current contribution from the flow map
                flow_base = flow.copy()
                for p, q in cur_segs:
                    flow_base -= frac[p][q] * d

                best_wp_mlu = mlu        # best MLU seen for this demand slot
                best_wp_wapl = wapl      # corresponding WAPL
                best_new_segs = None

                # Test direct route (s->t) and every single-node waypoint
                candidates = [[(s, t)]] + [[(s, w), (w, t)]
                                           for w in range(self.__n)
                                           if w != s and w != t]

                for new_segs in candidates:
                    test_flow = flow_base.copy()
                    for p, q in new_segs:
                        test_flow += frac[p][q] * d
                    _, test_mlu = self._loads_and_mlu(test_flow)

                    # Incremental WAPL: only this demand's path length changes
                    new_path_len = sum(distances[p][q] for p, q in new_segs)
                    test_wapl = wapl + d * (new_path_len - cur_path_len) / total_demand

                    # Acceptance criteria (spec steps 5a / 5b)
                    if test_mlu < best_wp_mlu - 1e-9:
                        best_wp_mlu = test_mlu
                        best_wp_wapl = test_wapl
                        best_new_segs = new_segs
                    elif (abs(test_mlu - best_wp_mlu) < 1e-9
                          and test_wapl < best_wp_wapl - 1e-9):
                        best_wp_wapl = test_wapl
                        best_new_segs = new_segs

                # Apply the accepted change and update the running flow state
                if best_new_segs is not None:
                    waypoints = dict(waypoints)
                    waypoints[d_idx] = best_new_segs
                    flow = flow_base.copy()
                    for p, q in best_new_segs:
                        flow += frac[p][q] * d
                    loads, mlu = self._loads_and_mlu(flow)
                    wapl = best_wp_wapl
                    improved_this_iter = True

            # ── Step 6: Re-optimise weights with HeurOSPF ────────────────────
            if improved_this_iter and not self._timed_out():
                new_weights, _, new_ospf_mlu = self._run_ospf(waypoints)
                if new_ospf_mlu < mlu:
                    weights = new_weights
                    g, distances = self._build_graph(weights)
                    frac = self._sp_fraction_map(g, distances, weights)
                    flow = self._flow_map(frac, waypoints)
                    loads, mlu = self._loads_and_mlu(flow)
                    wapl = self._wapl(distances, waypoints)

                # ── Step 7: Track best solution ───────────────────────────────
                if mlu < best_mlu:
                    best_mlu = mlu
                    best_wapl = wapl
                    best_solution = dict(weights=dict(weights),
                                        waypoints=dict(waypoints),
                                        loads=dict(loads),
                                        objective=mlu,
                                        wapl=wapl)
            else:
                # No improvement in this full pass → early termination
                break

        solution = dict(best_solution)
        solution['execution_time'] = time.time() - t_start
        solution['process_time'] = time.process_time() - pt_start
        return solution

    def get_name(self) -> str:
        return "ica_joint_heuristic"
