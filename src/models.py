"""The four optimization approaches, all on one capacitated facility-location
instance:

  (1) Transportation LP        -- exact flows for a fixed open set; yields duals.
  (2) LP relaxation of the CFLP -- continuous y in [0,1]; the lower bound.
  (3) MILP (exact)             -- binary y via branch-and-bound (HiGHS).
  (4) Heuristics: genetic algorithm and simulated annealing over the open set.
  (5) Gradient-based: Weiszfeld center-of-gravity (single + multi-facility).

The MILP is the ground truth the heuristics and the continuous method are judged
against.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
import geo

BIG = 1e15   # infeasible-cost sentinel


# ============================================================ transportation LP
def solve_transport_lp(inst, open_mask, want_duals=False):
    """Min-cost flow given which DCs are open. Returns cost, flows, duals.

    open_mask : boolean array over candidates (True = open).
    Splittable demand (multi-source). Infeasible -> cost = BIG.

    Only *allowed* arcs from *open* DCs are created as decision variables. This
    matters for the duals: fixing forbidden/closed arcs to zero via bounds
    instead would leave degenerate fixed variables that corrupt the demand
    shadow prices, so we exclude those arcs from the model entirely.
    """
    m, n = inst["m"], inst["n"]
    cost, cap, demand, allowed = inst["cost"], inst["cap"], inst["demand"], inst["allowed"]
    open_mask = np.asarray(open_mask, bool)
    open_idx = np.where(open_mask)[0]

    # quick feasibility screen
    if cap[open_mask].sum() + 1e-6 < demand.sum():
        return dict(status="infeasible-capacity", cost=BIG, flows=None)
    reachable = allowed[open_mask].any(axis=0) if open_mask.any() else np.zeros(n, bool)
    if not reachable.all():
        return dict(status="infeasible-coverage", cost=BIG, flows=None)

    # build variable list over allowed open arcs only
    arcs = [(i, j) for i in open_idx for j in range(n) if allowed[i, j]]
    col = {a: k for k, a in enumerate(arcs)}
    nv = len(arcs)
    c = np.array([cost[i, j] for (i, j) in arcs])
    # demand equality (n rows): sum over open i of x_ij = d_j
    A_eq = np.zeros((n, nv))
    for (i, j) in arcs:
        A_eq[j, col[(i, j)]] = 1.0
    b_eq = demand.copy()
    # capacity (per open DC): sum_j x_ij <= cap_i
    A_ub = np.zeros((len(open_idx), nv))
    for r, i in enumerate(open_idx):
        for j in range(n):
            if (i, j) in col:
                A_ub[r, col[(i, j)]] = 1.0
    b_ub = cap[open_idx]
    bounds = [(0.0, float(demand[j])) for (i, j) in arcs]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        return dict(status="infeasible-lp", cost=BIG, flows=None)
    flows = np.zeros((m, n))
    for k, (i, j) in enumerate(arcs):
        flows[i, j] = res.x[k]
    out = dict(status="optimal", cost=float(res.fun), flows=flows)
    if want_duals:
        # eqlin marginals = marginal cost of one more ton delivered to zone j
        out["dual_demand"] = np.asarray(res.eqlin.marginals, float)
        dc = np.zeros(m)
        dc[open_idx] = np.asarray(res.ineqlin.marginals, float)
        out["dual_capacity"] = dc
    return out


# ============================================================ MILP / LP relaxation
def _milp_matrices(inst, force_p=None, strong=True):
    m, n = inst["m"], inst["n"]
    cost, cap, fixed, demand, allowed = (inst["cost"], inst["cap"], inst["fixed"],
                                         inst["demand"], inst["allowed"])
    N = m + m * n                                        # [y (m)] + [x (m*n)]
    c = np.concatenate([fixed, cost.flatten()])
    cons = []
    # demand equality
    A_dem = np.zeros((n, N))
    for j in range(n):
        A_dem[j, m + j::n] = 1.0
    cons.append(LinearConstraint(A_dem, demand, demand))
    # capacity: sum_j x_ij - cap_i y_i <= 0
    A_cap = np.zeros((m, N))
    for i in range(m):
        A_cap[i, i] = -cap[i]
        A_cap[i, m + i * n:m + (i + 1) * n] = 1.0
    cons.append(LinearConstraint(A_cap, -np.inf, np.zeros(m)))
    # strong (disaggregated) linking: x_ij - d_j y_i <= 0  -> tightens LP relaxation
    if strong:
        A_link = np.zeros((m * n, N))
        r = 0
        for i in range(m):
            for j in range(n):
                A_link[r, m + i * n + j] = 1.0
                A_link[r, i] = -demand[j]
                r += 1
        cons.append(LinearConstraint(A_link, -np.inf, np.zeros(m * n)))
    if force_p is not None:                             # sum_i y_i = p
        A_p = np.zeros((1, N)); A_p[0, :m] = 1.0
        cons.append(LinearConstraint(A_p, force_p, force_p))
    # bounds
    lb = np.zeros(N)
    ub = np.ones(N)                                     # y_i <= 1
    xub = np.where(allowed, demand[None, :], 0.0).flatten()
    ub[m:] = xub
    return c, cons, lb, ub, m, n, N


def solve_milp(inst, force_p=None, relax=False, strong=True, time_limit=60.0):
    """Exact CFLP (relax=False) or its LP relaxation (relax=True)."""
    c, cons, lb, ub, m, n, N = _milp_matrices(inst, force_p, strong=strong)
    integ = np.zeros(N) if relax else np.concatenate([np.ones(m), np.zeros(m * n)])
    t0 = time.perf_counter()
    res = milp(c=c, constraints=cons, integrality=integ,
               bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "presolve": True})
    dt = time.perf_counter() - t0
    if res.x is None:
        return dict(status="infeasible", cost=BIG, time=dt)
    y = res.x[:m]
    x = res.x[m:].reshape(m, n)
    fixed_cost = float(inst["fixed"] @ (y if relax else np.round(y)))
    transport_cost = float((inst["cost"] * x).sum())
    return dict(status="optimal" if res.success else res.status,
                y=y, x=x, cost=float(res.fun),
                fixed_cost=fixed_cost, transport_cost=transport_cost,
                open_mask=(y > 0.5), n_open=int(round(y.sum())) if not relax else float(y.sum()),
                time=dt, mip_gap=getattr(res, "mip_gap", None))


# ============================================================ heuristics
def _repair(inst, open_vec, rng):
    """Nudge an open-vector toward feasibility: guarantee coverage (every zone
    reachable by an open DC) and aggregate capacity (open capacity >= total
    demand). These two conditions are necessary but not sufficient, because a
    local subset of zones can still exceed the capacity of the only DCs that can
    reach them. That residual case is handled exactly by the evaluator: the
    transportation LP is the feasibility gate, and it assigns any infeasible open
    set an infinite cost, so the genetic algorithm and annealer discard it. On
    the base instance about 1% of randomly repaired sets hit this case.
    """
    m, n = inst["m"], inst["n"]
    cap, demand, allowed, dist = inst["cap"], inst["demand"], inst["allowed"], inst["dist"]
    v = np.asarray(open_vec, bool).copy()
    if not v.any():
        v[rng.integers(m)] = True
    # coverage: every zone reachable by some open DC
    reachable = allowed[v].any(axis=0)
    for j in np.where(~reachable)[0]:
        cands = np.where(allowed[:, j])[0]              # DCs that can reach j
        if len(cands):
            v[cands[np.argmin(dist[cands, j])]] = True  # open the nearest one
    # aggregate capacity: open more DCs (most central first) until enough
    order = np.argsort(dist.mean(axis=1))
    k = 0
    while cap[v].sum() + 1e-6 < demand.sum() and k < m:
        v[order[k]] = True
        k += 1
    return v


def _make_evaluator(inst):
    """Cost of an open-vector, with memoization of the inner LP."""
    cache = {}
    fixed = inst["fixed"]

    def evaluate(open_vec):
        key = tuple(bool(b) for b in open_vec)
        if key in cache:
            return cache[key]
        mask = np.asarray(open_vec, bool)
        lp = solve_transport_lp(inst, mask)
        if lp["cost"] >= BIG:
            val = BIG
        else:
            val = float(fixed[mask].sum() + lp["cost"])
        cache[key] = val
        return val

    return evaluate, cache


def genetic_algorithm(inst, rng, cfg):
    """GA over the binary open-set vector. Fitness = total network cost."""
    m = inst["m"]
    evaluate, _ = _make_evaluator(inst)
    t0 = time.perf_counter()

    def random_ind():
        return _repair(inst, rng.random(m) < 0.5, rng)

    pop = [random_ind() for _ in range(cfg["pop"])]
    fit = np.array([evaluate(ind) for ind in pop])
    history = [float(fit.min())]

    for _ in range(cfg["gen"]):
        new = []
        elite = np.argsort(fit)[:cfg["elite"]]
        for e in elite:
            new.append(pop[e].copy())
        while len(new) < cfg["pop"]:
            # tournament selection (two parents)
            def pick():
                idx = rng.integers(0, cfg["pop"], cfg["tour"])
                return pop[idx[np.argmin(fit[idx])]]
            p1, p2 = pick(), pick()
            mask = rng.random(m) < 0.5                  # uniform crossover
            child = np.where(mask, p1, p2)
            flip = rng.random(m) < cfg["mut"]           # bit-flip mutation
            child = np.where(flip, ~child, child)
            new.append(_repair(inst, child, rng))
        pop = new
        fit = np.array([evaluate(ind) for ind in pop])
        history.append(float(fit.min()))

    best = int(np.argmin(fit))
    return dict(open_mask=np.asarray(pop[best], bool), cost=float(fit[best]),
                history=history, time=time.perf_counter() - t0,
                n_open=int(np.asarray(pop[best]).sum()))


def simulated_annealing(inst, rng, cfg):
    """SA over the binary open-set vector."""
    m = inst["m"]
    evaluate, _ = _make_evaluator(inst)
    t0 = time.perf_counter()
    cur = _repair(inst, rng.random(m) < 0.5, rng)
    cur_cost = evaluate(cur)
    best, best_cost = cur.copy(), cur_cost
    T = cfg["T0"]
    history = [best_cost]
    for _ in range(cfg["iters"]):
        cand = cur.copy()
        k = rng.integers(m)
        cand[k] = ~cand[k]                              # flip exactly one site
        cand = _repair(inst, cand, rng)
        cand_cost = evaluate(cand)
        d = cand_cost - cur_cost
        if d < 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur, cur_cost = cand, cand_cost
            if cur_cost < best_cost:
                best, best_cost = cur.copy(), cur_cost
        T *= cfg["cooling"]
        history.append(best_cost)
    return dict(open_mask=np.asarray(best, bool), cost=float(best_cost),
                history=history, time=time.perf_counter() - t0,
                n_open=int(np.asarray(best).sum()))


# ============================================================ gradient-based
def weiszfeld(points_xy, weights, max_iters, tol):
    """Weiszfeld's algorithm: minimize sum_j w_j ||p - a_j|| (the Weber point).

    A classic gradient/fixed-point method for the convex continuous-location
    problem; each step moves toward the inverse-distance-weighted centroid.
    """
    w = np.asarray(weights, float)
    P = np.asarray(points_xy, float)
    p = np.average(P, axis=0, weights=w)                # weighted-centroid start
    for _ in range(max_iters):
        d = np.sqrt(((P - p) ** 2).sum(axis=1))
        d = np.maximum(d, 1e-9)                         # avoid divide-by-zero
        wnew = w / d
        p_new = (P * wnew[:, None]).sum(axis=0) / wnew.sum()
        if np.linalg.norm(p_new - p) < tol:
            p = p_new
            break
        p = p_new
    total = float((w * np.sqrt(((P - p) ** 2).sum(axis=1))).sum())
    return p, total


def _haversine_transport(inst, fac_lat, fac_lon):
    """Transport dollars for continuous facilities, using the SAME haversine x
    circuity x rate metric as the discrete network so the comparison is exact.
    Each zone is served by its nearest facility (single-source)."""
    fac_lat = np.atleast_1d(np.asarray(fac_lat, float))
    fac_lon = np.atleast_1d(np.asarray(fac_lon, float))
    D = np.vstack([geo.road_miles(fac_lat[k], fac_lon[k], inst["zlat"], inst["zlon"])
                   for k in range(len(fac_lat))])        # p x n road miles
    return float(inst["rate"] * (inst["demand"] * D.min(axis=0)).sum())


def continuous_center_of_gravity(inst):
    """Single-facility center of gravity (gradient-based). Returns lat/lon, cost."""
    x, y, lat0, lon0 = geo.project_miles(inst["zlat"], inst["zlon"])
    P = np.column_stack([x, y])
    p, ton_miles = weiszfeld(P, inst["demand"], 500, 1e-6)
    lat, lon = geo.unproject_miles(p[0], p[1], lat0, lon0)
    return dict(lat=float(lat), lon=float(lon), ton_miles=ton_miles,
                transport_cost=_haversine_transport(inst, lat, lon))


def _kpp_seed(A, w, p, rng):
    """Weighted k-means++ seeding: spread initial facilities by demand-distance."""
    n = len(A)
    first = rng.choice(n, p=w / w.sum())
    idx = [first]
    d2 = ((A - A[first]) ** 2).sum(axis=1)
    for _ in range(1, p):
        prob = w * d2
        s = prob.sum()
        nxt = rng.choice(n, p=prob / s) if s > 0 else rng.choice(n)
        idx.append(nxt)
        d2 = np.minimum(d2, ((A - A[nxt]) ** 2).sum(axis=1))
    return A[idx].astype(float)


def continuous_p_median(inst, p, rng, restarts=40):
    """Multi-facility continuous location via alternating assignment + Weiszfeld.

    Cooper's location-allocation loop: (i) assign each zone to its nearest
    facility, (ii) re-place each facility at the Weiszfeld point of its cluster.
    Restarted from many weighted k-means++ seeds; best total distance kept.
    Capacity and delivery-time limits are ignored, so this is the continuous
    lower envelope the discrete network is measured against.
    """
    xj, yj, lat0, lon0 = geo.project_miles(inst["zlat"], inst["zlon"])
    A = np.column_stack([xj, yj])
    w = inst["demand"]
    best = None
    for _ in range(restarts):
        F = _kpp_seed(A, w, p, rng)
        assign = None
        for _ in range(50):
            d = np.sqrt(((A[:, None, :] - F[None, :, :]) ** 2).sum(axis=2))  # n x p
            new_assign = d.argmin(axis=1)
            if assign is not None and np.array_equal(new_assign, assign):
                break
            assign = new_assign
            for k in range(p):
                idx = np.where(assign == k)[0]
                if len(idx):
                    F[k], _ = weiszfeld(A[idx], w[idx], 200, 1e-6)
        d = np.sqrt(((A[:, None, :] - F[None, :, :]) ** 2).sum(axis=2))
        total = float((w * d.min(axis=1)).sum())
        if best is None or total < best["ton_miles"]:
            lat, lon = geo.unproject_miles(F[:, 0], F[:, 1], lat0, lon0)
            best = dict(ton_miles=total, lat=lat, lon=lon, assign=assign.copy())
    # cost with the exact haversine metric (comparable to the discrete network)
    best["transport_cost"] = _haversine_transport(inst, best["lat"], best["lon"])
    return best
