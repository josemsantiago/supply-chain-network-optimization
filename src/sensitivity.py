"""Sensitivity analysis and cross-model scaling experiment.

Everything here re-solves the MILP under a perturbed instance so we can trace how
the optimal network responds to the business parameters: how many DCs to open,
the freight rate, the delivery-time promise, DC capacity, and demand growth.
Shadow prices come from the transportation LP's duals.
"""
from __future__ import annotations
import numpy as np
import config as C
import data
import models


def _clone(inst, **over):
    d = dict(inst)
    d.update(over)
    return d


# ------------------------------------------------------- how many DCs? (U-curve)
def dc_count_sweep(inst):
    rows = []
    for p in range(1, inst["m"] + 1):
        r = models.solve_milp(inst, force_p=p)
        if r["cost"] >= models.BIG or r.get("y") is None:
            rows.append(dict(p=p, feasible=False, total=None, fixed=None, transport=None))
        else:
            rows.append(dict(p=p, feasible=True, total=r["cost"],
                             fixed=r["fixed_cost"], transport=r["transport_cost"]))
    feas = [r for r in rows if r["feasible"]]
    best = min(feas, key=lambda r: r["total"])
    return dict(rows=rows, best_p=best["p"], best_total=best["total"])


# ------------------------------------------------------- freight-rate sensitivity
def rate_sweep(inst):
    rows = []
    base_cost = inst["cost"]
    for mult in C.RATE_MULTIPLIERS:
        r = models.solve_milp(_clone(inst, cost=base_cost * mult))
        avg_mi = _avg_delivery_miles(inst, r)
        rows.append(dict(mult=mult, rate=inst["rate"] * mult, n_open=r["n_open"],
                         total=r["cost"], transport=r["transport_cost"],
                         fixed=r["fixed_cost"], avg_miles=avg_mi))
    return rows


# ------------------------------------------------------- delivery-time sensitivity
def delivery_sweep(inst):
    rows = []
    for days in C.DELIVERY_DAYS_GRID:
        max_mi = C.MILES_PER_DRIVING_DAY * days
        allowed = inst["dist"] <= max_mi
        r = models.solve_milp(_clone(inst, allowed=allowed))
        feasible = r["cost"] < models.BIG and r.get("y") is not None
        rows.append(dict(days=days, max_miles=max_mi, feasible=feasible,
                         n_open=r["n_open"] if feasible else None,
                         total=r["cost"] if feasible else None,
                         transport=r["transport_cost"] if feasible else None))
    return rows


# ------------------------------------------------------- capacity sensitivity
def capacity_sweep(inst, factors=(0.5, 0.75, 1.0, 1.5, 2.0)):
    rows = []
    for f in factors:
        r = models.solve_milp(_clone(inst, cap=inst["cap"] * f))
        feasible = r["cost"] < models.BIG and r.get("y") is not None
        util = None
        if feasible:
            used = r["x"].sum(axis=1)[r["open_mask"]]
            util = float((used / (inst["cap"][r["open_mask"]] * f)).mean())
        rows.append(dict(factor=f, cap=C.DC_CAPACITY_TONS * f, feasible=feasible,
                         n_open=r["n_open"] if feasible else None,
                         total=r["cost"] if feasible else None,
                         mean_util=util))
    return rows


# ------------------------------------------------------- demand-growth sensitivity
def demand_sweep(inst):
    rows = []
    for g in C.DEMAND_GROWTH:
        r = models.solve_milp(_clone(inst, demand=inst["demand"] * (1 + g)))
        feasible = r["cost"] < models.BIG and r.get("y") is not None
        rows.append(dict(growth=g, feasible=feasible,
                         n_open=r["n_open"] if feasible else None,
                         total=r["cost"] if feasible else None))
    return rows


# ------------------------------------------------------- shadow prices (marginals)
def shadow_prices(inst, open_mask, eps=1.0):
    """Marginal cost of one more ton to each zone, and of one more ton of DC
    capacity, computed by re-optimizing the transportation LP under a unit
    perturbation. Finite differencing is used instead of the LP duals because
    the transportation polytope is highly degenerate, so its duals are not
    unique and do not match the economic marginal (e.g., an on-site-served zone
    can carry a nonzero degenerate dual); the perturbation marginal is exact.
    """
    base = models.solve_transport_lp(inst, open_mask)["cost"]
    n, m = inst["n"], inst["m"]
    # demand marginals
    dd = np.zeros(n)
    for j in range(n):
        d2 = inst["demand"].copy(); d2[j] += eps
        c2 = models.solve_transport_lp(_clone(inst, demand=d2), open_mask)["cost"]
        dd[j] = (c2 - base) / eps
    order = np.argsort(dd)[::-1]
    top = [dict(state=inst["zname"][j], demand=float(inst["demand"][j]),
                marginal=float(dd[j])) for j in order[:10]]
    # capacity marginals (per open DC): value of one more ton of capacity
    cap_shadow = []
    for i in range(m):
        if not open_mask[i]:
            continue
        cap2 = inst["cap"].copy(); cap2[i] += eps
        c2 = models.solve_transport_lp(_clone(inst, cap=cap2), open_mask)["cost"]
        s = (c2 - base) / eps
        cap_shadow.append(dict(dc=inst["cname"][i], shadow=float(s)))
    cap_binding = [c for c in cap_shadow if c["shadow"] < -1e-6]
    return dict(top_demand=top, cap_binding=cap_binding, cap_shadow=cap_shadow,
                mean_marginal=float(dd.mean()))


# ------------------------------------------------------- helpers
def _avg_delivery_miles(inst, milp_res):
    """Demand-weighted average shipped distance under a solution."""
    if milp_res.get("x") is None:
        return None
    x = milp_res["x"]
    tot = x.sum()
    return float((x * inst["dist"]).sum() / tot) if tot > 0 else None


# ------------------------------------------------------- scaling / method race
def scaling_experiment(inst_big, rng_seed):
    """MILP vs GA vs SA on the larger all-states-candidate instance."""
    mi = models.solve_milp(inst_big, time_limit=120)
    ga = models.genetic_algorithm(inst_big, np.random.default_rng(rng_seed),
                                  dict(pop=C.GA_POP_SIZE, gen=C.GA_GENERATIONS,
                                       mut=C.GA_MUTATION_RATE, tour=C.GA_TOURNAMENT,
                                       elite=C.GA_ELITE))
    sa = models.simulated_annealing(inst_big, np.random.default_rng(rng_seed),
                                    dict(iters=C.SA_ITERATIONS, T0=C.SA_T0,
                                         cooling=C.SA_COOLING))
    def gap(v):
        return 100.0 * (v - mi["cost"]) / mi["cost"]
    return dict(
        m=inst_big["m"],
        milp=dict(cost=mi["cost"], n_open=mi["n_open"], time=mi["time"]),
        ga=dict(cost=ga["cost"], n_open=ga["n_open"], time=ga["time"], gap=gap(ga["cost"])),
        sa=dict(cost=sa["cost"], n_open=sa["n_open"], time=sa["time"], gap=gap(sa["cost"])),
        ga_history=ga["history"], sa_history=sa["history"],
    )
