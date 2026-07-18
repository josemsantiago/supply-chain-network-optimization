"""Orchestrator: build the instance, run every model and sensitivity, dump
results.json + CSV tables, and regenerate all figures. Deterministic under the
master seed. Run from src/:  python3 run_analysis.py
"""
from __future__ import annotations
import csv
import json
import time
import numpy as np

import config as C
import data
import geo
import models
import sensitivity as sens
import plots


def _num(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main():
    t_start = time.perf_counter()
    inst = data.build_instance()
    cov = data.coverage_report(inst)
    print(f"Instance: {inst['n']} demand zones, {inst['m']} candidate sites, "
          f"rate=${inst['rate']}/ton-mi, D_max={C.MAX_SERVICE_MILES:.0f} mi")

    # ---------------------------------------------------------------- models
    milp = models.solve_milp(inst)
    lp_strong = models.solve_milp(inst, relax=True, strong=True)
    lp_weak = models.solve_milp(inst, relax=True, strong=False)
    tlp = models.solve_transport_lp(inst, milp["open_mask"], want_duals=True)
    ga = models.genetic_algorithm(inst, np.random.default_rng(C.SEED),
                                  dict(pop=C.GA_POP_SIZE, gen=C.GA_GENERATIONS,
                                       mut=C.GA_MUTATION_RATE, tour=C.GA_TOURNAMENT,
                                       elite=C.GA_ELITE))
    sa = models.simulated_annealing(inst, np.random.default_rng(C.SEED),
                                    dict(iters=C.SA_ITERATIONS, T0=C.SA_T0,
                                         cooling=C.SA_COOLING))
    cog = models.continuous_center_of_gravity(inst)
    pmed = models.continuous_p_median(inst, milp["n_open"], np.random.default_rng(C.SEED))

    def gap(v):
        return 100.0 * (v - milp["cost"]) / milp["cost"]

    print(f"MILP optimum: ${milp['cost']:,.0f}  ({milp['n_open']} DCs, {milp['time']*1000:.1f} ms)")
    print(f"  opened: {[inst['cname'][i] for i in range(inst['m']) if milp['open_mask'][i]]}")
    print(f"LP relax strong=${lp_strong['cost']:,.0f} (gap {gap(lp_strong['cost']):.2f}%)  "
          f"weak=${lp_weak['cost']:,.0f} (gap {gap(lp_weak['cost']):.2f}%)")
    print(f"GA=${ga['cost']:,.0f} (gap {gap(ga['cost']):.3f}%)  "
          f"SA=${sa['cost']:,.0f} (gap {gap(sa['cost']):.3f}%)")
    print(f"Continuous p-median transport=${pmed['transport_cost']:,.0f} "
          f"({100*pmed['transport_cost']/milp['transport_cost']:.1f}% of discrete transport)")

    # ---------------------------------------------------------------- solution table
    used = milp["x"].sum(axis=1)
    sol_rows = []
    for i in range(inst["m"]):
        if milp["open_mask"][i]:
            served = np.where(milp["x"][i] > 1e-3)[0]
            avg_mi = float((milp["x"][i] * inst["dist"][i]).sum() / max(used[i], 1e-9))
            sol_rows.append([inst["cname"][i], f"{used[i]:,.0f}",
                             f"{100*used[i]/inst['cap'][i]:.1f}%", len(served),
                             f"{avg_mi:.0f}"])
    write_csv("../outputs/table_solution.csv",
              ["Distribution center", "Throughput (tons/yr)", "Utilization",
               "Zones served", "Avg. haul (mi)"], sol_rows)

    # ---------------------------------------------------------------- model comparison
    cmp_primary = [
        dict(name="LP relax (weak)", cost=lp_weak["cost"], n_open=lp_weak["n_open"],
             time=lp_weak["time"], gap=gap(lp_weak["cost"])),
        dict(name="MILP (exact)", cost=milp["cost"], n_open=milp["n_open"],
             time=milp["time"], gap=0.0),
        dict(name="Genetic alg.", cost=ga["cost"], n_open=ga["n_open"],
             time=ga["time"], gap=gap(ga["cost"])),
        dict(name="Sim. annealing", cost=sa["cost"], n_open=sa["n_open"],
             time=sa["time"], gap=gap(sa["cost"])),
    ]
    write_csv("../outputs/table_model_comparison.csv",
              ["Method", "Objective ($)", "DCs", "Solve time (s)", "Gap vs MILP (%)"],
              [[c["name"], f"{c['cost']:,.0f}",
                (f"{c['n_open']:.2f}" if isinstance(c["n_open"], float) else c["n_open"]),
                f"{c['time']:.3f}", f"{c['gap']:.2f}"] for c in cmp_primary])

    # ---------------------------------------------------------------- sensitivity
    sweep = sens.dc_count_sweep(inst)
    rate_rows = sens.rate_sweep(inst)
    deliv_rows = sens.delivery_sweep(inst)
    cap_rows = sens.capacity_sweep(inst)
    dem_rows = sens.demand_sweep(inst)
    shadow = sens.shadow_prices(inst, milp["open_mask"])
    print(f"Optimal #DCs (U-curve) = {sweep['best_p']} at ${sweep['best_total']:,.0f}")

    write_csv("../outputs/table_dc_sweep.csv",
              ["DCs (p)", "Total ($)", "Fixed ($)", "Transport ($)"],
              [[r["p"], f"{r['total']:,.0f}", f"{r['fixed']:,.0f}", f"{r['transport']:,.0f}"]
               for r in sweep["rows"] if r["feasible"]])
    write_csv("../outputs/table_rate_sensitivity.csv",
              ["Rate mult", "Rate ($/ton-mi)", "DCs", "Total ($)", "Transport ($)", "Avg haul (mi)"],
              [[r["mult"], f"{r['rate']:.4f}", r["n_open"], f"{r['total']:,.0f}",
                f"{r['transport']:,.0f}", f"{r['avg_miles']:.0f}"] for r in rate_rows])
    write_csv("../outputs/table_delivery_sensitivity.csv",
              ["Delivery days", "Max haul (mi)", "Feasible", "DCs", "Total ($)"],
              [[r["days"], f"{r['max_miles']:.0f}", r["feasible"],
                r["n_open"], (f"{r['total']:,.0f}" if r["total"] else "-")] for r in deliv_rows])
    write_csv("../outputs/table_shadow_prices.csv",
              ["State", "Demand (tons/yr)", "Marginal cost ($/ton)"],
              [[t["state"], f"{t['demand']:,.0f}", f"{t['marginal']:.2f}"]
               for t in shadow["top_demand"]])

    # ---------------------------------------------------------------- scaling experiment
    big = data.build_instance(candidate_fips="ALL")
    scal = sens.scaling_experiment(big, C.SEED)
    print(f"Scaling ({scal['m']} sites): MILP {scal['milp']['time']:.3f}s | "
          f"SA {scal['sa']['time']:.2f}s gap {scal['sa']['gap']:.2f}% | "
          f"GA {scal['ga']['time']:.2f}s gap {scal['ga']['gap']:.2f}%")

    # ---------------------------------------------------------------- data description
    write_csv("../outputs/table_data_description.csv",
              ["Source", "Variable", "Units", "Preprocessing"],
              [["U.S. Census Bureau, 2020 Centers of Population",
                "State population", "persons", "48 contiguous states + DC; AK/HI/PR dropped"],
               ["U.S. Census Bureau, 2020 Centers of Population",
                "Population-center latitude/longitude", "degrees", "used as zone and candidate coordinates"],
               ["Derived (Census pop x company volume)",
                "Zone demand", "tons/year", f"pop-weighted split of {C.TOTAL_DEMAND_TONS:,.0f} t/yr"],
               ["BTS NTS Table 3-21 (2007)",
                "Freight rate", "$/ton-mile", f"{inst['rate']} applied to road miles"],
               ["FMCSA HOS x BTS truck speed",
                "Daily driving range", "miles/day", f"{C.MILES_PER_DRIVING_DAY:.0f}; {C.MAX_DELIVERY_DAYS}-day service cap"],
               ["Great-circle x circuity",
                "Road distance", "miles", f"haversine x {C.CIRCUITY_FACTOR} detour factor"]])

    # ---------------------------------------------------------------- results.json
    R = dict(
        seed=C.SEED,
        instance=dict(n_zones=inst["n"], m_candidates=inst["m"], rate=inst["rate"],
                      total_demand=float(inst["demand"].sum()),
                      total_capacity=float(inst["cap"].sum()),
                      dc_capacity=C.DC_CAPACITY_TONS, dc_fixed=C.DC_FIXED_COST,
                      max_service_miles=C.MAX_SERVICE_MILES, coverage=cov),
        milp=dict(cost=milp["cost"], fixed=milp["fixed_cost"], transport=milp["transport_cost"],
                  n_open=milp["n_open"], time=milp["time"],
                  opened=[inst["cname"][i] for i in range(inst["m"]) if milp["open_mask"][i]]),
        lp_relaxation=dict(strong=dict(cost=lp_strong["cost"], gap=gap(lp_strong["cost"]),
                                       sum_y=lp_strong["n_open"]),
                           weak=dict(cost=lp_weak["cost"], gap=gap(lp_weak["cost"]),
                                     sum_y=lp_weak["n_open"])),
        heuristics=dict(ga=dict(cost=ga["cost"], n_open=ga["n_open"], time=ga["time"],
                                gap=gap(ga["cost"])),
                        sa=dict(cost=sa["cost"], n_open=sa["n_open"], time=sa["time"],
                                gap=gap(sa["cost"]))),
        gradient=dict(center_of_gravity=dict(lat=cog["lat"], lon=cog["lon"],
                                             transport=cog["transport_cost"]),
                      p_median=dict(p=milp["n_open"], transport=pmed["transport_cost"],
                                    pct_of_discrete=100*pmed["transport_cost"]/milp["transport_cost"],
                                    lat=[float(v) for v in pmed["lat"]],
                                    lon=[float(v) for v in pmed["lon"]])),
        dc_sweep=dict(best_p=sweep["best_p"], best_total=sweep["best_total"], rows=sweep["rows"]),
        rate_sensitivity=rate_rows,
        delivery_sensitivity=deliv_rows,
        capacity_sensitivity=cap_rows,
        demand_sensitivity=dem_rows,
        shadow_prices=shadow,
        scaling=scal,
        runtime_seconds=time.perf_counter() - t_start,
    )
    with open("../outputs/results.json", "w") as fh:
        json.dump(R, fh, indent=2, default=_num)

    # ---------------------------------------------------------------- figures
    plots.fig_network(inst, milp, "../figures/fig01_network.png")
    plots.fig_dc_curve(sweep, "../figures/fig02_dc_curve.png")
    plots.fig_methods(cmp_primary, scal, "../figures/fig03_methods.png")
    plots.fig_sensitivity(rate_rows, dem_rows, "../figures/fig04_sensitivity.png")
    plots.fig_gradient(inst, milp, cog, pmed, "../figures/fig05_gradient.png")
    plots.fig_convergence(scal["ga_history"], scal["sa_history"], shadow,
                          "../figures/fig06_convergence.png")

    print(f"Done in {R['runtime_seconds']:.1f}s. Wrote results.json, 6 tables, 6 figures.")


if __name__ == "__main__":
    main()
