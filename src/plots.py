"""Publication-quality figures for the report. Pure matplotlib (no basemap):
points are plotted in lon/lat with a latitude-corrected aspect ratio so the
contiguous U.S. reads correctly.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

INK = "#1b2a41"        # primary ink
DEMAND = "#9db4c0"     # demand zones
OPEN = "#c0392b"       # opened DCs
CAND = "#b8b8b8"       # unopened candidates
GRAD = "#1f7a4d"       # gradient / continuous
BAR = ["#2c6fbb", "#c0392b", "#e0a458", "#4d9078", "#7d5ba6"]
_LAT0 = 39.0


def _mapaxes(ax):
    ax.set_aspect(1.0 / np.cos(np.radians(_LAT0)))
    ax.set_xlim(-125, -66); ax.set_ylim(24, 50)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")


# =============================================================== fig 1: network
def fig_network(inst, milp, path):
    x, xflows = milp["open_mask"], milp["x"]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    _mapaxes(ax)
    # demand zones sized by demand
    s = 30 + 900 * inst["demand"] / inst["demand"].max()
    ax.scatter(inst["zlon"], inst["zlat"], s=s, c=DEMAND, alpha=0.55,
               edgecolors="white", linewidths=0.5, zorder=2, label="Demand zone (state)")
    # unopened candidates
    closed = [i for i in range(inst["m"]) if not x[i]]
    ax.scatter(inst["clon"][closed], inst["clat"][closed], marker="s", s=45,
               c=CAND, edgecolors="k", linewidths=0.4, zorder=3, label="Candidate site (unused)")
    # assignment flows: each zone -> its dominant serving DC
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    open_idx = [i for i in range(inst["m"]) if x[i]]
    color_of = {i: palette[k % 10] for k, i in enumerate(open_idx)}
    for j in range(inst["n"]):
        i = int(np.argmax(xflows[:, j]))
        if xflows[i, j] <= 0:
            continue
        ax.plot([inst["clon"][i], inst["zlon"][j]], [inst["clat"][i], inst["zlat"][j]],
                "-", color=color_of[i], lw=0.6, alpha=0.5, zorder=1)
    # opened DCs
    ax.scatter(inst["clon"][x], inst["clat"][x], marker="*", s=340, c=OPEN,
               edgecolors="k", linewidths=0.6, zorder=5, label="Opened DC")
    for i in open_idx:
        ax.annotate(inst["cname"][i], (inst["clon"][i], inst["clat"][i]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8, weight="bold")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title("Optimal distribution network (MILP): opened DCs and demand assignment")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# =============================================================== fig 2: U-curve
def fig_dc_curve(sweep, path):
    rows = [r for r in sweep["rows"] if r["feasible"]]
    p = [r["p"] for r in rows]
    tot = np.array([r["total"] for r in rows]) / 1e6
    fx = np.array([r["fixed"] for r in rows]) / 1e6
    tr = np.array([r["transport"] for r in rows]) / 1e6
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(p, tr, "o-", color=BAR[0], label="Transport cost")
    ax.plot(p, fx, "s-", color=BAR[2], label="Fixed cost")
    ax.plot(p, tot, "^-", color=OPEN, lw=2.2, label="Total cost")
    bp = sweep["best_p"]
    ax.axvline(bp, color="k", ls="--", lw=1, alpha=0.6)
    ax.annotate(f"optimum p = {bp}", (bp, tot.min()), textcoords="offset points",
                xytext=(8, 10), fontsize=9, weight="bold")
    ax.set_xlabel("Number of distribution centers opened")
    ax.set_ylabel("Annual cost ($ millions)")
    ax.set_title("Cost trade-off: transport vs. facilities")
    ax.legend(); fig.tight_layout(); fig.savefig(path); plt.close(fig)


# =============================================================== fig 3: methods
def fig_methods(cmp_primary, scaling, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    # A: total cost on primary instance
    names = [c["name"] for c in cmp_primary]
    costs = [c["cost"] / 1e6 for c in cmp_primary]
    cols = [BAR[0], OPEN, BAR[3], BAR[2], GRAD][:len(names)]
    ax1.bar(names, costs, color=cols, edgecolor="k", linewidth=0.4)
    ax1.set_ylabel("Objective ($ millions)")
    ax1.set_title("A. Objective by method (primary, 14 sites)")
    ax1.set_ylim(min(costs) * 0.9, max(costs) * 1.03)
    for i, v in enumerate(costs):
        ax1.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax1.tick_params(axis="x", labelrotation=20)
    # B: runtime on scaling instance (log) with gap labels
    meth = ["MILP", "SA", "GA"]
    times = [scaling["milp"]["time"], scaling["sa"]["time"], scaling["ga"]["time"]]
    gaps = [0.0, scaling["sa"]["gap"], scaling["ga"]["gap"]]
    b = ax2.bar(meth, times, color=[OPEN, BAR[3], BAR[2]], edgecolor="k", linewidth=0.4)
    ax2.set_yscale("log"); ax2.set_ylabel("Solve time (s, log scale)")
    ax2.set_title(f"B. Speed on scaling instance ({scaling['m']} sites)")
    for rect, g in zip(b, gaps):
        ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                 f"gap {g:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# =============================================================== fig 4: sensitivity
def fig_sensitivity(rate_rows, demand_rows, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    # A: freight rate -> #DC and total cost
    mult = [r["mult"] for r in rate_rows]
    nopen = [r["n_open"] for r in rate_rows]
    tot = [r["total"] / 1e6 for r in rate_rows]
    ax1.plot(mult, nopen, "o-", color=OPEN, label="DCs opened")
    ax1.set_xlabel("Freight-rate multiplier (x BTS rate)")
    ax1.set_ylabel("DCs opened", color=OPEN)
    ax1.set_title("A. Response to freight cost")
    axb = ax1.twinx(); axb.plot(mult, tot, "s--", color=BAR[0], label="Total cost")
    axb.set_ylabel("Total cost ($M)", color=BAR[0]); axb.grid(False)
    # B: demand growth -> #DC and total cost
    g = [100 * r["growth"] for r in demand_rows if r["feasible"]]
    nd = [r["n_open"] for r in demand_rows if r["feasible"]]
    td = [r["total"] / 1e6 for r in demand_rows if r["feasible"]]
    ax2.plot(g, nd, "o-", color=OPEN, label="DCs opened")
    ax2.set_xlabel("Demand growth (%)")
    ax2.set_ylabel("DCs opened", color=OPEN)
    ax2.set_title("B. Response to demand growth")
    ax2.set_yticks(sorted(set(nd)))
    axc = ax2.twinx(); axc.plot(g, td, "s--", color=BAR[0], label="Total cost")
    axc.set_ylabel("Total cost ($M)", color=BAR[0]); axc.grid(False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# =============================================================== fig 5: gradient
def fig_gradient(inst, milp, cog, pmed, path):
    fig, ax = plt.subplots(figsize=(9, 5.6))
    _mapaxes(ax)
    s = 20 + 700 * inst["demand"] / inst["demand"].max()
    ax.scatter(inst["zlon"], inst["zlat"], s=s, c=DEMAND, alpha=0.5,
               edgecolors="white", linewidths=0.4, zorder=2, label="Demand zone")
    ax.scatter(inst["clon"][milp["open_mask"]], inst["clat"][milp["open_mask"]],
               marker="*", s=340, c=OPEN, edgecolors="k", linewidths=0.6, zorder=5,
               label="Discrete MILP DC")
    ax.scatter(pmed["lon"], pmed["lat"], marker="D", s=90, c=GRAD, edgecolors="k",
               linewidths=0.6, zorder=6, label="Continuous Weiszfeld optimum")
    ax.scatter([cog["lon"]], [cog["lat"]], marker="P", s=220, c="#8e44ad",
               edgecolors="k", linewidths=0.6, zorder=7, label="Single center of gravity")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title("Gradient-based continuous optimum vs. discrete MILP network")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# =============================================================== fig 6: convergence + duals
def fig_convergence(ga_hist, sa_hist, shadow, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    ax1.plot(np.arange(len(ga_hist)), np.array(ga_hist) / 1e6, "-", color=BAR[2], label="GA best")
    # SA history is long; subsample for a clean line
    sh = np.array(sa_hist) / 1e6
    xs = np.linspace(0, len(sh) - 1, min(len(sh), 400)).astype(int)
    ax1.plot(np.linspace(0, len(ga_hist) - 1, len(xs)), sh[xs], "-", color=BAR[3], label="SA best")
    ax1.set_xlabel("Iteration (rescaled)"); ax1.set_ylabel("Best cost ($M)")
    ax1.set_title("A. Metaheuristic convergence (scaling instance)")
    ax1.legend()
    top = shadow["top_demand"][:8]
    names = [t["state"] for t in top][::-1]
    vals = [t["marginal"] for t in top][::-1]
    ax2.barh(names, vals, color=BAR[0], edgecolor="k", linewidth=0.4)
    ax2.set_xlabel("Marginal delivery cost ($/ton)")
    ax2.set_title("B. Shadow prices: costliest states to serve")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
