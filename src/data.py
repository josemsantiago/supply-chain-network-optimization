"""Build the optimization instance from real U.S. Census data + BTS constants.

Demand zones  : the 48 contiguous states + DC, each located at its real Census
                2020 population center and sized by real 2020 population.
Candidate DCs : a 14-state subset (config.CANDIDATE_FIPS), also at real Census
                population centers.
Costs         : per-ton delivery cost = BTS freight rate ($/ton-mile) x road miles
                (great-circle x circuity). A delivery-time cap forbids arcs longer
                than the service promise.
"""
from __future__ import annotations
import csv
import numpy as np
import pandas as pd
import config as C
import geo


# ------------------------------------------------------------------ BTS constants
def load_bts():
    """Return (rate $/ton-mile, miles_per_day) and write a documented CSV.

    The freight rate is anchored to the BTS average freight revenue per ton-mile
    for for-hire trucking; the daily range follows FMCSA hours-of-service. Both
    are written to data/bts_rates.csv so the report's data table is reproducible.
    """
    rate = C.FREIGHT_COST_PER_TON_MILE
    mpd = C.MILES_PER_DRIVING_DAY
    rows = [
        ["variable", "value", "units", "source"],
        ["freight_cost_per_ton_mile", rate, "USD/ton-mile",
         "BTS National Transportation Statistics Table 3-21, for-hire (LTL) truck, 2007"],
        ["miles_per_driving_day", mpd, "miles/day",
         "FMCSA hours-of-service 11 h x ~50 mph effective (below BTS ~55 mph free-flow)"],
        ["circuity_factor", C.CIRCUITY_FACTOR, "ratio",
         "road-network detour vs great-circle (standard 1.2)"],
    ]
    with open(C.BTS_RATES_CSV, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    return rate, mpd


# ------------------------------------------------------------------ Census load
def _load_centers():
    """Read the cached Census 2020 centers-of-population file into a DataFrame."""
    df = pd.read_csv(C.CENPOP_CSV, dtype={"STATEFP": str})
    df["STATEFP"] = df["STATEFP"].str.zfill(2)
    df = df[~df["STATEFP"].isin(C.EXCLUDE_FIPS)].copy()
    df["POPULATION"] = df["POPULATION"].astype(float)
    df["LATITUDE"] = df["LATITUDE"].astype(float)
    df["LONGITUDE"] = df["LONGITUDE"].astype(float)
    df = df.sort_values("STATEFP").reset_index(drop=True)
    return df


# ------------------------------------------------------------------ instance
def build_instance(candidate_fips=None, total_demand=None):
    """Assemble every array the models need. Returns a dict ('instance').

    candidate_fips : list of state FIPS to use as candidate DC sites; None uses
                     config.CANDIDATE_FIPS; "ALL" makes every demand zone a site
                     (used for the scaling experiment).
    total_demand   : override total tons/year (None uses config default).
    """
    rate, mpd = load_bts()
    df = _load_centers()

    # Demand zones (all contiguous states + DC)
    zones = df.reset_index(drop=True)
    pop = zones["POPULATION"].to_numpy()
    T = C.TOTAL_DEMAND_TONS if total_demand is None else total_demand
    demand = T * pop / pop.sum()                           # tons/yr, pop-weighted
    zlat = zones["LATITUDE"].to_numpy()
    zlon = zones["LONGITUDE"].to_numpy()
    zname = zones["STNAME"].tolist()
    zfips = zones["STATEFP"].tolist()

    # Candidate DC sites (subset of the same real centroids)
    if candidate_fips == "ALL":
        cfips_sel = zfips
    elif candidate_fips is None:
        cfips_sel = C.CANDIDATE_FIPS
    else:
        cfips_sel = candidate_fips
    cand = df[df["STATEFP"].isin(cfips_sel)].reset_index(drop=True)
    clat = cand["LATITUDE"].to_numpy()
    clon = cand["LONGITUDE"].to_numpy()
    cname = cand["STNAME"].tolist()
    cfips = cand["STATEFP"].tolist()

    m, n = len(cname), len(zname)                            # DCs, zones

    # Road-distance and per-ton cost matrices (m x n)
    dist = np.zeros((m, n))
    for i in range(m):
        dist[i] = geo.road_miles(clat[i], clon[i], zlat, zlon)
    cost = rate * dist                                       # $/ton on arc (i,j)

    # Delivery-time feasibility mask: arc allowed iff within the service promise
    allowed = dist <= C.MAX_SERVICE_MILES

    cap = np.full(m, C.DC_CAPACITY_TONS)
    fixed = np.full(m, C.DC_FIXED_COST)

    inst = dict(
        rate=rate, miles_per_day=mpd,
        zname=zname, zfips=zfips, zlat=zlat, zlon=zlon, pop=pop, demand=demand,
        cname=cname, cfips=cfips, clat=clat, clon=clon,
        cap=cap, fixed=fixed, dist=dist, cost=cost, allowed=allowed,
        m=m, n=n,
    )
    return inst


def coverage_report(inst):
    """Diagnostic: nearest candidate distance per zone (feasibility check)."""
    dist, allowed = inst["dist"], inst["allowed"]
    nearest = dist.min(axis=0)
    reachable = allowed.any(axis=0)
    worst = int(np.argmax(nearest))
    return dict(
        max_nearest=float(nearest.max()),
        worst_zone=inst["zname"][worst],
        n_unreachable=int((~reachable).sum()),
        unreachable=[inst["zname"][j] for j in range(inst["n"]) if not reachable[j]],
        total_demand=float(inst["demand"].sum()),
        total_capacity=float(inst["cap"].sum()),
    )


if __name__ == "__main__":
    inst = build_instance()
    print(f"zones={inst['n']}  candidates={inst['m']}  rate=${inst['rate']}/ton-mi")
    print("coverage:", coverage_report(inst))
