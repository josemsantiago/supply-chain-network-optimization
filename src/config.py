"""All tunable constants for the supply-chain network-design study.

Every number a reviewer might question lives here, with a one-line justification
and (where relevant) its real-world source. Nothing downstream hard-codes a
magic value; everything reads from this module so the whole study is one edit
away from a sensitivity run.
"""
from __future__ import annotations

# --------------------------------------------------------------- reproducibility
SEED = 20260712                      # master seed (report date 2026-07-12)

# --------------------------------------------------------------- data files
CENPOP_CSV = "../data/CenPop2020_Mean_ST.csv"   # cached U.S. Census 2020 centers of population
BTS_RATES_CSV = "../data/bts_rates.csv"          # BTS-sourced freight constants (written by data.py)

# --------------------------------------------------------------- geography scope
# We model the 48 contiguous states + DC. Alaska (02), Hawaii (15), and
# Puerto Rico (72) are dropped because a road-freight (great-circle x circuity)
# distance model is not meaningful across oceans.
EXCLUDE_FIPS = {"02", "15", "72"}

# Candidate distribution-center sites: the population centroids of 14 states that
# are genuine national logistics hubs, chosen to blanket the country (two on the
# West coast, one Southwest, one Mountain, one South-Central, three Midwest, four
# Southeast, two Northeast). Each candidate's coordinate is the real Census 2020
# population center of that state, so every coordinate in the study is Census data.
CANDIDATE_FIPS = [
    "06",  # California       (West)
    "53",  # Washington       (Pacific NW)
    "04",  # Arizona          (Southwest)
    "08",  # Colorado         (Mountain)
    "48",  # Texas            (South-Central)
    "29",  # Missouri         (Midwest, central)
    "17",  # Illinois         (Midwest)
    "39",  # Ohio             (Midwest, east)
    "47",  # Tennessee        (Southeast, central)
    "13",  # Georgia          (Southeast)
    "37",  # North Carolina   (Mid-Atlantic)
    "12",  # Florida          (Southeast, tip)
    "42",  # Pennsylvania     (Northeast)
    "34",  # New Jersey       (Northeast, coast)
]

# --------------------------------------------------------------- demand model
# The company's total annual outbound distribution volume (tons/year), allocated
# to each demand zone in proportion to its 2020 Census population. This scalar
# sets the SIZE of the problem; because demand scales linearly, the optimal
# facility LOCATIONS are invariant to it (only capacities/counts scale) -- a fact
# we exploit in the sensitivity analysis. Chosen to represent a mid-size regional
# distributor.
TOTAL_DEMAND_TONS = 1_000_000.0

# --------------------------------------------------------------- facility model
DC_CAPACITY_TONS = 300_000.0     # throughput ceiling of one DC (tons/year)
DC_FIXED_COST = 3_000_000.0      # annualized fixed cost to open/operate one DC ($/yr)

# --------------------------------------------------------------- transport model
# Freight cost per ton-mile. Anchored to the BTS average freight revenue per
# ton-mile for for-hire (general-freight, mostly LTL) trucking: 16.54 cents/
# ton-mile, BTS National Transportation Statistics Table 3-21 (2007, the most
# recent year of that discontinued series). The 2007 vintage is disclosed in the
# report and covered by the freight-rate sensitivity sweep, which spans 0.5x-2x
# to bracket current price levels.
FREIGHT_COST_PER_TON_MILE = 0.1654   # $/ton-mile (BTS NTS Table 3-21, 2007)

# Great-circle distances understate road distances; multiply by a circuity factor
# to approximate driving miles. 1.20 is a standard, widely-cited road-network
# detour factor for U.S. inter-city truck routing.
CIRCUITY_FACTOR = 1.20

# --------------------------------------------------------------- delivery-time model
# A truck's usable daily range under FMCSA hours-of-service (11 h driving) at a
# realistic average highway speed. 550 mi/day is a conservative, defensible day.
MILES_PER_DRIVING_DAY = 550.0
MAX_DELIVERY_DAYS = 2.0                       # service-level promise: <= 2 days
MAX_SERVICE_MILES = MILES_PER_DRIVING_DAY * MAX_DELIVERY_DAYS   # 1,100 road-miles

# --------------------------------------------------------------- heuristics
GA_POP_SIZE = 40
GA_GENERATIONS = 60
GA_MUTATION_RATE = 0.08
GA_TOURNAMENT = 3
GA_ELITE = 2

SA_ITERATIONS = 4000
SA_T0 = 5.0e6         # initial temperature (dollars scale)
SA_COOLING = 0.9975   # geometric cooling factor

# --------------------------------------------------------------- gradient method
WEISZFELD_MAX_ITERS = 500
WEISZFELD_TOL = 1e-6

# --------------------------------------------------------------- sensitivity grids
RATE_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]          # x freight rate
DELIVERY_DAYS_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0]          # service-day promise
DEMAND_GROWTH = [0.0, 0.10, 0.25, 0.50]                      # +% demand
EARTH_RADIUS_MI = 3958.7561                                  # mean Earth radius (miles)
