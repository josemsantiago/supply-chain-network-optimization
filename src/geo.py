"""Geography: great-circle distance, road-distance approximation, projection.

Distances are great-circle (haversine) miles inflated by a circuity factor to
approximate driving miles. For the gradient-based (Weiszfeld) method we also
project lat/lon onto a local equirectangular plane in miles, which preserves
distances well over a continental extent and lets us run the center-of-gravity
algorithm in ordinary 2-D Euclidean space.
"""
from __future__ import annotations
import numpy as np
import config as C


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles. Scalar or broadcastable arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * C.EARTH_RADIUS_MI * np.arcsin(np.sqrt(a))


def road_miles(lat1, lon1, lat2, lon2):
    """Approximate driving miles = great-circle x circuity factor."""
    return C.CIRCUITY_FACTOR * haversine_miles(lat1, lon1, lat2, lon2)


def travel_days(road_mi):
    """Delivery time in days given the daily driving range."""
    return road_mi / C.MILES_PER_DRIVING_DAY


def project_miles(lat, lon, lat0=None, lon0=None):
    """Equirectangular projection to (x, y) miles about a reference point.

    x = R * (lon-lon0) * cos(lat0),  y = R * (lat-lat0), with R in miles and
    angles in radians. Accurate to a few percent over the contiguous U.S.
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    if lat0 is None:
        lat0 = float(np.mean(lat))
    if lon0 is None:
        lon0 = float(np.mean(lon))
    rlat0 = np.radians(lat0)
    x = np.radians(lon - lon0) * np.cos(rlat0) * C.EARTH_RADIUS_MI
    y = np.radians(lat - lat0) * C.EARTH_RADIUS_MI
    return x, y, lat0, lon0


def unproject_miles(x, y, lat0, lon0):
    """Inverse of project_miles: (x, y) miles back to (lat, lon) degrees."""
    rlat0 = np.radians(lat0)
    lat = lat0 + np.degrees(y / C.EARTH_RADIUS_MI)
    lon = lon0 + np.degrees(x / (C.EARTH_RADIUS_MI * np.cos(rlat0)))
    return lat, lon
