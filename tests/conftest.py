from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from mappatrakka.config import CleaningConfig

T0 = datetime(2024, 5, 1, 8, 0, tzinfo=timezone.utc)


def straight_line(lat0, lon0, lat1, lon1, n=50, start=T0, step_s=2.0, timed=True):
    """n evenly spaced (lat, lon, time) points from one spot to another."""
    out = []
    for i in range(n):
        f = i / (n - 1)
        t = start + timedelta(seconds=i * step_s) if timed else None
        out.append((lat0 + (lat1 - lat0) * f, lon0 + (lon1 - lon0) * f, t))
    return out


def as_arrays(points):
    lat = np.array([p[0] for p in points], dtype=float)
    lon = np.array([p[1] for p in points], dtype=float)
    t = np.array([p[2].timestamp() if p[2] else np.nan for p in points], dtype=float)
    return lat, lon, t


@pytest.fixture
def cleaning():
    return CleaningConfig()
