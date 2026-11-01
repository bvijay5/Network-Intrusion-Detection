"""
Shared test fixtures.

Tests must run WITHOUT the real dataset (e.g. on a fresh clone or in CI), so we
generate a small synthetic table with the UNSW-NB15 schema. Its values are
made up; it is only used to check that the code behaves correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config


def make_synthetic(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Fake UNSW-NB15-shaped data where attacks have a higher 'sttl' and 'sbytes'."""
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, size=n)
    df = pd.DataFrame({"id": np.arange(1, n + 1)})
    for col in config.NUMERIC_FEATURES:
        df[col] = rng.exponential(scale=10.0, size=n).round(3)
    df["sttl"] = np.where(label == 1, rng.choice([62, 254], n), rng.choice([31, 62], n))
    df["sbytes"] = df["sbytes"] * (1 + 4 * label)
    df["proto"] = rng.choice(["tcp", "udp", "arp"], size=n)
    df["service"] = rng.choice(["-", "dns", "http"], size=n)
    df["state"] = rng.choice(["FIN", "INT", "CON"], size=n)
    df["attack_cat"] = np.where(label == 1, rng.choice(["Generic", "Exploits"], n), "Normal")
    df["label"] = label
    return df


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic()


@pytest.fixture
def synthetic_csv(tmp_path, synthetic_df):
    path = tmp_path / "traffic.csv"
    synthetic_df.to_csv(path, index=False)
    return path
