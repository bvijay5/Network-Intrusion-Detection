"""
Create the small demo file used by the dashboard (data/demo/demo_traffic.csv).

It is a 500-row excerpt of the official UNSW-NB15 TESTING file, with every
attack category represented. It keeps the label columns so the dashboard can
compare predictions with the truth, but the model never sees those columns.

Run with:   python -m src.make_demo
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.data_loader import read_csv_safely

DEMO_ROWS = 500
MIN_PER_CATEGORY = 10


def make_demo(n_rows: int = DEMO_ROWS) -> pd.DataFrame:
    test = read_csv_safely(config.TEST_FILE)
    rs = config.RANDOM_STATE
    # A few rows from every category, so rare attacks (e.g. Worms) appear too.
    # (every category in the official test file has at least 44 rows)
    guaranteed = test.groupby("attack_cat").sample(n=MIN_PER_CATEGORY, random_state=rs)
    rest = test.drop(guaranteed.index).sample(n_rows - len(guaranteed), random_state=rs)
    demo = pd.concat([guaranteed, rest]).sample(frac=1, random_state=rs)
    config.DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    demo.to_csv(config.DEMO_FILE, index=False)
    return demo


if __name__ == "__main__":
    demo = make_demo()
    print(f"Wrote {len(demo)} rows to {config.DEMO_FILE.relative_to(config.PROJECT_ROOT)}")
    print(demo["attack_cat"].value_counts().to_string())
