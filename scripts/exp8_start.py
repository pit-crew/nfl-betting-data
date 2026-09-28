# Experiment 8: baseline and logistic regression when the data starts at a later season (video 1 split).
import pandas as pd

from features import load
from logreg import per_game, run

if __name__ == "__main__":
    df = load()
    rows = []
    for start in range(1966, 2021):
        d = df[df["schedule_season"] >= start].reset_index(drop=True)
        data, cols = per_game(d)
        stats, _ = run(data, cols, 2023, True)
        rows.append({"start": start, "val rows": stats["val rows"],
                     "baseline": stats["baseline acc"], "logreg": stats["logreg acc"]})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
