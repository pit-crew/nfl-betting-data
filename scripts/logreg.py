import numpy as np
import pandas as pd

from features import game_features, load, team_games

FEATS = ["games_played", "wins", "avg_diff"]

# (label, last season, training seasons are even?)
VIDEOS = [("video 1", 2023, True), ("video 2", 2024, False)]


def fit(X, y):
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b))
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


def predict(b, X):
    return 1 / (1 + np.exp(-(b[0] + X @ b[1:])))


def per_game(df):
    g = game_features(df)
    cols = [f"home_{f}" for f in FEATS] + [f"away_{f}" for f in FEATS]
    return g.rename(columns={"schedule_season": "season", "schedule_week": "week", "home_win": "y"}), cols


def per_team(df):
    t = team_games(df)
    opp = t[["game_idx", "team"] + FEATS].rename(columns={"team": "opponent"})
    opp = opp.rename(columns={f: f"opp_{f}" for f in FEATS})
    t = t.merge(opp, on=["game_idx", "opponent"])
    t = t.rename(columns={**{f: f"own_{f}" for f in FEATS}, "win": "y"})
    t["week"] = df.loc[t["game_idx"], "schedule_week"].to_numpy()
    cols = [f"own_{f}" for f in FEATS] + [f"opp_{f}" for f in FEATS]
    return t, cols


def split(data, last_season, train_even):
    d = data[(data["season"] <= last_season) & ~((data["season"] == last_season) & (data["week"] == "Superbowl"))]
    is_train = (d["season"] % 2 == 0) == train_even
    return d[is_train], d[~is_train]


def run(data, cols, last_season, train_even):
    tr, va = split(data, last_season, train_even)
    b = fit(tr[cols].to_numpy(float), tr["y"].to_numpy(float))
    p = predict(b, va[cols].to_numpy(float))
    y = va["y"].to_numpy()
    rate = y.mean()
    ame = (p * (1 - p)).mean() * b[1:]
    return {
        "train rows": len(tr), "val rows": len(va),
        "val win rate": rate,
        "baseline acc": max(rate, 1 - rate),
        "logreg acc": ((p > 0.5) == y).mean(),
    }, pd.DataFrame({"coef": b[1:], "avg effect on P(win)": ame}, index=cols)


if __name__ == "__main__":
    df = load()
    for layout, (data, cols) in [("one row per game (home view)", per_game(df)),
                                 ("one row per team per game", per_team(df))]:
        for name, last, even in VIDEOS:
            stats, coefs = run(data, cols, last, even)
            print(f"== {name}, {layout}")
            for k, v in stats.items():
                print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
            print(coefs.round(4).to_string())
            print()
