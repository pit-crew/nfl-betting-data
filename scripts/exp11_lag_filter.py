# Experiment 11: baseline and logistic regression on rows where teams have at least `lag` earlier games
# this season (filter applied to training and validation rows).
from features import load
from logreg import VIDEOS, fit, per_game, per_team, predict, split


def scores(tr, va, cols):
    y = va["y"].to_numpy()
    scored = y != 0.5  # ties with TIE_VALUE 0.5 are not scored
    rate = y.mean()
    b = fit(tr[cols].to_numpy(float), tr["y"].to_numpy(float))
    p = predict(b, va[cols].to_numpy(float))
    return len(va), (y[scored] == (rate > 0.5)).mean(), ((p > 0.5) == y)[scored].mean()


if __name__ == "__main__":
    df = load()
    game, gcols = per_game(df)
    team, tcols = per_team(df)
    for (name, last, even), lag in zip(VIDEOS, [5, 10]):
        print(f"== {name}, lag {lag}")
        filters = [
            ("home view, home team", game, gcols, lambda d: d["home_games_played"] >= lag),
            ("home view, away team", game, gcols, lambda d: d["away_games_played"] >= lag),
            ("home view, both", game, gcols, lambda d: (d["home_games_played"] >= lag) & (d["away_games_played"] >= lag)),
            ("per team, both", team, tcols, lambda d: (d["own_games_played"] >= lag) & (d["opp_games_played"] >= lag)),
        ]
        for label, data, cols, keep in filters:
            tr, va = split(data, last, even)
            n, base, lr = scores(tr[keep(tr)], va[keep(va)], cols)
            print(f"{label}: val rows {n}, baseline {base:.4f}, logreg {lr:.4f}")
