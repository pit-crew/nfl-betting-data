# Experiment 9: three guesses at how the author computed the baseline.
import features
from features import load
from logreg import VIDEOS, per_game, per_team, split


def baseline(y):
    scored = y != 0.5  # ties with TIE_VALUE 0.5 are not scored
    rate = y.mean()
    r = y[scored].mean()
    return rate, (y[scored] == (rate > 0.5)).mean(), rate * r + (1 - rate) * (1 - r)


if __name__ == "__main__":
    df = load()
    team, _ = per_team(df)
    game, _ = per_game(df)
    features.TIE_VALUE = 0
    team_tie_loss, _ = per_team(df)
    features.TIE_VALUE = 0.5

    for name, last, even in VIDEOS:
        print(f"== {name}")
        _, va = split(team, last, even)
        rate, acc, _ = baseline(va[va["own_games_played"] >= 5]["y"].to_numpy())
        print(f"1: per team, >= 5 earlier games: win rate {rate:.4f}, baseline {acc:.4f}")
        _, va = split(team_tie_loss, last, even)
        rate, acc, _ = baseline(va["y"].to_numpy())
        print(f"2: per team, ties as losses: win rate {rate:.4f}, baseline {acc:.4f}")
        _, va = split(game, last, even)
        rate, acc, exp = baseline(va["y"].to_numpy())
        print(f"3: home view: win rate {rate:.4f}, baseline {acc:.4f}, expected accuracy {exp:.4f}")
