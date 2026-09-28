# Experiment 10: baseline rule "predict a win when the team's average score difference is higher than the opponent's".
from features import load
from logreg import VIDEOS, per_game, per_team, split


def acc(y, pred):
    scored = y != 0.5  # ties with TIE_VALUE 0.5 are not scored
    return (y[scored] == pred[scored]).mean()


if __name__ == "__main__":
    df = load()
    team, _ = per_team(df)
    game, _ = per_game(df)
    for (name, last, even), lag in zip(VIDEOS, [5, 10]):
        print(f"== {name}, lag {lag}")
        _, va = split(team, last, even)
        va = va[(va["own_games_played"] >= lag) & (va["opp_games_played"] >= lag)]
        pred = (va["own_avg_diff"] > va["opp_avg_diff"]).to_numpy()
        print(f"per team: rows {len(va)}, {acc(va['y'].to_numpy(), pred):.4f}")
        _, va = split(game, last, even)
        va = va[(va["home_games_played"] >= lag) & (va["away_games_played"] >= lag)]
        pred = (va["home_avg_diff"] >= va["away_avg_diff"]).to_numpy()  # equal -> home win
        print(f"home view: rows {len(va)}, {acc(va['y'].to_numpy(), pred):.4f}")
