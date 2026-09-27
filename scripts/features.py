import pandas as pd

PATH = "nfl-history.csv"


def load():
    df = pd.read_csv(PATH)
    df["schedule_date"] = pd.to_datetime(df["schedule_date"], format="%m/%d/%Y")
    return df.sort_values("schedule_date", kind="stable").reset_index(drop=True)


def team_games(df):
    home = pd.DataFrame({
        "game_idx": df.index, "season": df["schedule_season"], "date": df["schedule_date"],
        "team": df["team_home"], "opponent": df["team_away"], "is_home": True,
        "points_for": df["score_home"], "points_against": df["score_away"],
    })
    away = pd.DataFrame({
        "game_idx": df.index, "season": df["schedule_season"], "date": df["schedule_date"],
        "team": df["team_away"], "opponent": df["team_home"], "is_home": False,
        "points_for": df["score_away"], "points_against": df["score_home"],
    })
    t = pd.concat([home, away]).sort_values(["date", "game_idx"], kind="stable").reset_index(drop=True)
    t["diff"] = t["points_for"] - t["points_against"]
    t["win"] = (t["diff"] > 0).astype(int)

    g = t.groupby(["team", "season"])
    t["games_played"] = g.cumcount()
    t["wins"] = g["win"].cumsum() - t["win"]
    diff_sum = g["diff"].cumsum() - t["diff"]
    t["avg_diff"] = (diff_sum / t["games_played"]).where(t["games_played"] > 0, 0.0)
    return t


def game_features(df):
    t = team_games(df)
    cols = ["game_idx", "games_played", "wins", "avg_diff"]
    home = t[t["is_home"]][cols].set_index("game_idx").add_prefix("home_")
    away = t[~t["is_home"]][cols].set_index("game_idx").add_prefix("away_")
    out = df.join(home).join(away)
    out["home_win"] = (out["score_home"] > out["score_away"]).astype(int)
    out["score_diff"] = out["score_home"] - out["score_away"]
    return out


if __name__ == "__main__":
    df = load()
    t = team_games(df)
    games = game_features(df)
    print("game rows:", len(games))
    print("team rows:", len(t))

    first = t[t["games_played"] == 0]
    print("first games with nonzero features:", int(((first["wins"] != 0) | (first["avg_diff"] != 0)).sum()))

    for team, season in [("Kansas City Chiefs", 2024), ("Philadelphia Eagles", 2024)]:
        sb = t[(t["team"] == team) & (t["season"] == season)].iloc[-1]
        print(f"{team} before {season} Super Bowl: games_played {sb['games_played']}, wins {sb['wins']}")

    kc = t[(t["team"] == "Kansas City Chiefs") & (t["season"] == 2023)].iloc[:-1]
    for n in (5, 10):
        w = int(kc["win"].tail(n).sum())
        print(f"Chiefs last {n} before 2023 Super Bowl: {w}-{n - w}")
