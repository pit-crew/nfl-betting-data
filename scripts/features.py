import pandas as pd

PATH = "spreadspoke_scores.csv"
TEAMS = "nfl_teams.csv"
DROP_TIES = False
TIE_VALUE = 0.5  # label and win credit for a tie; 0 = loss
RESET_BY_SEASON = True  # False: features accumulate over a team's whole history


def load():
    df = pd.read_csv(PATH)
    if DROP_TIES:
        df = df[df["score_home"] != df["score_away"]].copy()
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
    t["win"] = (t["diff"] > 0) + TIE_VALUE * (t["diff"] == 0)

    ids = pd.read_csv(TEAMS).set_index("team_name")["team_id"]
    t["team_id"] = t["team"].map(ids)
    g = t.groupby(["team_id", "season"] if RESET_BY_SEASON else ["team_id"])
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
    out["home_win"] = (out["score_home"] > out["score_away"]) + TIE_VALUE * (out["score_home"] == out["score_away"])
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
