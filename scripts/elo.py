import argparse
import math
import sys

import pandas as pd

HISTORY = "nfl-history.csv"
SCHEDULE = "nfl-2026.csv"
TEAMS = "nfl_teams.csv"

K = 20          # update size
HFA = 48        # home-field advantage in rating points (0 for the Super Bowl)
MEAN = 1505     # starting rating and target of season reversion
REVERT = 1 / 3  # share of distance to MEAN removed at the start of each season
EVAL_FROM = 2000


def expected(diff):
    return 1 / (1 + 10 ** (-diff / 400))


def team_ids():
    t = pd.read_csv(TEAMS)
    ids = dict(zip(t["team_name"], t["team_id"]))
    ids.update(zip(t["team_name_short"], t["team_id"]))
    ids["Commanders"] = "WAS"
    return ids


def run(games):
    ratings, season, rows = {}, None, []
    for g in games.itertuples():
        if g.season != season:
            ratings = {t: MEAN + (1 - REVERT) * (r - MEAN) for t, r in ratings.items()}
            season = g.season
        h, a = ratings.setdefault(g.home, MEAN), ratings.setdefault(g.away, MEAN)
        diff = h - a + (0 if g.week == "Superbowl" else HFA)
        p = expected(diff)
        margin = g.score_home - g.score_away
        result = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        winner_diff = diff if margin > 0 else -diff
        mult = math.log(abs(margin) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
        shift = K * mult * (result - p)
        ratings[g.home] += shift
        ratings[g.away] -= shift
        rows.append({"season": g.season, "p_home": p, "result": result})
    return ratings, pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=pd.Timestamp.today().strftime("%Y-%m-%d"),
                    help="yyyy-mm-dd; shows the two weeks starting the Thursday on or after this date")
    ap.add_argument("--csv", action="store_true", help="output CSV instead of a markdown table")
    args = ap.parse_args()
    start = pd.Timestamp(args.date)
    start += pd.Timedelta(days=(3 - start.weekday()) % 7)
    end = start + pd.Timedelta(days=13)

    ids = team_ids()
    h = pd.read_csv(HISTORY)
    games = pd.DataFrame({
        "date": pd.to_datetime(h["schedule_date"], format="%m/%d/%Y"),
        "season": h["schedule_season"], "week": h["schedule_week"],
        "home": h["team_home"].map(ids), "away": h["team_away"].map(ids),
        "score_home": h["score_home"], "score_away": h["score_away"],
    }).sort_values("date", kind="stable")
    ratings, res = run(games)

    ev = res[(res["season"] >= EVAL_FROM) & (res["result"] != 0.5)]
    print(f"seasons {EVAL_FROM}+: games {len(ev)}, "
          f"accuracy {((ev['p_home'] > 0.5) == (ev['result'] == 1)).mean():.4f}, "
          f"Brier {((ev['p_home'] - ev['result']) ** 2).mean():.4f}",
          file=sys.stderr if args.csv else sys.stdout)

    s = pd.read_csv(SCHEDULE)
    s["date"] = pd.to_datetime(s["date"])
    s["home"], s["away"] = s["home_team"].map(ids), s["away_team"].map(ids)
    played = set(zip(games["date"], games["home"]))
    s = s[[(d, t) not in played for d, t in zip(s["date"], s["home"])]]
    s = s[(s["date"] >= start) & (s["date"] <= end)]
    diff = s["home"].map(ratings) - s["away"].map(ratings) + HFA
    home_fav = diff >= 0
    s["favourite"] = s["home_team"].where(home_fav, s["away_team"])
    s["prob"] = expected(diff.abs())
    s["spread"] = diff.abs() / 25
    s = s.sort_values("prob", ascending=False, kind="stable")

    if args.csv:
        out = s[["date", "home_team", "away_team", "favourite", "prob", "spread"]].round({"spread": 1})
        out["prob"] = out["prob"].fillna(0.0).map("{:.4f}".format)
        out.to_csv(sys.stdout, index=False, date_format="%Y-%m-%d")
        sys.exit()

    print()
    print("| Date | Home | Away | Favourite | Prob | Spread |")
    print("|---|---|---|---|---|---|")
    for r in s.itertuples():
        print(f"| {r.date:%b} {r.date.day} | {r.home_team} | {r.away_team} | {r.favourite} | {r.prob:.1%} | {r.spread:.1f} |")
