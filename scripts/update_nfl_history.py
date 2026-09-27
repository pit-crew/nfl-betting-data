import json
import sys
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

PATH = "nfl-history.csv"
API = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
PLAYOFF_WEEKS = {1: "Wildcard", 2: "Division", 3: "Conference", 5: "Superbowl"}

now = datetime.now(timezone.utc)
season = now.year - 1 if now.month <= 2 else now.year

with open(PATH) as f:
    csv = f.read()
known = set()
for line in csv.strip().split("\n")[1:]:
    f = line.split(",")
    known.add((f[1], f[2], f[4], f[7]))


def et_date(iso):
    d = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(ZoneInfo("America/New_York"))
    return f"{d.month}/{d.day}/{d.year}"


weeks = [(2, n, str(n)) for n in range(1, 19)] + [(3, n, label) for n, label in PLAYOFF_WEEKS.items()]
rows = []
for season_type, n, label in weeks:
    url = f"{API}?dates={season}&seasontype={season_type}&week={n}&limit=100"
    req = urllib.request.Request(url, headers={"accept": "application/json"})
    with urllib.request.urlopen(req) as r:
        events = json.load(r).get("events", [])
    for e in events:
        c = e["competitions"][0]
        if not c["status"]["type"]["completed"]:
            continue
        home = next(t for t in c["competitors"] if t["homeAway"] == "home")
        away = next(t for t in c["competitors"] if t["homeAway"] == "away")
        h, a = home["team"]["displayName"], away["team"]["displayName"]
        if (str(season), label, h, a) in known:
            continue
        playoff = "TRUE" if season_type == 3 else "FALSE"
        rows.append((e["date"], ",".join([et_date(e["date"]), str(season), label, playoff, h, home["score"], away["score"], a])))

if not rows:
    print("No new final games.")
    sys.exit(0)
rows.sort()
with open(PATH, "w") as f:
    f.write(csv.rstrip("\n") + "\n" + "\n".join(line for _, line in rows) + "\n")
print(f"Added {len(rows)} game(s) to {PATH}.")
