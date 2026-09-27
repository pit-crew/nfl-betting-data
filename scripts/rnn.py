import numpy as np
import pandas as pd
import torch
from torch import nn

from features import load, team_games
from logreg import VIDEOS, per_game, split

LAGS = [1, 3, 5, 10, 15]
HIDDEN = [2, 8, 32]
SEEDS = [0, 1, 2]
EPOCHS = 20


def game_seqs(df, lag):
    # (games, 2, lag): game indices of the home and away team's last `lag` games this season,
    # ending with this game; -1 = padding at the front.
    t = team_games(df)
    g = t.groupby(["team", "season"])["game_idx"]
    seq = np.stack([g.shift(k).fillna(-1).to_numpy(int) for k in reversed(range(lag))], axis=1)
    out = np.full((len(df), 2, lag), -1)
    home = t["is_home"].to_numpy()
    out[t["game_idx"][home], 0] = seq[home]
    out[t["game_idx"][~home], 1] = seq[~home]
    return out


class Net(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        self.rnn = nn.RNN(n_in, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.out(h[-1]).squeeze(-1)


def train(X, y, hidden, seed):
    torch.manual_seed(seed)
    model = Net(X.shape[2], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    X, y = torch.from_numpy(X), torch.from_numpy(y)
    for _ in range(EPOCHS):
        for b in torch.randperm(len(X)).split(64):
            opt.zero_grad()
            loss_fn(model(X[b]), y[b]).backward()
            opt.step()
    return model


def prob(model, X):
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).numpy()


if __name__ == "__main__":
    df = load()
    data, cols = per_game(df)
    seqs = {lag: game_seqs(df, lag) for lag in LAGS}

    rows = []
    for name, last, even in VIDEOS:
        tr, va = split(data, last, even)
        sb = data.index[(data["season"] == last) & (data["week"] == "Superbowl")]
        chiefs_home = data.loc[sb[0], "team_home"] == "Kansas City Chiefs"
        mean, std = tr[cols].mean(), tr[cols].std()
        # last row is zeros, used by padding index -1
        Z = np.vstack([((data[cols] - mean) / std).to_numpy(np.float32), np.zeros((1, len(cols)), np.float32)])
        for lag in LAGS:
            home = Z[seqs[lag][:, 0]]
            for variant, X in [("A", home), ("B", np.concatenate([home, Z[seqs[lag][:, 1]]], axis=2))]:
                for hidden in HIDDEN:
                    for seed in SEEDS:
                        m = train(X[tr.index], tr["y"].to_numpy(np.float32), hidden, seed)
                        acc = ((prob(m, X[va.index]) > 0.5) == va["y"].to_numpy()).mean()
                        p_home = float(prob(m, X[sb])[0])
                        rows.append({"video": name, "variant": variant, "lag": lag, "hidden": hidden, "seed": seed,
                                     "acc": acc, "p_chiefs": p_home if chiefs_home else 1 - p_home})
                        print(rows[-1], flush=True)

    res = pd.DataFrame(rows)
    summary = res.groupby(["video", "variant", "lag", "hidden"]).agg(
        acc_mean=("acc", "mean"), acc_std=("acc", "std"), p_chiefs=("p_chiefs", "mean"))
    print(summary.round(4).to_string())
