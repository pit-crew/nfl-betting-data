# Experiment 4: logistic regression trained season by season in time order vs shuffled.
import numpy as np
import torch
from torch import nn

from features import load
from logreg import VIDEOS, per_game, split

EPOCHS = 20
SEEDS = [0, 1, 2]


def fit(X, y, seasons, seed):
    # seasons=None: shuffled. Otherwise: each season in time order, EPOCHS passes in date order.
    torch.manual_seed(seed)
    model = nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    X, y = torch.from_numpy(X), torch.from_numpy(y)

    def step(b):
        opt.zero_grad()
        loss_fn(model(X[b]).squeeze(-1), y[b]).backward()
        opt.step()

    if seasons is None:
        for _ in range(EPOCHS):
            for b in torch.randperm(len(X)).split(64):
                step(b)
    else:
        for s in np.unique(seasons):
            idx = torch.from_numpy(np.flatnonzero(seasons == s))
            for _ in range(EPOCHS):
                for b in idx.split(64):
                    step(b)
    return model


if __name__ == "__main__":
    data, cols = per_game(load())
    for name, last, even in VIDEOS:
        tr, va = split(data, last, even)
        mean, std = tr[cols].mean(), tr[cols].std()
        Xtr = ((tr[cols] - mean) / std).to_numpy(np.float32)
        Xva = torch.from_numpy(((va[cols] - mean) / std).to_numpy(np.float32))
        ytr, yva = tr["y"].to_numpy(np.float32), va["y"].to_numpy()
        rate = yva.mean()
        print(f"== {name}: baseline {max(rate, 1 - rate):.4f}")
        for order, seasons in [("shuffled", None), ("season by season", tr["season"].to_numpy())]:
            accs = []
            for seed in SEEDS:
                m = fit(Xtr, ytr, seasons, seed)
                with torch.no_grad():
                    accs.append(((m(Xva).squeeze(-1) > 0).numpy() == yva).mean())
            print(f"{order}: acc mean {np.mean(accs):.4f}, min {min(accs):.4f}, max {max(accs):.4f}")
