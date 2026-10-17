"""
TRAIN_CAUSAL_CAP_NN.PY   (causal cap network, step 2: train + compare with the cap table)
================================================================================
What this does, in plain words
------------------------------
Reads nn_stage_table.csv (one row per alarm stage). Trains a small PyTorch network
that predicts the CAP (kW kept per shed bus) using ONLY what is known at that alarm.
Then compares it, fairly, with the cap table (group averages) that the staged
limiter already uses.

Three input sets are tried (the same network each time):
    group           n_flagged, has_ef_flagged                       (same facts as the table)
    group_timing    + since_prev_alarm_s
    group_snapshot  + since_prev_alarm_s + the 36 grid measurements at the alarm
NEVER used as inputs: final_n, final_has_ef, is_final_stage, label, plateau_kw, kept_kw,
t_stage, flagged_buses, shed_buses_final  (they would leak the future).

Training rows: only is_final_stage == 1 rows of the TRAIN split (stage k of a scenario whose
final group has k buses: the only rows whose target is really the cap for "k buses").

The 9 steps of a PyTorch training recipe, as they appear in the code:
    1 prepare data (load_data, standardize)      2 define the model (CapNet)
    3 define the loss (MSE)                      4 define the optimizer (Adam, lr 0.01, wd 1e-3)
    loop:  5 forward pass   6 loss   7 backward pass (loss.backward())
           8 update weights (opt.step())   9 zero the gradients (opt.zero_grad())
    plus: validation loss every epoch, early stopping (patience 100), keep best weights.

Run (CPU, a minute or two):
    python train_causal_cap_nn.py --table nn_stage_table.csv --out-dir causal_nn_results
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split

# ---------------- settings (all in one place) ----------------
HIDDEN = (16, 8)
LR = 0.01
WEIGHT_DECAY = 1e-3
MAX_EPOCHS = 2000
PATIENCE = 100
N_SEEDS = 5
N_FOLDS = 5
VAL_FRACTION = 0.2
TARGET_UNSAFE = 0.05          # same rule as before: smallest margin with <= 5 % unsafe
TOL_FRAC = 0.025              # half a label grid step (0.05 / 2) x plateau
TABLE_MARGIN_KW = 5.0         # the margin the cap table uses today
MARGIN_STEP_KW = 0.5
MARGIN_MAX_KW = 25.0

META = ["scenario_id", "kind", "split", "stage", "final_n", "final_has_ef", "is_final_stage",
        "label", "plateau_kw", "kept_kw", "t_stage", "flagged_buses", "shed_buses_final"]
GROUP = ["n_flagged", "has_ef_flagged"]
TIMING = ["since_prev_alarm_s"]
FORBIDDEN = set(META) - {"scenario_id"}
TORCH_DEVICE = "cpu"


# ---------------- step 1: data ----------------
def load_data(path):
    d = pd.read_csv(path)
    snap = [c for c in d.columns if c not in META and c not in GROUP + ["n_ef_flagged"] + TIMING]
    variants = {"group": GROUP, "group_timing": GROUP + TIMING, "group_snapshot": GROUP + TIMING + snap}
    for name, cols in variants.items():
        bad = FORBIDDEN & set(cols)
        assert not bad, f"leaking columns in {name}: {bad}"
    assert not d[sum(variants.values(), [])].isna().any().any(), "NaN in inputs"
    return d, variants


def standardize_fit(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-9] = 1.0
    return mu, sd


# ---------------- step 2: model ----------------
class CapNet(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, HIDDEN[0]), nn.ReLU(),
                                 nn.Linear(HIDDEN[0], HIDDEN[1]), nn.ReLU(),
                                 nn.Linear(HIDDEN[1], 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one(Xtr, ytr, Xva, yva, seed):
    """Train one network with early stopping. Returns (predict_function, history)."""
    torch.manual_seed(seed)
    mu, sd = standardize_fit(Xtr)
    ym, ys = float(ytr.mean()), float(ytr.std() or 1.0)
    f = lambda X: torch.tensor((X - mu) / sd, dtype=torch.float32)
    xt, xv = f(Xtr), f(Xva)
    yt = torch.tensor((ytr - ym) / ys, dtype=torch.float32)
    yv = torch.tensor((yva - ym) / ys, dtype=torch.float32)
    model = CapNet(Xtr.shape[1])                                   # 2 model
    loss_fn = nn.MSELoss()                                         # 3 loss
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)   # 4 optimizer
    best, best_state, since_best, hist = np.inf, None, 0, {"train": [], "val": []}
    for epoch in range(MAX_EPOCHS):
        model.train()
        pred = model(xt)                                           # 5 forward
        loss = loss_fn(pred, yt)                                   # 6 loss
        loss.backward()                                            # 7 backward
        opt.step()                                                 # 8 update
        opt.zero_grad()                                            # 9 zero gradients
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(xv), yv).item()
        hist["train"].append(loss.item()); hist["val"].append(vl)
        if vl < best - 1e-6:
            best, since_best = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()

    def predict(X):
        with torch.no_grad():
            return model(f(X)).numpy() * ys + ym
    hist["best_val"] = best
    hist["stopped_epoch"] = len(hist["val"])
    return predict, hist


def cell_of(df):
    return (df["n_flagged"].astype(int) * 10 + df["has_ef_flagged"].astype(int)).to_numpy()


def fit_ensemble(df, cols):
    """5 networks (different seeds and different 80/20 early-stopping splits), averaged."""
    X = df[cols].to_numpy(float); y = df["kept_kw"].to_numpy(float); strat = cell_of(df)
    members, hists = [], []
    for s in range(N_SEEDS):
        itr, iva = train_test_split(np.arange(len(df)), test_size=VAL_FRACTION,
                                    random_state=s, stratify=strat)
        p, h = train_one(X[itr], y[itr], X[iva], y[iva], seed=s)
        members.append(p); hists.append(h)
    return (lambda Xn: np.mean([m(Xn) for m in members], axis=0)), hists


# ---------------- the cap table baseline ----------------
def fit_table(df):
    g = df.groupby(["n_flagged", "has_ef_flagged"])["kept_kw"].mean()
    X = np.c_[np.ones(len(df)), df["n_flagged"], df["has_ef_flagged"]]
    beta = np.linalg.lstsq(X, df["kept_kw"].to_numpy(float), rcond=None)[0]

    def predict(dd):
        return np.array([g[(n, e)] if (n, e) in g.index else beta[0] + beta[1] * n + beta[2] * e
                         for n, e in zip(dd["n_flagged"], dd["has_ef_flagged"])], float)
    return predict


# ---------------- safety rule ----------------
def unsafe_mask(cap_pred, kept, plateau, margin):
    """cap used = prediction - margin. Unsafe if it is above the true safe level by more than half a grid step."""
    return (cap_pred - margin) - kept > TOL_FRAC * plateau


def choose_margin(cap_pred, kept, plateau):
    for m in np.arange(0.0, MARGIN_MAX_KW + 1e-9, MARGIN_STEP_KW):
        if unsafe_mask(cap_pred, kept, plateau, m).mean() <= TARGET_UNSAFE:
            return float(m)
    return MARGIN_MAX_KW


def summarize(cap_pred, kept, plateau, margin):
    cap = cap_pred - margin
    diff = cap - kept                                  # >0 : cap above the safe level (risk); <0 : extra cut
    un = unsafe_mask(cap_pred, kept, plateau, margin)
    return {"n": int(len(kept)), "margin_kw": round(float(margin), 2),
            "MAE_kw": round(float(np.abs(cap_pred - kept).mean()), 2),
            "unsafe": int(un.sum()), "unsafe_rate": round(float(un.mean()), 3),
            "mean_extra_cut_kw": round(float(np.clip(-diff, 0, None).mean()), 2),
            "worst_overshoot_kw": round(float(diff.max()), 2)}


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="nn_stage_table.csv")
    ap.add_argument("--out-dir", default="causal_nn_results")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    t0 = time.time()

    d, variants = load_data(a.table)
    tr = d[(d.split == "train") & (d.is_final_stage == 1)].reset_index(drop=True)
    assert tr.scenario_id.is_unique
    ev = d[d.split.isin(["test", "profile"])].reset_index(drop=True)      # all stages of test + H rows
    print(f"training rows (train, final stage): {len(tr)};  evaluation rows (test+profile, all stages): {len(ev)}")
    print("inputs:", {k: len(v) for k, v in variants.items()})

    # ---- 5-fold cross-validation on the training rows (grouped by scenario: one row per scenario) ----
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=0)
    folds = list(skf.split(tr, cell_of(tr)))
    oof = {"table": np.zeros(len(tr))}
    for v in variants:
        oof[v] = np.zeros(len(tr))
    for k, (i_tr, i_va) in enumerate(folds):
        sub, hold = tr.iloc[i_tr], tr.iloc[i_va]
        oof["table"][i_va] = fit_table(sub)(hold)
        for v, cols in variants.items():
            pred, _ = fit_ensemble(sub, cols)
            oof[v][i_va] = pred(hold[cols].to_numpy(float))
        print(f"  fold {k + 1}/{N_FOLDS} done  ({time.time() - t0:.0f}s)")
    kept, plat = tr.kept_kw.to_numpy(float), tr.plateau_kw.to_numpy(float)
    cv_margin = {m: choose_margin(oof[m], kept, plat) for m in oof}
    cv_rows = []
    for m in oof:
        for label, mg in (("own_margin", cv_margin[m]), ("table_margin_5kW", TABLE_MARGIN_KW)):
            cv_rows.append({"model": m, "margin_rule": label, **summarize(oof[m], kept, plat, mg)})
    cv_df = pd.DataFrame(cv_rows)
    print("\n=== 5-fold CV on train rows (out-of-fold predictions) ===")
    print(cv_df.to_string(index=False))
    pd.DataFrame({"scenario_id": tr.scenario_id, "n_flagged": tr.n_flagged, "has_ef": tr.has_ef_flagged,
                  "kept_kw": kept, **{f"oof_{m}": oof[m] for m in oof}}).round(3).to_csv(out / "cv_oof_predictions.csv", index=False)

    # ---- final models on ALL train rows, evaluated on test + H rows ----
    preds = {"table": fit_table(tr)(ev)}
    hist_keep = {}
    final_models = {}
    for v, cols in variants.items():
        pred, hists = fit_ensemble(tr, cols)
        preds[v] = pred(ev[cols].to_numpy(float))
        hist_keep[v] = hists
        final_models[v] = pred
        print(f"  final {v}: stopped at epochs {[h['stopped_epoch'] for h in hists]}")

    kept_e, plat_e = ev.kept_kw.to_numpy(float), ev.plateau_kw.to_numpy(float)
    is_fin = ev.is_final_stage.to_numpy() == 1
    res_rows = []
    for m in preds:
        for label, mg in (("own_margin", cv_margin[m]), ("table_margin_5kW", TABLE_MARGIN_KW)):
            for sel, name in ((is_fin & (ev.split == "test").to_numpy(), "test_final_stage"),
                              (is_fin & (ev.split == "profile").to_numpy(), "H01-H03_final_stage"),
                              (~is_fin, "non_final_stages(test+H)")):
                if sel.sum():
                    res_rows.append({"model": m, "margin_rule": label, "rows": name,
                                     **summarize(preds[m][sel], kept_e[sel], plat_e[sel], mg)})
    res_df = pd.DataFrame(res_rows)
    print("\n=== held-out results (models trained on all train rows) ===")
    print(res_df.to_string(index=False))
    res_df.to_csv(out / "heldout_results.csv", index=False)

    # per-row predictions (all stages) for inspection
    pr = ev[["scenario_id", "split", "stage", "final_n", "final_has_ef", "is_final_stage",
             "flagged_buses", "plateau_kw", "kept_kw"]].copy()
    for m in preds:
        pr[f"cap_{m}"] = preds[m]
    pr.round(3).to_csv(out / "heldout_predictions_all_stages.csv", index=False)
    print("\nper-scenario, final stage (cap before margin, kW):")
    print(pr[pr.is_final_stage == 1].drop(columns=["flagged_buses", "is_final_stage"]).round(1).to_string(index=False))

    # cap cells for the limiter (group-only network): same layout as staged_cap_groups.csv
    cells = []
    for n in (1, 2, 3):
        for e in (0, 1):
            x = np.array([[n, e]], float)
            nn_cap = float(final_models["group"](x)[0])
            tb_rows = tr[(tr.n_flagged == n) & (tr.has_ef_flagged == e)]
            cells.append({"n_target": n, "has_ef": e, "n_rows": len(tb_rows),
                          "std_kw": round(float(tb_rows.kept_kw.std()), 3),
                          "cap_kw_nn": round(nn_cap, 3), "cap_kw_table": round(float(tb_rows.kept_kw.mean()), 3)})
    cells = pd.DataFrame(cells)
    cells["cap_kw_margin_nn"] = (cells.cap_kw_nn - cv_margin["group"]).round(3)
    cells["cap_kw_margin_table"] = (cells.cap_kw_table - TABLE_MARGIN_KW).round(3)
    cells.to_csv(out / "cap_cells_group_network.csv", index=False)
    print("\ncap cells (group-only network vs table):")
    print(cells.to_string(index=False))

    # loss curves (white background)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(variants), figsize=(5 * len(variants), 3.8), facecolor="white")
    for ax, (v, hs) in zip(np.atleast_1d(axs), hist_keep.items()):
        for i, h in enumerate(hs):
            ax.plot(h["train"], color="#1f4e9c", alpha=0.7, lw=1, label="training loss" if i == 0 else None)
            ax.plot(h["val"], color="#c0392b", alpha=0.7, lw=1, label="validation loss" if i == 0 else None)
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE (standardized)")
        ax.set_title(f"inputs: {v}"); ax.set_facecolor("white"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "loss_curves.png", dpi=150, facecolor="white")

    json.dump({"cv_margin_kw": cv_margin, "table_margin_kw": TABLE_MARGIN_KW,
               "settings": {"hidden": HIDDEN, "lr": LR, "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS,
                            "patience": PATIENCE, "seeds": N_SEEDS, "folds": N_FOLDS,
                            "target_unsafe": TARGET_UNSAFE, "tol_frac": TOL_FRAC},
               "inputs": {k: len(v) for k, v in variants.items()}},
              open(out / "run_summary.json", "w"), indent=1)
    print(f"\nDone in {time.time() - t0:.0f}s. Files in {out}/")


if __name__ == "__main__":
    main()
