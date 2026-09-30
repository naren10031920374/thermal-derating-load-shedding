"""
TRAIN_SHED_MODELS.PY  (dataset pipeline, step 5b)
================================================================================
Trains and honestly evaluates models that predict the minimum safe shed
(ShedFrac = fraction of load KEPT) from grid state at the detector trigger.

Data: nn_table.csv from build_nn_table.py, one row per scenario. Only ~40
training rows exist, so this script compares the NN with simple models on the
same footing and reports all of them:

  mean_baseline   always predicts the training-set mean (the bar to beat)
  ridge           standardized linear model, penalty picked by inner CV
  gbr             small gradient-boosted trees
  rf              random forest
  nn_alpha1/10    NN: 5 small MLPs (16-8) averaged, L2 penalty 1 and 10
                  (two settings are reported, not picked on the test set)

Evaluation
  - LOO       leave-one-scenario-out on the TRAIN rows (the model-selection number)
  - TEST      the held-out scenarios in the 'split' column, scored once, at the end
Metrics (labels sit on a 0.05 grid)
  mae            mean |pred - true|, ShedFrac units (x100 = percentage points of shed)
  within_1_step  share of predictions within one grid step (0.05)
  unsafe_rate    share where pred > true + 0.025, i.e. the cut is MILDER than the
                 measured minimum by more than half a grid step. This is the error
                 that matters: a milder-than-needed cut can leave a bus collapsing.
  mean_overshoot average of (pred - true) over those unsafe predictions
SAFETY MARGIN: a margin m (subtract from every prediction = cut harder) is chosen
on the TRAIN LOO predictions as the smallest m with unsafe_rate <= TARGET_UNSAFE
and then applied unchanged to the test rows. 'extra_shed' is the price: how much
more load is shed on average than the measured minimum.
IMPORTANT: a model's prediction is not proof of safety. The next step is to run the
predicted cut in the full simulation on the test scenarios and check that all 10
buses survive.

Run:
    python train_shed_models.py
    python train_shed_models.py --table path/to/nn_table.csv --out-dir results_dir
"""
from __future__ import annotations
import argparse, json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]

META_COLS = ["scenario_id", "kind", "split", "label", "label_source", "t_star",
             "n_shed_buses", "shed_buses"]
GRID_STEP = 0.05
UNSAFE_TOL = GRID_STEP / 2
TARGET_UNSAFE = 0.05
SEEDS = (0, 1, 2, 3, 4)


def make_mlp(alpha):
    """NN for a tiny dataset: standardized inputs AND target, L-BFGS (much better than
    Adam on ~40 rows), strong L2 penalty, and 5 seeds averaged so one unlucky init
    cannot decide the result."""
    est = [(f"m{s}", make_pipeline(StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(16, 8), alpha=alpha, solver="lbfgs",
                         max_iter=5000, random_state=s)))
           for s in SEEDS]
    return TransformedTargetRegressor(regressor=VotingRegressor(est), transformer=StandardScaler())


def make_models():
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "gbr": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2,
                                         subsample=0.8, random_state=0),
        "rf": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0),
        "nn_alpha1": make_mlp(1.0),
        "nn_alpha10": make_mlp(10.0),
    }


def metrics(y, p, margin=0.0):
    p = np.clip(p - margin, GRID_STEP, 1.0)
    err = p - y
    unsafe = err > UNSAFE_TOL
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "mae_pct_points": float(100 * np.mean(np.abs(err))),
        "within_1_step": float(np.mean(np.abs(err) <= GRID_STEP + 1e-9)),
        "unsafe_rate": float(np.mean(unsafe)),
        "mean_overshoot": float(err[unsafe].mean()) if unsafe.any() else 0.0,
        "extra_shed_pct_points": float(100 * np.mean(y - p)),   # >0: sheds more than the measured minimum
    }


def pick_margin(y, p):
    for m in np.arange(0.0, 0.501, 0.01):
        if np.mean(np.clip(p - m, GRID_STEP, 1.0) - y > UNSAFE_TOL) <= TARGET_UNSAFE:
            return float(round(m, 2))
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    root = Path(os.environ.get("DATASET_OUTPUT_ROOT") or PROJECT_ROOT / "model_outputs" / "dataset_pipeline")
    ap.add_argument("--table", default=str(root / "nn_table.csv"))
    ap.add_argument("--out-dir", default=str(root / "shed_models"))
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.table)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    tr = df[df["split"] == "train"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    Xtr, ytr = tr[feat_cols].to_numpy(float), tr["label"].to_numpy(float)
    Xte, yte = te[feat_cols].to_numpy(float), te["label"].to_numpy(float)
    print(f"train rows {len(tr)}  test rows {len(te)}  features {len(feat_cols)}")
    print("label range train: %.2f..%.2f  mean %.3f" % (ytr.min(), ytr.max(), ytr.mean()))

    models = make_models()
    report, loo_pred, test_pred, margins = {}, {}, {}, {}
    for name, mdl in models.items():
        p_loo = cross_val_predict(clone(mdl), Xtr, ytr, cv=LeaveOneOut())
        m = pick_margin(ytr, p_loo)
        fitted = clone(mdl).fit(Xtr, ytr)
        p_te = fitted.predict(Xte) if len(te) else np.array([])
        loo_pred[name], test_pred[name], margins[name] = p_loo, p_te, m
        report[name] = {"margin": m,
                        "loo": metrics(ytr, p_loo), "loo_with_margin": metrics(ytr, p_loo, m)}
        if len(te):
            report[name]["test"] = metrics(yte, p_te)
            report[name]["test_with_margin"] = metrics(yte, p_te, m)
        joblib.dump({"model": fitted, "features": feat_cols, "margin": m}, out / f"shed_model_{name}.pkl")

    def row(name, key):
        r = report[name][key]
        return (f"{name:14s} MAE {r['mae']:.3f} ({r['mae_pct_points']:4.1f} pts)  "
                f"within1 {r['within_1_step']:.2f}  unsafe {r['unsafe_rate']:.2f}  "
                f"overshoot {r['mean_overshoot']:.3f}  extra_shed {r['extra_shed_pct_points']:+5.1f} pts")
    for key, title in (("loo", "LEAVE-ONE-OUT on train (no margin)"),
                       ("loo_with_margin", "LEAVE-ONE-OUT on train (with safety margin)")):
        print(f"\n=== {title} ===")
        for n in models: print(row(n, key))
    if len(te):
        for key, title in (("test", "HELD-OUT TEST (no margin)"),
                           ("test_with_margin", "HELD-OUT TEST (margin chosen on train LOO)")):
            print(f"\n=== {title} ===")
            for n in models: print(row(n, key))
        print("\nmargins:", margins)

    # per-scenario test predictions
    if len(te):
        pr = te[["scenario_id", "kind", "shed_buses", "label"]].copy()
        for n in models:
            pr[f"pred_{n}"] = np.round(test_pred[n], 3)
            pr[f"pred_{n}_margin"] = np.round(np.clip(test_pred[n] - margins[n], GRID_STEP, 1.0), 3)
        pr.to_csv(out / "test_predictions.csv", index=False)
        print("\n", pr.to_string(index=False))
    tr_pr = tr[["scenario_id", "kind", "label"]].copy()
    for n in models: tr_pr[f"loo_{n}"] = np.round(loo_pred[n], 3)
    tr_pr.to_csv(out / "train_loo_predictions.csv", index=False)

    rd = make_models()["ridge"].fit(Xtr, ytr)
    coef = pd.Series(rd[-1].coef_, index=feat_cols).sort_values(key=np.abs, ascending=False)
    print("\nridge: strongest standardized coefficients")
    print(coef.head(8).round(4).to_string())

    (out / "metrics.json").write_text(json.dumps(
        {"n_train": len(tr), "n_test": len(te), "features": feat_cols,
         "unsafe_tolerance": UNSAFE_TOL, "target_unsafe_rate": TARGET_UNSAFE, "models": report}, indent=2))
    print(f"\nWrote models, metrics.json and predictions to {out}")


if __name__ == "__main__":
    main()
