"""
Vehicle Classification - the whole project in one file.

    01_EDA.ipynb            exploratory data analysis
    02_Preprocessing.ipynb  cleaning, the test-noise model, feature engineering
    03_Models.ipynb         basic models, advanced models, ensembles, experiments

The notebooks explain every step; this module holds the same code once, so it
can be imported (``import vehicle_pipeline as vp``) or run end to end:

    python vehicle_pipeline.py               # everything (~26 min on an RTX 4060 laptop GPU)
    python vehicle_pipeline.py --quick       # one CV repeat, no TabPFN / FT-Transformer
    python vehicle_pipeline.py --no-gpu      # skip XGBoost/CatBoost GPU, MLPs, FT-Transformer, TabPFN
    python vehicle_pipeline.py --final-only  # just train the final model and write the submission
    python vehicle_pipeline.py --data PATH   # CSVs somewhere other than data/ (also --out PATH)

Task: predict vehicle_class (0 = Sedan, 1 = SUV, 2 = Sports Car) from 20 numeric
specifications. Metric: accuracy.

Validation: repeated stratified 5-fold CV (3 repeats = 15 folds, seeds 42/7/123)
on the 7,500 de-duplicated training rows; the same folds for every model. Each
validation fold is also scored under the measurement noise found in the test
file (0.08 sd on every value, and 0.12 sd on a random half of the values,
re-clipped and re-rounded). "Test-like accuracy" is the mean of those two and is
the selection metric. Promising models are re-tested on 15 fresh folds (seeds
2024/99/314). Test predictions come from a refit on all training rows.

Outputs (results/): one CSV per metrics table - the numbers in Model_Metrics.pdf -
summary.json, and the submission file submission_final_logreg.csv.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import time
import warnings
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.metrics import (adjusted_rand_score, confusion_matrix, homogeneity_score,
                             precision_recall_fscore_support, roc_auc_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (FunctionTransformer, PolynomialFeatures, PowerTransformer,
                                   QuantileTransformer, RobustScaler, SplineTransformer,
                                   StandardScaler)
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("VEHICLE_DATA", ROOT / "data"))              # --data overrides
RESULTS = Path(os.environ.get("VEHICLE_RESULTS", ROOT / "results"))     # --out overrides

TARGET, ID_COL = "vehicle_class", "vehicle_id"
CLASS_NAMES = {0: "Sedan", 1: "SUV", 2: "Sports Car"}
SEED = 42
SCREEN_SEEDS = (42, 7, 123)          # 3 repeats x 5 folds = 15 screening folds
CONFIRM_SEEDS = (2024, 99, 314)      # 15 fresh folds for the re-test
VARIANTS = {      # name: (noise sd as a fraction of feature sd, share of values hit, draws)
    "clean": (0.00, 1.0, 1),
    "u08":   (0.08, 1.0, 2),         # reading 1 of the test noise: 0.08 sd on every value
    "mix12": (0.12, 0.5, 2),         # reading 2: 0.12 sd on a random half of the values
    "u15":   (0.15, 1.0, 1),         # stress level
}
BASE = "Logistic regression (final)"


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + str(msg), flush=True)


# ==========================================================================
# 1. data: load, remove duplicates, drop the identifier
# ==========================================================================
@lru_cache(maxsize=1)
def load_raw():
    for f in ("vehicle_train_10000.csv", "vehicle_test.csv"):
        if not (DATA / f).exists():
            raise FileNotFoundError(f"put the competition files in {DATA} (missing {f})")
    return pd.read_csv(DATA / "vehicle_train_10000.csv"), pd.read_csv(DATA / "vehicle_test.csv")


@lru_cache(maxsize=1)
def load():
    """De-duplicated training arrays, test arrays, clip limits and recorded precision.

    2,500 of the 10,000 training rows are exact copies (measurements and label);
    they are dropped before any split. vehicle_id is a record number and is never
    a feature."""
    train_raw, test_raw = load_raw()
    features = [c for c in train_raw.columns if c not in (ID_COL, TARGET)]
    train = train_raw.drop_duplicates(subset=features + [TARGET]).reset_index(drop=True)
    X = train[features].to_numpy(float)
    return SimpleNamespace(
        features=features, X=X, y=train[TARGET].to_numpy(int),
        X_test=test_raw[features].to_numpy(float), test_ids=test_raw[ID_COL].to_numpy(),
        X_dup=train_raw[features].to_numpy(float), y_dup=train_raw[TARGET].to_numpy(int),
        lo=X.min(0), hi=X.max(0),                                            # clip limits
        dec=np.array([0 if np.all(np.round(c) == c) else 2 for c in X.T]))  # recorded precision


# ==========================================================================
# 2. the measurement noise of the test file, and training-time augmentation
# ==========================================================================
def test_noise(Xv, s, share, col_sd, rng):
    """N(0, s * feature sd) on a random `share` of the values, then re-clipped to
    the limits and re-rounded to the recorded precision - the pattern of the test file."""
    if s <= 0:
        return Xv.copy()
    D = load()
    E = rng.normal(0.0, 1.0, Xv.shape) * (s * col_sd)
    if share < 1:
        E *= rng.random(Xv.shape) < share
    Z = np.clip(Xv + E, D.lo, D.hi)
    for j, d in enumerate(D.dec):
        Z[:, j] = np.round(Z[:, j], d)
    return Z


def jitter_copies(Xa, ya, sd=0.20, copies=4, seed=SEED):
    """Training rows + `copies` Gaussian-jittered copies (the final model's augmentation)."""
    rng = np.random.default_rng(seed)
    col_sd = Xa.std(0, ddof=1)
    parts = [Xa] + [Xa + rng.normal(0.0, sd * col_sd, Xa.shape) for _ in range(copies)]
    return np.vstack(parts), np.concatenate([ya] * (copies + 1))


def test_noise_copies(Xa, ya, s=0.15, share=1.0, copies=4, seed=0):
    """Training rows + copies carrying the test-file noise exactly (clip + round)."""
    rng = np.random.default_rng(seed)
    col_sd = Xa.std(0, ddof=1)
    parts = [Xa] + [test_noise(Xa, s, share, col_sd, rng) for _ in range(copies)]
    return np.vstack(parts), np.tile(ya, copies + 1)


def alternating_noise_copies(Xa, ya, K, seed=0):
    """K test-noise copies alternating the two readings (0.08 sd / 0.12 sd on half)."""
    rng = np.random.default_rng(seed)
    col_sd = Xa.std(0, ddof=1)
    parts = [Xa] + [test_noise(Xa, *((0.08, 1.0) if k % 2 == 0 else (0.12, 0.5)), col_sd, rng)
                    for k in range(K)]
    return np.vstack(parts), np.tile(ya, K + 1)


def noise_calibration():
    """How much noise does the test file carry? Measurement error weakens
    correlations; simulated noise is added until the train correlations shrink
    as much as the test ones do."""
    D = load()
    iu = np.triu_indices(D.X.shape[1], 1)
    ca, cb = np.corrcoef(D.X, rowvar=False), np.corrcoef(D.X_test, rowvar=False)
    strong = np.abs(ca[iu]) > 0.6
    observed = float(np.mean(cb[iu][strong] / ca[iu][strong]))
    sd = D.X.std(0, ddof=1)
    levels = np.round(np.arange(0, 0.161, 0.02), 2)
    sim = [np.mean(np.corrcoef(test_noise(D.X, s, 1.0, sd, np.random.default_rng(0)),
                               rowvar=False)[iu][strong] / ca[iu][strong]) for s in levels]
    return {"pairs": int(strong.sum()), "observed_ratio": observed,
            "noise_sd": float(np.interp(observed, sim[::-1], levels[::-1]))}


# ==========================================================================
# 3. EDA checks used in the report
# ==========================================================================
def duplicate_leakage():
    """Plain 5-fold CV with and without the duplicate rows."""
    D = load()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in {"Gaussian Naive Bayes": GaussianNB(),
                        "1-nearest-neighbour": make_pipeline(StandardScaler(), KNeighborsClassifier(1)),
                        "Decision tree (unlimited depth)": DecisionTreeClassifier(random_state=0)}.items():
        a = cross_val_score(model, D.X_dup, D.y_dup, cv=cv).mean() * 100
        b = cross_val_score(model, D.X, D.y, cv=cv).mean() * 100
        rows.append({"model": name, "with duplicates": a, "without": b, "inflation": a - b})
    return pd.DataFrame(rows).set_index("model").round(2)


def adversarial_validation(seeds=(0, 1, 2)):
    """AUC of a classifier telling training rows from test rows (0.5 = indistinguishable)."""
    from lightgbm import LGBMClassifier
    D = load()
    A = np.vstack([D.X, D.X_test])
    t = np.r_[np.zeros(len(D.X), int), np.ones(len(D.X_test), int)]
    aucs = []
    for s in seeds:
        oof = np.zeros(len(t))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=s).split(A, t):
            m = LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, min_child_samples=40,
                               verbose=-1).fit(A[tr], t[tr])
            oof[va] = m.predict_proba(A[va])[:, 1]
        aucs.append(roc_auc_score(t, oof))
    return float(np.mean(aucs)), float(np.std(aucs))


# ==========================================================================
# 4. validation protocol and result storage
# ==========================================================================
@lru_cache(maxsize=4)
def make_folds(seeds=SCREEN_SEEDS):
    """Repeated stratified 5-fold CV; every validation fold carries its noisy copies."""
    D = load()
    folds = []
    for seed in seeds:
        for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(D.X, D.y)):
            sd = D.X[tr].std(0, ddof=1)
            variants = {name: [test_noise(D.X[va], s, share, sd, np.random.default_rng([seed, k, vi, r]))
                               for r in range(reps)]
                        for vi, (name, (s, share, reps)) in enumerate(VARIANTS.items())}
            folds.append({"key": f"{seed}_{k}", "tr": tr, "va": va, "variants": variants})
    return folds


def proba(model, Z):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Z)
    d = model.decision_function(Z)                      # margin models: softmax of the scores
    e = np.exp(d - d.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def stacked(f):
    return np.vstack([Z for Zs in f["variants"].values() for Z in Zs])


def unstacked(P_all, f):
    out, i, n = {}, 0, len(f["va"])
    for v, Zs in f["variants"].items():
        out[v] = []
        for _ in Zs:
            out[v].append(P_all[i:i + n])
            i += n
    return out


class Results:
    """Out-of-fold probabilities of every model on a set of folds:
    P[name] = one entry per fold, {variant: [probabilities for each noise draw]}."""

    def __init__(self, folds):
        self.folds, self.P = folds, {}

    def run(self, name, fit, n_jobs=-1, verbose=True):
        """fit(X_train, y_train) -> fitted model; scores every noisy copy of every fold."""
        X, y = load().X, load().y

        def one(f):
            m = fit(X[f["tr"]], y[f["tr"]])
            return {v: [proba(m, Z) for Z in Zs] for v, Zs in f["variants"].items()}
        t = time.time()
        self.P[name] = Parallel(n_jobs=n_jobs)(delayed(one)(f) for f in self.folds)
        if verbose:
            log(f"{name:<52} {self.table([name]).loc[name, 'test-like']:6.2f}%   ({time.time() - t:.0f}s)")

    def fold_acc(self, name):
        y = load().y
        return [{v: np.mean([(p.argmax(1) == y[f["va"]]).mean() for p in ps]) for v, ps in probs.items()}
                for f, probs in zip(self.folds, self.P[name])]

    def table(self, names=None, base=BASE):
        """Accuracy (%) per noise condition and the paired gap to the final model (+- one s.e.)."""
        names = list(self.P) if names is None else names
        b = (np.array([(a["u08"] + a["mix12"]) / 2 for a in self.fold_acc(base)])
             if base in self.P else None)
        rows = []
        for n in names:
            acc = self.fold_acc(n)
            tl = np.array([(a["u08"] + a["mix12"]) / 2 for a in acc])
            row = {"model": n, "clean": np.mean([a["clean"] for a in acc]) * 100,
                   "test-like": tl.mean() * 100, "stress 0.15": np.mean([a["u15"] for a in acc]) * 100}
            if b is not None:
                d = (tl - b[:len(tl)]) * 100
                row["gap vs final"], row["+- s.e."] = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
            rows.append(row)
        return pd.DataFrame(rows).set_index("model").round(2) + 0.0     # + 0.0: no "-0.00"

    def average(self, name, members):
        """Probability average of stored models (same folds, same noisy copies)."""
        first = self.P[members[0]]
        self.P[name] = [{v: [np.mean([self.P[m][i][v][r] for m in members], axis=0)
                             for r in range(len(ps))] for v, ps in first[i].items()}
                        for i in range(len(first))]


# ==========================================================================
# 5. building blocks: transformers and custom estimators
# ==========================================================================
def logreg(C=0.1, **kw):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=20000, **kw))


def lr_on(*steps, C=0.1, **kw):
    return make_pipeline(*steps, LogisticRegression(C=C, max_iter=20000, **kw))


def tree(**kw):
    return DecisionTreeClassifier(random_state=0, **kw)


def add_ratios(Z):
    """10 physics ratios (power/weight, hp per litre, ...)."""
    f = {n: i for i, n in enumerate(load().features)}
    g = lambda n: Z[:, f[n]]
    volume = g("vehicle_length_mm") * g("vehicle_width_mm") * g("vehicle_height_mm") / 1e9
    return np.hstack([Z, np.column_stack([
        g("horsepower_hp") / g("vehicle_weight_kg"), g("torque_nm") / g("vehicle_weight_kg"),
        g("horsepower_hp") / g("engine_displacement_l"), g("engine_displacement_l") / g("engine_cylinders"),
        volume, g("cargo_capacity_l") / volume, g("battery_capacity_kwh") / g("vehicle_weight_kg"),
        g("vehicle_length_mm") / g("vehicle_width_mm"), g("vehicle_height_mm") - g("ground_clearance_mm"),
        g("highway_fuel_efficiency_kmpl") / g("city_fuel_efficiency_kmpl")])])


def add_clip_flags(Z):
    """1 where a value sits exactly on a clip limit that holds > 1% of the training rows."""
    D = load()
    flags = [np.isclose(Z[:, j], b).astype(float) for j in range(D.X.shape[1])
             for b in (D.lo[j], D.hi[j]) if np.mean(D.X[:, j] == b) > 0.01]
    return np.hstack([Z, np.column_stack(flags)])


class WithLDA(BaseEstimator, ClassifierMixin):
    """Appends the two LDA discriminant scores (fitted on the training rows only), so
    trees can split along the tilted class boundary instead of building a staircase."""

    def __init__(self, base):
        self.base = base

    def _tf(self, Z):
        return np.hstack([Z, self.lda_.transform(self.sc_.transform(Z))])

    def fit(self, Z, t):
        self.sc_ = StandardScaler().fit(Z)
        self.lda_ = LDA(n_components=2).fit(self.sc_.transform(Z), t)
        self.base.fit(self._tf(Z), t)
        self.classes_ = np.unique(t)
        return self

    def predict_proba(self, Z):
        return self.base.predict_proba(self._tf(Z))

    def predict(self, Z):
        return self.predict_proba(Z).argmax(1)


class Uncensor(BaseEstimator, TransformerMixin):
    """Values exactly on a clip limit -> expected hidden value beyond the limit
    (truncated-normal mean around a linear prediction from the other features,
    fitted on unclipped rows)."""

    def fit(self, Xa, ya=None):
        D = load()
        self.jobs_ = []
        for j in range(Xa.shape[1]):
            lo, hi = Xa[:, j] == D.lo[j], Xa[:, j] == D.hi[j]
            if max(lo.mean(), hi.mean()) < 0.01:
                continue
            inside = ~(lo | hi)
            others = np.delete(np.arange(Xa.shape[1]), j)
            reg = LinearRegression().fit(Xa[inside][:, others], Xa[inside, j])
            self.jobs_.append((j, others, reg, np.std(Xa[inside, j] - reg.predict(Xa[inside][:, others]))))
        return self

    def transform(self, Z):
        D = load()
        Z = Z.copy()
        for j, others, reg, s in self.jobs_:
            m = reg.predict(Z[:, others])
            lo, hi = Z[:, j] == D.lo[j], Z[:, j] == D.hi[j]
            a, b = (D.lo[j] - m[lo]) / s, (D.hi[j] - m[hi]) / s
            Z[lo, j] = m[lo] - s * norm.pdf(a) / np.maximum(norm.cdf(a), 1e-12)
            Z[hi, j] = m[hi] + s * norm.pdf(b) / np.maximum(norm.sf(b), 1e-12)
        return Z


class CleanThenFit(BaseEstimator, ClassifierMixin):
    """Confident-learning style cleaning: drop training rows whose out-of-fold
    probability for their own label is below tau, then fit the final model."""

    def __init__(self, tau=0.05):
        self.tau = tau

    def fit(self, Xa, ya):
        P = cross_val_predict(logreg(0.1), Xa, ya, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              method="predict_proba")
        keep = P[np.arange(len(ya)), ya] >= self.tau
        self.m_ = logreg(0.1).fit(*jitter_copies(Xa[keep], ya[keep]))
        self.classes_ = self.m_.classes_
        return self

    def predict_proba(self, Z):
        return self.m_.predict_proba(Z)


class RDA(BaseEstimator, ClassifierMixin):
    """Friedman's regularised discriminant analysis: class covariances blended with
    the pooled one (alpha: 0 = LDA, 1 = QDA) and shrunk toward a scaled identity (gamma)."""

    def __init__(self, alpha=0.1, gamma=0.02):
        self.alpha, self.gamma = alpha, gamma

    def fit(self, Xa, ya):
        self.sc_ = StandardScaler().fit(Xa)
        Z = self.sc_.transform(Xa)
        p = Z.shape[1]
        self.mu_ = np.stack([Z[ya == k].mean(0) for k in range(3)])
        covs = [np.cov(Z[ya == k], rowvar=False) for k in range(3)]
        pooled = sum((np.sum(ya == k) - 1) * covs[k] for k in range(3)) / (len(ya) - 3)
        self.inv_, self.logdet_ = [], []
        for k in range(3):
            S = self.alpha * covs[k] + (1 - self.alpha) * pooled
            S = (1 - self.gamma) * S + self.gamma * np.trace(S) / p * np.eye(p)
            self.inv_.append(np.linalg.inv(S))
            self.logdet_.append(np.linalg.slogdet(S)[1])
        self.logprior_ = np.log(np.bincount(ya) / len(ya))
        self.classes_ = np.arange(3)
        return self

    def predict_proba(self, Za):
        Z = self.sc_.transform(Za)
        L = np.column_stack([-0.5 * np.einsum("ij,jk,ik->i", Z - self.mu_[k], self.inv_[k], Z - self.mu_[k])
                             - 0.5 * self.logdet_[k] + self.logprior_[k] for k in range(3)])
        P = np.exp(L - L.max(1, keepdims=True))
        return P / P.sum(1, keepdims=True)


class MixtureDA(BaseEstimator, ClassifierMixin):
    """Mixture discriminant analysis: a Gaussian mixture per class."""

    def __init__(self, k=4, cov="tied"):
        self.k, self.cov = k, cov

    def fit(self, Xa, ya):
        self.sc_ = StandardScaler().fit(Xa)
        Z = self.sc_.transform(Xa)
        self.prior_ = np.bincount(ya) / len(ya)
        self.gm_ = [GaussianMixture(self.k, covariance_type=self.cov, reg_covar=1e-2, n_init=3,
                                    random_state=0).fit(Z[ya == c]) for c in range(3)]
        self.classes_ = np.arange(3)
        return self

    def predict_proba(self, Za):
        Z = self.sc_.transform(Za)
        L = np.column_stack([g.score_samples(Z) for g in self.gm_]) + np.log(self.prior_)
        P = np.exp(L - L.max(1, keepdims=True))
        return P / P.sum(1, keepdims=True)


def jitter(Xt, yt):
    return jitter_copies(Xt, yt)


def gam(C=0.05):
    return make_pipeline(SplineTransformer(n_knots=5, degree=3, extrapolation="linear"),
                         StandardScaler(), LogisticRegression(C=C, max_iter=20000))


def xgb(depth, n, rate):
    from xgboost import XGBClassifier
    return XGBClassifier(device="cuda", tree_method="hist", n_estimators=n, max_depth=depth,
                         learning_rate=rate, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=5, verbosity=0)


# ==========================================================================
# 6. model registry: name -> (fit(X, y) -> model, joblib jobs)
#    jobs = 1 for models that use the GPU or their own threads
# ==========================================================================
def feature_engineering_experiments():
    """Section 5 of the report: ideas placed in front of the final logistic regression."""
    return {
        "10 physics ratios (power/weight, hp per litre, ...)":
            lambda Xt, yt: lr_on(FunctionTransformer(add_ratios), StandardScaler()).fit(*jitter(Xt, yt)),
        "Degree-2 polynomial features (230)":
            lambda Xt, yt: lr_on(PolynomialFeatures(2, include_bias=False), StandardScaler(), C=0.01).fit(*jitter(Xt, yt)),
        "Spline expansion (GAM, 5 knots)": lambda Xt, yt: gam(0.05).fit(*jitter(Xt, yt)),
        "Clip-limit indicator flags":
            lambda Xt, yt: lr_on(FunctionTransformer(add_clip_flags), StandardScaler()).fit(
                *test_noise_copies(Xt, yt, 0.12, 0.5)),
        "Yeo-Johnson power transform": lambda Xt, yt: lr_on(PowerTransformer(), C=0.3).fit(*jitter(Xt, yt)),
        "Rank-Gauss (quantile) transform":
            lambda Xt, yt: lr_on(QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                                     random_state=0), StandardScaler()).fit(*jitter(Xt, yt)),
        "Robust scaling (median / IQR)": lambda Xt, yt: lr_on(RobustScaler()).fit(*jitter(Xt, yt)),
        "Un-censoring clipped values": lambda Xt, yt: lr_on(Uncensor(), StandardScaler()).fit(*jitter(Xt, yt)),
        "PCA denoising (17 components)": lambda Xt, yt: lr_on(StandardScaler(), PCA(17)).fit(*jitter(Xt, yt)),
        "Feature selection (top 17 by ANOVA)":
            lambda Xt, yt: lr_on(StandardScaler(), SelectKBest(f_classif, k=17)).fit(*jitter(Xt, yt)),
        "Lasso (L1) logistic regression":
            lambda Xt, yt: lr_on(StandardScaler(), C=0.05, penalty="l1", solver="saga", tol=1e-3).fit(*jitter(Xt, yt)),
        "Data cleaning (drop implausible rows)": lambda Xt, yt: CleanThenFit(0.05).fit(Xt, yt),
        "Training on 10 test-noise copies": lambda Xt, yt: logreg(0.1).fit(*alternating_noise_copies(Xt, yt, 10)),
    }


def basic_models():
    """Section 7: logistic regression (baseline -> tuned), decision tree, Naive Bayes."""
    m = {
        "Logistic regression - baseline (C = 1)": (lambda Xt, yt: logreg(1.0).fit(Xt, yt), -1),
        BASE: (lambda Xt, yt: logreg(0.1).fit(*jitter(Xt, yt)), -1),
    }
    for C in (0.01, 0.03, 0.3, 1.0):
        m[f"Logistic regression C = {C} + augmentation"] = (lambda Xt, yt, C=C: logreg(C).fit(*jitter(Xt, yt)), -1)
    m.update({
        "Logistic regression one-vs-rest": (lambda Xt, yt: make_pipeline(
            StandardScaler(), OneVsRestClassifier(LogisticRegression(C=0.1, max_iter=20000))).fit(*jitter(Xt, yt)), -1),
        "Linear SVM (hinge loss)": (lambda Xt, yt: make_pipeline(
            StandardScaler(), LinearSVC(C=0.05, dual=False)).fit(*jitter(Xt, yt)), -1),
        "Ridge classifier (least squares)": (lambda Xt, yt: make_pipeline(
            StandardScaler(), RidgeClassifier(alpha=10.0)).fit(*jitter(Xt, yt)), -1),
        "Decision tree - depth 6": (lambda Xt, yt: tree(max_depth=6, min_samples_leaf=20).fit(Xt, yt), -1),
        "Decision tree - depth 10": (lambda Xt, yt: tree(max_depth=10, min_samples_leaf=10).fit(Xt, yt), -1),
        "Decision tree - depth 4 + LDA features":
            (lambda Xt, yt: WithLDA(tree(max_depth=4, min_samples_leaf=20)).fit(Xt, yt), -1),
        "Gaussian Naive Bayes (raw features)": (lambda Xt, yt: GaussianNB().fit(Xt, yt), -1),
        "Naive Bayes + PCA whitening (10 comps)": (lambda Xt, yt: make_pipeline(
            StandardScaler(), PCA(10, whiten=True, random_state=SEED), GaussianNB()).fit(*jitter(Xt, yt)), -1),
        "Naive Bayes + LDA(2) projection + aug.": (lambda Xt, yt: make_pipeline(
            StandardScaler(), LDA(n_components=2), GaussianNB()).fit(*jitter(Xt, yt)), -1),
    })
    return m


def advanced_models(gpu=True):
    """Section 8: discriminant analysis, SVM, bagging, forests, boosting."""
    rf = lambda **k: RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=0, **k)
    m = {
        "LDA, covariance shrinkage 0.1": (lambda Xt, yt: make_pipeline(
            StandardScaler(), LDA(solver="lsqr", shrinkage=0.1)).fit(Xt, yt), -1),
        "QDA, reg_param 0.1": (lambda Xt, yt: make_pipeline(StandardScaler(), QDA(reg_param=0.1)).fit(Xt, yt), -1),
        "Regularised DA (alpha = 0.1)": (lambda Xt, yt: RDA(0.1, 0.02).fit(Xt, yt), -1),
        "Mixture DA (4 tied Gaussians / class)": (lambda Xt, yt: MixtureDA(4, "tied").fit(Xt, yt), -1),
        "SVM, RBF kernel, C = 1": (lambda Xt, yt: make_pipeline(
            StandardScaler(), SVC(C=1.0, probability=True, random_state=0)).fit(Xt, yt), -1),
        "Spline GAM + logistic regression": (lambda Xt, yt: gam(0.05).fit(*jitter(Xt, yt)), -1),
        "Bagged decision trees (300)": (lambda Xt, yt: BaggingClassifier(
            tree(), n_estimators=300, max_samples=0.8, n_jobs=1, random_state=0).fit(Xt, yt), -1),
        "Bagged logistic regression (50)": (lambda Xt, yt: BaggingClassifier(
            make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
            n_estimators=50, max_samples=0.8, n_jobs=1, random_state=0).fit(Xt, yt), -1),
        "Random forest (500 trees)": (lambda Xt, yt: rf().fit(Xt, yt), -1),
        "Random forest + LDA features + noise aug.":
            (lambda Xt, yt: WithLDA(rf(min_samples_leaf=5)).fit(*test_noise_copies(Xt, yt, 0.15)), -1),
        "Extra-trees + LDA features": (lambda Xt, yt: WithLDA(ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=5, n_jobs=1, random_state=0)).fit(Xt, yt), -1),
        "AdaBoost (depth-2 trees)": (lambda Xt, yt: AdaBoostClassifier(
            tree(max_depth=2), n_estimators=300, learning_rate=0.3, random_state=0).fit(Xt, yt), -1),
        "Gradient boosting (scikit-learn)": (lambda Xt, yt: GradientBoostingClassifier(
            n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0).fit(Xt, yt), -1),
        "HistGradientBoosting": (lambda Xt, yt: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=500, l2_regularization=1.0, random_state=0).fit(Xt, yt), -1),
        "Yeo-Johnson + logistic regression": (lambda Xt, yt: make_pipeline(
            PowerTransformer(), LogisticRegression(C=0.1, max_iter=20000)).fit(*jitter(Xt, yt)), -1),
        "Polynomial (degree-2) logistic regression": (lambda Xt, yt: make_pipeline(
            PolynomialFeatures(2, include_bias=False), StandardScaler(),
            LogisticRegression(C=0.01, max_iter=20000)).fit(*jitter(Xt, yt)), -1),
        "Spline GAM on 5 test-noise copies": (lambda Xt, yt: gam(0.05).fit(*alternating_noise_copies(Xt, yt, 5)), -1),
    }
    try:
        from lightgbm import LGBMClassifier
        m["LightGBM + LDA features + noise aug."] = (lambda Xt, yt: WithLDA(LGBMClassifier(
            n_estimators=600, num_leaves=7, learning_rate=0.03, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, min_child_samples=20, verbose=-1, n_jobs=8)).fit(
            *test_noise_copies(Xt, yt, 0.15)), 1)
    except ImportError:
        log("lightgbm not installed - skipped")
    if gpu:
        from catboost import CatBoostClassifier
        m["XGBoost on GPU"] = (lambda Xt, yt: xgb(2, 1200, 0.05).fit(Xt, yt), 1)
        m["XGBoost on GPU + LDA features"] = (lambda Xt, yt: WithLDA(xgb(2, 600, 0.03)).fit(Xt, yt), 1)
        m["CatBoost on GPU + LDA features"] = (lambda Xt, yt: WithLDA(CatBoostClassifier(
            task_type="GPU", iterations=1000, depth=4, learning_rate=0.03, l2_leaf_reg=3, verbose=0,
            random_seed=0, allow_writing_files=False)).fit(Xt, yt), 1)
    return m


# ==========================================================================
# 7. neural networks on the GPU: batched MLPs, FT-Transformer, TabPFN
# ==========================================================================
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    HAVE_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
except ImportError:                                     # pragma: no cover
    HAVE_TORCH = False

MLP_CONFIGS = {   # name: (width, depth, settings)
    "MLP 64x2, input noise 0.2, dropout 0.1":  (64, 2, dict(noise=0.2, share=1.0, wd=0.01, drop=0.1)),
    "MLP 64x2, no regularisation":             (64, 2, dict(noise=0.0, share=1.0, wd=1e-4, drop=0.0)),
    "MLP 256x2, input noise 0.2, dropout 0.1": (256, 2, dict(noise=0.2, share=1.0, wd=0.01, drop=0.1)),
    "MLP 256x3, input noise 0.3, dropout 0.1": (256, 3, dict(noise=0.3, share=1.0, wd=0.01, drop=0.1)),
}

if HAVE_TORCH:
    class BatchedMLP(nn.Module):
        """M independent MLPs evaluated together: x has shape (M, batch, features)."""

        def __init__(self, M, d_in, width, depth, n_out=3, seed=0):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            dims = [d_in] + [width] * depth + [n_out]
            self.W, self.b = nn.ParameterList(), nn.ParameterList()
            for a, b in zip(dims[:-1], dims[1:]):
                bound = np.sqrt(6.0 / a) if b != n_out else np.sqrt(1.0 / a)
                self.W.append(nn.Parameter((torch.rand(M, a, b, generator=g) * 2 - 1) * bound))
                self.b.append(nn.Parameter(torch.zeros(M, 1, b)))

        def forward(self, x, keep=None):
            for i, (W, b) in enumerate(zip(self.W, self.b)):
                x = torch.baddbmm(b, x, W)
                if i < len(self.W) - 1:
                    x = Fn.silu(x)
                    if keep is not None:
                        x = x * (torch.rand_like(x) < keep).float() / keep
            return x

    def train_mlps(train_sets, eval_sets, width, depth, settings, n_seeds=3, epochs=80, batch=256,
                   lr_max=3e-3):
        """Train len(settings) x len(train_sets) x n_seeds networks in one pass. Every
        network has its own weights, data, input noise, weight decay and dropout but they
        share the batched matmuls. Returns {(setting, set): seed-averaged probabilities}."""
        D = load()
        combos = [(si, ti, s) for si in range(len(settings)) for ti in range(len(train_sets))
                  for s in range(n_seeds)]
        M = len(combos)
        t_idx = torch.tensor([c[1] for c in combos], device=DEVICE)
        mu = np.stack([Xt.mean(0) for Xt, _ in train_sets])
        sd = np.stack([Xt.std(0) for Xt, _ in train_sets])
        n_max = max(len(Xt) for Xt, _ in train_sets)
        pad = lambda A: np.concatenate([A, A[:n_max - len(A)]]) if len(A) < n_max else A
        XS = torch.tensor(np.stack([pad((Xt - mu[i]) / sd[i]) for i, (Xt, _) in enumerate(train_sets)]),
                          dtype=torch.float32, device=DEVICE)[t_idx]
        YS = torch.tensor(np.stack([pad(yt) for _, yt in train_sets]), device=DEVICE)[t_idx]
        n_tr = torch.tensor([len(train_sets[c[1]][0]) for c in combos], device=DEVICE)[:, None]
        lo = torch.tensor((D.lo - mu) / sd, dtype=torch.float32, device=DEVICE)[t_idx][:, None]
        hi = torch.tensor((D.hi - mu) / sd, dtype=torch.float32, device=DEVICE)[t_idx][:, None]
        par = lambda k: torch.tensor([settings[c[0]][k] for c in combos], dtype=torch.float32,
                                     device=DEVICE)[:, None, None]
        s_noise, share, wd, keep = par("noise"), par("share"), par("wd"), 1 - par("drop")
        keep = keep if bool((keep < 1).any()) else None
        net = BatchedMLP(M, XS.shape[2], width, depth, seed=1000 * width + depth).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=lr_max)
        steps = epochs * (n_max // batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr_max, total_steps=steps, pct_start=0.1)
        gen = torch.Generator(device=DEVICE).manual_seed(7 * width + depth)
        for _ in range(steps):
            idx = (torch.rand(M, batch, device=DEVICE, generator=gen) * n_tr).long()
            xb = torch.gather(XS, 1, idx[..., None].expand(-1, -1, XS.shape[2]))
            yb = torch.gather(YS, 1, idx)
            e = torch.randn(xb.shape, device=DEVICE, generator=gen) * s_noise
            e = e * (torch.rand(xb.shape, device=DEVICE, generator=gen) < share).float()
            xb = torch.minimum(torch.maximum(xb + e, lo), hi)        # re-clip like the test file
            loss = Fn.cross_entropy(net(xb, keep).reshape(-1, 3), yb.reshape(-1), reduction="none")
            loss = loss.view(M, batch).mean(1).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():                                    # decoupled, per-network decay
                for p in net.W:
                    p.mul_(1 - sched.get_last_lr()[0] * wd)
            sched.step()
        n_ev = max(len(Z) for Z in eval_sets)
        pad_ev = lambda A: np.concatenate([A, A[:n_ev - len(A)]]) if len(A) < n_ev else A
        ZE = torch.tensor(np.stack([pad_ev((Z - mu[i]) / sd[i]) for i, Z in enumerate(eval_sets)]),
                          dtype=torch.float32, device=DEVICE)
        P = torch.empty(M, n_ev, 3)
        with torch.no_grad():
            for i in range(0, n_ev, 1000):
                P[:, i:i + 1000] = torch.softmax(net(ZE[t_idx, i:i + 1000]), -1).cpu()
        P = P.numpy()
        return {(si, ti): P[[m for m, c in enumerate(combos) if c[:2] == (si, ti)], :len(eval_sets[ti])].mean(0)
                for si in range(len(settings)) for ti in range(len(eval_sets))}

    class FTTransformer(nn.Module):
        """FT-Transformer: each feature becomes a token; a [CLS] token is read out."""

        def __init__(self, n_feat, d=32, layers=2, heads=8, drop=0.1):
            super().__init__()
            self.w = nn.Parameter(torch.randn(n_feat, d) / d ** 0.5)
            self.b = nn.Parameter(torch.zeros(n_feat, d))
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.GELU(), nn.Linear(d, 3))

        def forward(self, x):
            tok = torch.cat([self.cls.expand(len(x), -1, -1), x[..., None] * self.w + self.b], 1)
            return self.head(self.enc(tok)[:, 0])

    def ft_fit_predict(Xtr, ytr, Z, d, layers, noise, epochs=60, seed=0):
        D = load()
        torch.manual_seed(seed)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        lo = torch.tensor((D.lo - mu) / sd, dtype=torch.float32, device=DEVICE)
        hi = torch.tensor((D.hi - mu) / sd, dtype=torch.float32, device=DEVICE)
        XT = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=DEVICE)
        YT = torch.tensor(ytr, device=DEVICE)
        net = FTTransformer(Xtr.shape[1], d, layers).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        steps = epochs * (len(XT) // 256)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
        net.train()
        for _ in range(steps):
            idx = torch.randint(0, len(XT), (256,), device=DEVICE)
            xb = torch.minimum(torch.maximum(XT[idx] + torch.randn_like(XT[idx]) * noise, lo), hi)
            loss = Fn.cross_entropy(net(xb), YT[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            Zt = torch.tensor((Z - mu) / sd, dtype=torch.float32, device=DEVICE)
            return torch.cat([torch.softmax(net(Zt[i:i + 2048]), -1)
                              for i in range(0, len(Zt), 2048)]).cpu().numpy()


def run_mlps(res, configs=None):
    D = load()
    configs = configs or MLP_CONFIGS
    for arch in sorted({(w, d) for w, d, _ in configs.values()}):
        names = [n for n, (w, d, _) in configs.items() if (w, d) == arch]
        t = time.time()
        probs = train_mlps([(D.X[f["tr"]], D.y[f["tr"]]) for f in res.folds], [stacked(f) for f in res.folds],
                           *arch, [configs[n][2] for n in names])
        for si, n in enumerate(names):
            res.P[n] = [unstacked(probs[(si, i)], f) for i, f in enumerate(res.folds)]
        log(f"MLP {arch[0]}x{arch[1]}: {len(names) * len(res.folds) * 3} networks in {time.time() - t:.0f}s")


FT_CONFIGS = {   # name: (token width, layers, input noise)
    "FT-Transformer (small: d=32, 2 layers)": (32, 2, 0.20),
    "FT-Transformer (large: d=64, 3 layers)": (64, 3, 0.15),
}
TABPFN = "TabPFN v2 (5 folds)"


def run_ft_transformers(res):
    D = load()
    for name, (d, layers, noise) in FT_CONFIGS.items():
        t = time.time()
        res.P[name] = [unstacked(ft_fit_predict(D.X[f["tr"]], D.y[f["tr"]], stacked(f), d, layers, noise), f)
                       for f in res.folds]
        log(f"{name:<52} {res.table([name]).loc[name, 'test-like']:6.2f}%   ({time.time() - t:.0f}s)")


def tabpfn_model():
    """TabPFN v2: the openly licensed weights (Prior Labs License 1.1), no account needed."""
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, device=DEVICE.type, n_estimators=4, memory_saving_mode=True,
        fit_mode="fit_preprocessors", random_state=0)


def run_tabpfn(res, n_folds=5):
    """Slow (~80 s per fold on an RTX 4060): the 5 folds of the first CV repeat."""
    D = load()
    t = time.time()
    res.P[TABPFN] = []
    for f in res.folds[:n_folds]:
        m = tabpfn_model().fit(D.X[f["tr"]], D.y[f["tr"]])
        Zs = stacked(f)
        res.P[TABPFN].append(unstacked(
            np.vstack([m.predict_proba(Zs[i:i + 1500]) for i in range(0, len(Zs), 1500)]), f))
    log(f"TabPFN v2 on {n_folds} folds ({time.time() - t:.0f}s)")


# ==========================================================================
# 8. ensembles, stacking, threshold tuning (cross-fitted inside each CV repeat)
# ==========================================================================
ENSEMBLES = {
    "avg: LogReg + spline GAM (5 noisy copies)": [BASE, "Spline GAM on 5 test-noise copies"],
    "avg: LogReg + polynomial LogReg": [BASE, "Polynomial (degree-2) logistic regression"],
    "avg: LogReg + LDA + Naive Bayes": [BASE, "LDA, covariance shrinkage 0.1", "Naive Bayes + LDA(2) projection + aug."],
    "avg: LogReg + SVM": [BASE, "SVM, RBF kernel, C = 1"],
    "avg: LogReg + MLP": [BASE, "MLP 64x2, input noise 0.2, dropout 0.1"],
    "avg: LogReg + random forest": [BASE, "Random forest + LDA features + noise aug."],
    "avg: LogReg + XGBoost": [BASE, "XGBoost on GPU + LDA features"],
    "avg: LogReg + CatBoost": [BASE, "CatBoost on GPU + LDA features"],
}
STACK_BASES = [BASE, "LDA, covariance shrinkage 0.1", "Naive Bayes + LDA(2) projection + aug.",
               "SVM, RBF kernel, C = 1", "MLP 64x2, input noise 0.2, dropout 0.1",
               "Random forest + LDA features + noise aug.", "XGBoost on GPU + LDA features",
               "CatBoost on GPU + LDA features", "Spline GAM + logistic regression", "QDA, reg_param 0.1",
               "FT-Transformer (small: d=32, 2 layers)", "Yeo-Johnson + logistic regression"]
TRAIN_V = [("u08", 0), ("u08", 1), ("mix12", 0), ("mix12", 1)]      # the test-like noisy copies
OFFSETS = np.round(np.arange(-0.40, 0.401, 0.02), 2)


def cross_fit(res, fit_apply):
    """For each fold: learn on the out-of-fold predictions of the other 4 folds of its repeat."""
    out = [None] * len(res.folds)
    for r in range(0, len(res.folds), 5):
        rep = list(range(r, r + 5))
        for i in rep:
            out[i] = fit_apply([j for j in rep if j != i], i)
    return out


def stacking(res, bases):
    y = load().y

    def fold(others, i):
        feats = lambda j, v, k: np.hstack([np.log(res.P[b][j][v][k] + 1e-12) for b in bases])
        Xm = np.vstack([feats(j, v, k) for j in others for v, k in TRAIN_V])
        tm = np.concatenate([y[res.folds[j]["va"]] for j in others for _ in TRAIN_V])
        meta = LogisticRegression(C=0.1, max_iter=5000).fit(Xm, tm)
        return {v: [meta.predict_proba(feats(i, v, k)) for k in range(len(ps))] for v, ps in res.P[BASE][i].items()}
    return cross_fit(res, fold)


def threshold_tuning(res):
    """Per-class offsets on the log-probabilities, tuned for accuracy."""
    y = load().y

    def fold(others, i):
        L = np.vstack([np.log(res.P[BASE][j][v][k] + 1e-12) for j in others for v, k in TRAIN_V])
        t = np.concatenate([y[res.folds[j]["va"]] for j in others for _ in TRAIN_V])
        best, best_acc = (0.0, 0.0), -1
        for d1 in OFFSETS:
            for d2 in OFFSETS:
                acc = ((L + [0.0, d1, d2]).argmax(1) == t).mean()
                if acc > best_acc + 1e-12 or (abs(acc - best_acc) < 1e-12 and
                                              abs(d1) + abs(d2) < abs(best[0]) + abs(best[1])):
                    best, best_acc = (d1, d2), acc
        shift = lambda p: (lambda q: q / q.sum(1, keepdims=True))(np.exp(np.log(p + 1e-12) + [0.0, *best]))
        return {v: [shift(p) for p in ps] for v, ps in res.P[BASE][i].items()}
    return cross_fit(res, fold)


def run_ensembles(res):
    for name, members in ENSEMBLES.items():
        if all(m in res.P for m in members):
            res.average(name, members)
    bases = [b for b in STACK_BASES if b in res.P]
    res.P[f"Stacking: meta-model over {len(bases)} models"] = stacking(res, bases)
    res.P["Per-class threshold tuning"] = threshold_tuning(res)


# ==========================================================================
# 9. experiments: noise robustness, 20% of the data, clustering
# ==========================================================================
LEVELS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)


def robustness_curves(folds):
    """Accuracy (%) as increasing noise is added to the validation rows of `folds`."""
    D = load()
    models = {
        "Logistic regression": lambda Xt, yt: logreg(0.1).fit(*jitter(Xt, yt)),
        "Naive Bayes + LDA(2)": lambda Xt, yt: make_pipeline(StandardScaler(), LDA(n_components=2),
                                                             GaussianNB()).fit(*jitter(Xt, yt)),
        "QDA (reg 0.1)": lambda Xt, yt: make_pipeline(StandardScaler(), QDA(reg_param=0.1)).fit(Xt, yt),
        "Random forest": lambda Xt, yt: RandomForestClassifier(500, n_jobs=-1, random_state=0).fit(Xt, yt),
        "Decision tree (depth 6)": lambda Xt, yt: tree(max_depth=6, min_samples_leaf=20).fit(Xt, yt),
        "Gaussian Naive Bayes": lambda Xt, yt: GaussianNB().fit(Xt, yt),
    }
    curves = {}
    for name, fit in models.items():
        accs = {s: [] for s in LEVELS}
        for f in folds:
            m = fit(D.X[f["tr"]], D.y[f["tr"]])
            sd = D.X[f["tr"]].std(0, ddof=1)
            for s in LEVELS:
                Zv = test_noise(D.X[f["va"]], s, 1.0, sd, np.random.default_rng([42, 7, int(s * 100)]))
                accs[s].append((proba(m, Zv).argmax(1) == D.y[f["va"]]).mean() * 100)
        curves[name] = {f"noise {s:g}": np.mean(v) for s, v in accs.items()}
    out = pd.DataFrame(curves).T
    out.index.name = "model"
    return out.round(2)


def twenty_percent(folds, gpu=True):
    """Each model on a stratified 20% of every training fold, scored on the same noisy validation rows."""
    D = load()
    rng = np.random.default_rng(0)
    sub = {}
    for f in folds:
        tr = f["tr"]
        sub[f["key"]] = np.sort(np.concatenate([rng.choice(tr[D.y[tr] == c], int(round(0.2 * np.sum(D.y[tr] == c))),
                                                           replace=False) for c in range(3)]))

    def test_like(m, f):
        return np.mean([(proba(m, Z).argmax(1) == D.y[f["va"]]).mean() for v in ("u08", "mix12")
                        for Z in f["variants"][v]]) * 100
    models = {
        "Logistic regression": lambda Xt, yt: logreg(0.1).fit(*jitter(Xt, yt)),
        "Decision tree (depth 6)": lambda Xt, yt: tree(max_depth=6, min_samples_leaf=20).fit(Xt, yt),
        "Gaussian Naive Bayes": lambda Xt, yt: GaussianNB().fit(Xt, yt),
        "Naive Bayes + LDA(2)": lambda Xt, yt: make_pipeline(StandardScaler(), LDA(n_components=2),
                                                             GaussianNB()).fit(*jitter(Xt, yt)),
        "QDA (reg 0.1)": lambda Xt, yt: make_pipeline(StandardScaler(), QDA(reg_param=0.1)).fit(Xt, yt),
        "SVM (RBF, C=1)": lambda Xt, yt: make_pipeline(StandardScaler(), SVC(C=1.0, random_state=0)).fit(Xt, yt),
        "Random forest": lambda Xt, yt: RandomForestClassifier(500, n_jobs=-1, random_state=0).fit(Xt, yt),
    }
    if gpu:
        models["XGBoost + LDA features"] = lambda Xt, yt: WithLDA(xgb(2, 600, 0.03)).fit(Xt, yt)
    rows = {}
    for name, fit in models.items():
        rows[name] = {"100% data": np.mean([test_like(fit(D.X[f["tr"]], D.y[f["tr"]]), f) for f in folds]),
                      "20% data": np.mean([test_like(fit(D.X[sub[f["key"]]], D.y[sub[f["key"]]]), f) for f in folds])}
    if gpu and HAVE_TORCH:
        evals = [np.vstack([Z for v in ("u08", "mix12") for Z in f["variants"][v]]) for f in folds]
        for label, idx_of, epochs in (("100% data", lambda f: f["tr"], 80), ("20% data", lambda f: sub[f["key"]], 400)):
            probs = train_mlps([(D.X[idx_of(f)], D.y[idx_of(f)]) for f in folds], evals, 64, 2,
                               [MLP_CONFIGS["MLP 64x2, input noise 0.2, dropout 0.1"][2]], epochs=epochs)
            rows.setdefault("Neural network (MLP 64x2)", {})[label] = np.mean(
                [(probs[(0, i)].argmax(1) == np.tile(D.y[f["va"]], 4)).mean() * 100 for i, f in enumerate(folds)])
    out = pd.DataFrame(rows).T
    out["change"] = out["20% data"] - out["100% data"]
    out.index.name = "model"
    return out.sort_values("100% data", ascending=False).round(2), len(folds[0]["tr"]), len(sub[folds[0]["key"]])


def clustering():
    """K-Means, Gaussian mixture, Ward and DBSCAN on the standardised features, vs the true classes."""
    D = load()
    Z = StandardScaler().fit_transform(D.X)
    eps = float(np.quantile(np.sort(NearestNeighbors(n_neighbors=11).fit(Z).kneighbors(Z)[0][:, 10]), 0.90))

    def matched(labels):
        ks = [k for k in np.unique(labels) if k != -1]
        M = np.array([[np.sum((labels == k) & (D.y == c)) for c in range(3)] for k in ks])
        r, c = linear_sum_assignment(-M)
        return M[r, c].sum() / len(D.y) * 100
    algos = {
        "K-Means (k=3)": lambda: KMeans(3, n_init=10, random_state=0).fit_predict(Z),
        "Gaussian mixture (k=3)": lambda: GaussianMixture(3, covariance_type="full", n_init=3,
                                                          random_state=0).fit(Z).predict(Z),
        "Agglomerative (Ward, k=3)": lambda: AgglomerativeClustering(3, linkage="ward").fit_predict(Z),
        f"DBSCAN (eps = {eps:.2f})": lambda: DBSCAN(eps=eps, min_samples=10).fit_predict(Z),
    }
    rows = []
    for name, fit in algos.items():
        lab = fit()
        n_cl = len([k for k in np.unique(lab) if k != -1])
        rows.append({"algorithm": name, "clusters": n_cl, "noise points": int((lab == -1).sum()),
                     "silhouette": silhouette_score(Z[lab != -1], lab[lab != -1]) if n_cl > 1 else np.nan,
                     "ARI": adjusted_rand_score(D.y, lab), "homogeneity": homogeneity_score(D.y, lab),
                     "matched accuracy %": matched(lab)})
    return pd.DataFrame(rows).set_index("algorithm").round(3)


# ==========================================================================
# 10. final model, submission, confusion matrix
# ==========================================================================
def final_model():
    """Logistic regression, C = 0.1, standardised features, trained on the training
    rows plus four noise-jittered copies."""
    D = load()
    return logreg(0.1).fit(*jitter_copies(D.X, D.y))


def write_submission(path=None):
    """Refit the final model on all training rows and predict the test file;
    returns the predicted class mix."""
    D = load()
    path = Path(path) if path else RESULTS / "submission_final_logreg.csv"
    path.parent.mkdir(exist_ok=True)
    pred = final_model().predict(D.X_test).astype(int)
    sub = pd.DataFrame({ID_COL: D.test_ids, TARGET: pred})
    sample = DATA / "vehicle_sample_submission.csv"
    if sample.exists():
        s = pd.read_csv(sample)
        assert sub.shape == s.shape and (sub[ID_COL].to_numpy() == s[ID_COL].to_numpy()).all()
    sub.to_csv(path, index=False)
    mix = {CLASS_NAMES[c]: round(float((pred == c).mean()), 4) for c in range(3)}
    log(f"wrote {path.name}  predicted mix: " + ", ".join(f"{k} {v:.1%}" for k, v in mix.items()))
    return mix


def confusion(res, n_folds=5):
    """Out-of-fold confusion matrix and per-class metrics of the final model under
    test-like noise (the first CV repeat: every vehicle once, two noise readings)."""
    y = load().y
    pred, true = [], []
    for i, f in enumerate(res.folds[:n_folds]):
        for v in ("u08", "mix12"):
            pred.append(res.P[BASE][i][v][0].argmax(1))
            true.append(y[f["va"]])
    pred, true = np.concatenate(pred), np.concatenate(true)
    names = [CLASS_NAMES[c] for c in range(3)]
    cm = pd.DataFrame((confusion_matrix(true, pred) / 2).round().astype(int), index=names, columns=names)
    p, r, f1, _ = precision_recall_fscore_support(true, pred)
    cm["precision"], cm["recall"], cm["F1"] = p.round(3), r.round(3), f1.round(3)
    cm.index.name = "true / predicted"
    return cm


# ==========================================================================
# 11. results: one CSV per metrics table (the numbers in Model_Metrics.pdf)
# ==========================================================================
def save_table(df, name):
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / f"{name}.csv")
    return df


def first_folds(res, n):
    """A view of `res` restricted to its first n folds."""
    out = Results(res.folds[:n])
    out.P = {k: v[:n] for k, v in res.P.items()}
    return out


# ==========================================================================
# 12. run everything
# ==========================================================================
CONFIRM_MODELS = [BASE, "Naive Bayes + LDA(2) projection + aug.", "LDA, covariance shrinkage 0.1",
                  "Polynomial (degree-2) logistic regression", "Spline GAM on 5 test-noise copies",
                  "Yeo-Johnson power transform", "Un-censoring clipped values"]


def main(quick=False, gpu=True, tabpfn=True, final_only=False):
    t0 = time.time()
    D = load()
    log(f"{len(D.X):,} training vehicles (after removing {len(D.X_dup) - len(D.X):,} duplicates), "
        f"{len(D.X_test):,} test vehicles, {len(D.features)} features")
    pred_mix = write_submission()
    if final_only:
        return None
    gpu = gpu and HAVE_TORCH and torch.cuda.is_available()
    if not gpu:
        log("no GPU: XGBoost, CatBoost, MLPs, FT-Transformer and TabPFN are skipped")

    log("train vs test checks")
    cal = noise_calibration()
    try:
        auc, auc_sd = adversarial_validation()
    except ImportError:
        auc = auc_sd = float("nan")
    save_table(pd.DataFrame({"value": [cal["pairs"], cal["observed_ratio"], cal["noise_sd"], auc, auc_sd]},
                            index=pd.Index(["pairs", "corr_ratio", "noise_sd", "adv_auc", "adv_auc_sd"],
                                           name="check")), "train_vs_test")
    save_table(duplicate_leakage(), "duplicates")

    folds = make_folds(SCREEN_SEEDS[:1] if quick else SCREEN_SEEDS)
    res = Results(folds)
    log(f"{len(folds)} folds - basic models")
    basic = basic_models()
    for name, (fit, jobs) in basic.items():
        res.run(name, fit, n_jobs=jobs)
    save_table(res.table(list(basic)), "basic_models")

    log("advanced models")
    adv = advanced_models(gpu)
    for name, (fit, jobs) in adv.items():
        res.run(name, fit, n_jobs=jobs)
    adv_names = list(adv)
    if gpu:
        run_mlps(res)
        adv_names += list(MLP_CONFIGS)
        if not quick:
            run_ft_transformers(res)
            adv_names += list(FT_CONFIGS)
    save_table(res.table(adv_names).sort_values("test-like", ascending=False), "advanced_models")
    if gpu and tabpfn and not quick:
        run_tabpfn(res)
        save_table(first_folds(res, 5).table([BASE, TABPFN]), "tabpfn")

    log("ensembles, stacking, threshold tuning")
    run_ensembles(res)
    save_table(res.table([n for n in res.P if n.startswith(("avg:", "Stacking", "Per-class"))]), "ensembles")
    save_table(confusion(res), "confusion")

    log("feature engineering and alternative preprocessing")
    fe = Results(folds)
    fe.P[BASE] = res.P[BASE]
    experiments = feature_engineering_experiments()
    for name, fit in experiments.items():
        fe.run(name, fit)
    save_table(fe.table([BASE] + list(experiments)), "feature_engineering")

    log("fresh-fold confirmation")
    conf = Results(make_folds(CONFIRM_SEEDS[:1] if quick else CONFIRM_SEEDS))
    pool = {**{n: f for n, (f, _) in basic.items()}, **{n: f for n, (f, _) in advanced_models(gpu=False).items()},
            **experiments}
    for name in CONFIRM_MODELS:
        conf.run(name, pool[name])
    conf.average("avg: LogReg + polynomial LogReg", [BASE, "Polynomial (degree-2) logistic regression"])
    conf.average("avg: LogReg + spline GAM (5 noisy copies)", [BASE, "Spline GAM on 5 test-noise copies"])
    save_table(conf.table(), "confirmation")

    log("noise robustness curves")
    save_table(robustness_curves(folds[:5]), "noise_curves")
    log("20% of the training data")
    exp20, n_full, n_sub = twenty_percent(folds[:5], gpu)
    save_table(exp20, "twenty_percent")
    log("clustering")
    save_table(clustering(), "clustering")

    screen, confirm = res.table([BASE]).loc[BASE], conf.table([BASE]).loc[BASE]
    summary = {
        "date": f"{datetime.date.today():%Y-%m-%d}", "quick": quick,
        "device": torch.cuda.get_device_name(0) if gpu else "CPU",
        "train_rows": int(len(D.X_dup)), "duplicates_removed": int(len(D.X_dup) - len(D.X)),
        "train_rows_used": int(len(D.X)), "test_rows": int(len(D.X_test)),
        "screen_folds": len(folds), "confirm_folds": len(conf.folds),
        "rows_per_fold_full": int(n_full), "rows_per_fold_20pct": int(n_sub),
        "final_clean": float(screen["clean"]), "final_test_like": float(screen["test-like"]),
        "final_stress": float(screen["stress 0.15"]), "final_test_like_confirm": float(confirm["test-like"]),
        "test_prediction_mix": pred_mix, "minutes": round((time.time() - t0) / 60, 1),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"done in {summary['minutes']} min - tables in {RESULTS}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="first CV repeat only, no TabPFN / FT-Transformer")
    ap.add_argument("--no-gpu", action="store_true", help="skip XGBoost/CatBoost GPU, MLPs, FT-Transformer, TabPFN")
    ap.add_argument("--no-tabpfn", action="store_true", help="skip TabPFN v2")
    ap.add_argument("--final-only", action="store_true", help="only train the final model and write the submission")
    ap.add_argument("--data", help="folder with the competition CSVs (default: data/ next to this file)")
    ap.add_argument("--out", help="folder for the result tables (default: results/ next to this file)")
    a = ap.parse_args()
    if a.data:                       # environment variables reach joblib's worker processes too
        os.environ["VEHICLE_DATA"] = str(Path(a.data).resolve())
    if a.out:
        os.environ["VEHICLE_RESULTS"] = str(Path(a.out).resolve())
    # run through the importable module rather than __main__, so joblib's worker processes
    # can unpickle the functions and classes defined in this file
    import vehicle_pipeline
    vehicle_pipeline.main(quick=a.quick, gpu=not a.no_gpu, tabpfn=not a.no_tabpfn, final_only=a.final_only)
