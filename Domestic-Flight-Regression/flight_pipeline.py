"""
Domestic Flight Price Prediction - shared pipeline used by the notebooks.

    01_EDA.ipynb            exploratory data analysis
    02_Preprocessing.ipynb  cleaning and feature engineering, step by step
    03_Models.ipynb         basic models, advanced models, tuning, ensemble

The notebooks explain every step; this module holds the same code once so
03_Models.ipynb can reuse it. Metric: RMSE on the raw price.

Validation: 5-fold KFold (shuffle, seed 42) on all 15,589 training rows, the
same folds for every model. Target statistics are computed with nested folds,
so no row ever sees its own price. Test predictions come from a refit on all
training rows.
"""
from __future__ import annotations

import json
import math
import time
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PRED_DIR = RESULTS / "predictions"
for _d in (RESULTS, PRED_DIR):
    _d.mkdir(exist_ok=True)

TARGET, ID_COL = "price", "id"
SEED = 42
N_FOLDS = 5
INNER_FOLDS = 10
PRICE_FLOOR = 3801.0                      # cheapest fare in train


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + str(msg), flush=True)


# ==========================================================================
# metrics
# ==========================================================================
def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def r2(y, p):
    y = np.asarray(y)
    return float(1 - np.sum((y - np.asarray(p)) ** 2) / np.sum((y - y.mean()) ** 2))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


# ==========================================================================
# 1. cleaning: recover the airline, impute the duration
# ==========================================================================
TIME_ORDER = {"Early_Morning": 0, "Morning": 1, "Afternoon": 2,
              "Evening": 3, "Night": 4, "Late_Night": 5}
STOPS_ORDER = {"zero": 0, "one": 1, "two_or_more": 2}
# IATA prefixes of the foreign carriers (Emirates, Etihad, Qatar, Singapore,
# Thai, SriLankan, Malaysia, Oman, Gulf Air, Kuwait, flydubai, Nepal Airlines)
FOREIGN = {"EK", "EY", "QR", "SQ", "TG", "UL", "MH", "WY", "GF", "KU", "FZ", "RA"}
ITIN = ["flight", "source_city", "destination_city", "stops",
        "departure_time", "arrival_time"]
# duration lookups, most specific first: the itinerary, then coarser keys
DUR_KEYS = [
    ITIN,
    ["flight", "source_city", "destination_city", "stops", "arrival_time"],
    ["flight", "source_city", "destination_city", "stops"],
    ["flight", "stops", "departure_time", "arrival_time"],
    ["flight", "stops"],
    ["flight"],
    ["source_city", "destination_city", "stops", "departure_time", "arrival_time"],
    ["source_city", "destination_city", "stops"],
    ["stops"],
]


def add_prefix(df):
    """'AI 2641' -> flight_prefix 'AI' (the airline's IATA code), flight_number 2641."""
    df = df.copy()
    parts = df["flight"].str.extract(r"^([A-Za-z0-9]+)[\s-]*(\d*)$")
    df["flight_prefix"] = parts[0]
    df["flight_number"] = pd.to_numeric(parts[1], errors="coerce")
    return df


def build_prefix_map(frames):
    """prefix -> airline from every row where the airline is known."""
    known = pd.concat([f[["flight_prefix", "airline"]] for f in frames],
                      ignore_index=True).dropna()
    return known.groupby("flight_prefix")["airline"].agg(lambda s: s.mode()[0])


def recover_airline(df, prefix_map):
    df = df.copy()
    df["airline"] = df["airline"].fillna(df["flight_prefix"].map(prefix_map))
    df["airline"] = df["airline"].fillna("Unknown")
    return df


def impute_duration(df, known):
    """Fill duration through DUR_KEYS and record which level was used."""
    df = df.copy()
    df["dur_imputed"] = df["duration"].isna().astype(int)
    df["dur_level"] = np.where(df["duration"].isna(), -1, 0)
    for lvl, keys in enumerate(DUR_KEYS, 1):
        need = df["duration"].isna()
        if not need.any():
            break
        med = known.groupby(keys)["duration"].median().rename("_m").reset_index()
        fill = df.loc[need, keys].merge(med, on=keys, how="left")["_m"].to_numpy()
        df.loc[need, "duration"] = fill
        df.loc[need & df["duration"].notna(), "dur_level"] = lvl
    rest = df["duration"].isna()
    df.loc[rest, "duration"] = known["duration"].median()
    df.loc[rest, "dur_level"] = len(DUR_KEYS) + 1
    return df


@lru_cache(maxsize=1)
def load_raw():
    return pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")


@lru_cache(maxsize=1)
def load_clean():
    """(all rows, y, n_train, test_ids): train rows first, then test."""
    train, test = load_raw()
    y = train[TARGET].to_numpy(float)
    allx = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    allx = add_prefix(allx)
    allx = recover_airline(allx, build_prefix_map([allx]))
    known = allx.dropna(subset=["duration"])
    allx = impute_duration(allx, known)
    return allx, y, len(train), test[ID_COL].to_numpy()


# ==========================================================================
# 2. target-free features
# ==========================================================================
CAT_COLS = ["airline", "flight", "flight_prefix", "source_city",
            "destination_city", "route", "departure_time", "arrival_time",
            "stops", "flight_class", "itin", "itin_c", "flight_c",
            "airline_c", "route_c", "time_pair"]
BASE_NUM = ["duration", "dur_imputed", "dur_level", "days_left", "inv_days",
            "log_days", "dow", "flight_number", "stops_num", "dep_slot", "arr_slot",
            "overnight", "is_business", "is_foreign", "dur_per_stop",
            "cnt_flight", "cnt_itin", "cnt_itin_c", "cnt_flight_c",
            "cnt_route_c", "cnt_airline", "cnt_twin_all", "itin_dur_nuniq",
            "itin_dur_std"]


@lru_cache(maxsize=1)
def build_base():
    """Row-wise features and train+test frequency counts (no target used),
    plus integer codes for every grouping the target statistics need."""
    d, y, n_train, test_ids = load_clean()
    d = d.copy()
    d["route"] = d["source_city"] + ">" + d["destination_city"]
    d["itin"] = d[ITIN].astype(str).agg("|".join, axis=1)
    d["itin_c"] = d["itin"] + "|" + d["flight_class"]
    d["flight_c"] = d["flight"] + "|" + d["flight_class"]
    d["airline_c"] = d["airline"] + "|" + d["flight_class"]
    d["route_c"] = d["route"] + "|" + d["flight_class"]
    d["time_pair"] = d["departure_time"] + "|" + d["arrival_time"]
    d["dur_key"] = (d["duration"] * 100).round().astype(int)

    d["days_left"] = d["days_left"].astype(float)
    d["inv_days"] = 1.0 / (d["days_left"] + 1.0)
    d["log_days"] = np.log1p(d["days_left"])
    d["dow"] = (d["days_left"] % 7).astype(float)        # travel weekday index
    d["stops_num"] = d["stops"].map(STOPS_ORDER).astype(float)
    d["dep_slot"] = d["departure_time"].map(TIME_ORDER).astype(float)
    d["arr_slot"] = d["arrival_time"].map(TIME_ORDER).astype(float)
    d["overnight"] = (d["arr_slot"] < d["dep_slot"]).astype(int)
    d["is_business"] = (d["flight_class"] == "Business").astype(int)
    d["is_foreign"] = d["flight_prefix"].isin(FOREIGN).astype(int)
    d["dur_per_stop"] = d["duration"] / (d["stops_num"] + 1.0)
    d["flight_number"] = d["flight_number"].astype(float)

    def cnt(cols):
        return d.groupby(cols)[cols[0]].transform("size").astype(float)
    d["cnt_flight"] = cnt(["flight"])
    d["cnt_itin"] = cnt(["itin"])
    d["cnt_itin_c"] = cnt(["itin_c"])
    d["cnt_flight_c"] = cnt(["flight_c"])
    d["cnt_route_c"] = cnt(["route_c"])
    d["cnt_airline"] = cnt(["airline"])
    d["cnt_twin_all"] = cnt(["itin_c", "days_left"])
    obs = d["dur_imputed"] == 0
    g = d[obs].groupby("itin")["duration"]
    d["itin_dur_nuniq"] = d["itin"].map(g.nunique()).fillna(0).astype(float)
    d["itin_dur_std"] = d["itin"].map(g.std()).fillna(0).astype(float)

    codes = {}
    for c in CAT_COLS + ["dur_key"]:
        codes[c] = pd.factorize(d[c])[0].astype(np.int64)

    def comb(cols):
        mi = pd.MultiIndex.from_arrays([codes[c] for c in cols])
        return pd.factorize(mi)[0].astype(np.int64)
    codes["is_foreign"] = d["is_foreign"].to_numpy().astype(np.int64)
    codes["days"] = d["days_left"].to_numpy().astype(np.int64)
    # day-level groups (travel-date effects)
    codes["d_seg"] = comb(["flight_class", "is_foreign", "days"])
    codes["d_air"] = comb(["airline_c", "days"])
    codes["d_route"] = comb(["route_c", "days"])
    codes["d_src"] = comb(["source_city", "flight_class", "days"])
    codes["d_dst"] = comb(["destination_city", "flight_class", "days"])
    codes["d_dep"] = comb(["departure_time", "flight_class", "is_foreign", "days"])
    codes["d_flt"] = comb(["flight_c", "days"])
    codes["d_fltx"] = comb(["flight", "days"])
    codes["d_itx"] = comb(["itin", "days"])
    # weekday groups (days_left mod 7)
    codes["dow"] = (codes["days"] % 7).astype(np.int64)
    codes["w_seg"] = comb(["flight_class", "is_foreign", "dow"])
    codes["w_air"] = comb(["airline_c", "dow"])
    codes["w_route"] = comb(["route_c", "dow"])
    codes["w_dst"] = comb(["destination_city", "flight_class", "dow"])
    codes["w_src"] = comb(["source_city", "flight_class", "dow"])
    # groups for the price statistics
    codes["g_c"] = codes["flight_class"]
    codes["g_fd"] = comb(["flight_class"]) * 2 + d["is_foreign"].to_numpy()
    codes["g_ac"] = codes["airline_c"]
    codes["g_asc"] = comb(["airline", "stops", "flight_class"])
    codes["g_rc"] = codes["route_c"]
    codes["g_arc"] = comb(["airline", "route", "flight_class"])
    codes["g_fc"] = codes["flight_c"]
    codes["g_ic"] = codes["itin_c"]
    codes["g_icd"] = comb(["itin_c", "dur_key"])
    return d, codes, y, n_train, test_ids


# ==========================================================================
# 3. target statistics (fitted on some rows, applied to others)
# ==========================================================================
CHAIN = ["g_c", "g_ac", "g_fc", "g_ic", "g_icd"]          # shrinkage hierarchy
STAT_GROUPS = ["g_c", "g_fd", "g_ac", "g_asc", "g_rc", "g_arc", "g_fc", "g_ic", "g_icd"]
RICH_GROUPS = {"g_arc", "g_fc", "g_ic", "g_icd"}          # + median / min / max / log-mean
SHRINK_M = 3.0
TAUS = (2.0, 6.0)
N_DAYS = 64
DAY_GROUPS = ["d_seg", "d_air", "d_route", "d_src", "d_dst", "d_dep",
              "d_flt", "d_fltx", "d_itx",
              "w_seg", "w_air", "w_route", "w_dst", "w_src"]
DAY_M = 5.0


def _group_stats(g_fit, y_fit, g_apply, n_groups, rich):
    cnt = np.bincount(g_fit, minlength=n_groups).astype(float)
    s1 = np.bincount(g_fit, weights=y_fit, minlength=n_groups)
    s2 = np.bincount(g_fit, weights=y_fit ** 2, minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        std = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 0.0))
    out = {"n": cnt[g_apply], "mean": mean[g_apply], "std": std[g_apply]}
    if rich:
        s = pd.Series(y_fit).groupby(g_fit)
        agg = pd.DataFrame({"med": s.median(), "min": s.min(), "max": s.max(),
                            "lmean": np.log(pd.Series(y_fit)).groupby(g_fit).mean()})
        agg = agg.reindex(np.arange(n_groups))
        for k in agg.columns:
            out[k] = agg[k].to_numpy()[g_apply]
    return out, cnt, mean


def target_stats(codes, days, fit_idx, y_fit, apply_idx):
    """Statistics of the price over the rows `fit_idx`, mapped onto `apply_idx`."""
    feats, level = {}, {}
    # group means, spreads and counts at several levels
    for gname in STAT_GROUPS:
        g = codes[gname]
        G = int(g.max()) + 1
        st, cnt, mean = _group_stats(g[fit_idx], y_fit, g[apply_idx], G, gname in RICH_GROUPS)
        for k, v in st.items():
            feats[f"{gname[2:]}_{k}"] = v
        level[gname] = (cnt, mean)
    # hierarchical shrinkage: each level's mean pulled toward its parent's
    prior_rows = None
    for gname in CHAIN:
        g = codes[gname]
        cnt, mean = level[gname]
        if prior_rows is None:
            sm = np.where(cnt > 0, mean, np.mean(y_fit))
        else:
            parent = np.zeros(len(cnt))
            parent[g] = prior_rows
            sm = (cnt * np.nan_to_num(mean) + SHRINK_M * parent) / (cnt + SHRINK_M)
        prior_rows = sm[g]
        feats[f"sm_{gname[2:]}"] = sm[g[apply_idx]]
    # exact twins and booking-curve neighbours inside itinerary x class
    for gname, tag in (("g_ic", "ic"), ("g_icd", "icd")):
        g = codes[gname]
        G = int(g.max()) + 1
        d_fit = days[fit_idx].astype(int)
        S = np.zeros((G, N_DAYS))
        C = np.zeros((G, N_DAYS))
        np.add.at(S, (g[fit_idx], d_fit), y_fit)
        np.add.at(C, (g[fit_idx], d_fit), 1.0)
        ga, da = g[apply_idx], days[apply_idx].astype(int)
        with np.errstate(invalid="ignore", divide="ignore"):
            feats[f"tw_{tag}_n"] = C[ga, da]
            feats[f"tw_{tag}_mean"] = S[ga, da] / C[ga, da]
            if tag == "ic":
                dd = np.arange(N_DAYS)
                for tau in TAUS:
                    K = np.exp(-np.abs(dd[:, None] - dd[None, :]) / tau)
                    Sk, Ck = S @ K, C @ K
                    feats[f"nb{int(tau)}_mean"] = Sk[ga, da] / Ck[ga, da]
                    feats[f"nb{int(tau)}_w"] = Ck[ga, da]
                    K0 = K.copy()
                    np.fill_diagonal(K0, 0.0)              # neighbours only
                    Sk0, Ck0 = S @ K0, C @ K0
                    feats[f"nb{int(tau)}x_mean"] = Sk0[ga, da] / Ck0[ga, da]
    # travel-date effects: each price relative to its itinerary's median,
    # averaged per day-level / weekday-level group, shrunk toward zero
    gi = codes["g_ic"][fit_idx]
    ys = pd.Series(y_fit)
    med = ys.groupby(gi).transform("median").to_numpy()
    size = ys.groupby(gi).transform("size").to_numpy()
    ok = size >= 3
    lrel = np.log(y_fit / med)
    base = np.where(np.isnan(feats["icd_med"]), feats["sm_icd"], feats["icd_med"])
    for gname in DAY_GROUPS:
        gd = codes[gname]
        G = int(gd.max()) + 1
        cnt = np.bincount(gd[fit_idx][ok], minlength=G).astype(float)
        s = np.bincount(gd[fit_idx][ok], weights=lrel[ok], minlength=G)
        eff = s / (cnt + DAY_M)
        ga = gd[apply_idx]
        feats[f"{gname}_eff"] = eff[ga]
        feats[f"{gname}_n"] = cnt[ga]
        feats[f"{gname}_xp"] = base * np.exp(eff[ga])
    df = pd.DataFrame(feats)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["nb2_rel"] = df["nb2_mean"] / df["ic_mean"]
        df["nb6_rel"] = df["nb6_mean"] / df["ic_mean"]
    return df


def nested_stats(codes, days, fit_idx, y_fit, apply_sets, seed=SEED, inner=INNER_FOLDS):
    """Target statistics for the fitting rows (inner out-of-fold: computed
    from the OTHER inner folds) and for each array in `apply_sets` (computed
    from all of fit_idx)."""
    fit_idx = np.asarray(fit_idx)
    kf = KFold(inner, shuffle=True, random_state=seed)
    pieces = []
    for a, b in kf.split(fit_idx):
        st = target_stats(codes, days, fit_idx[a], y_fit[a], fit_idx[b])
        st.index = b
        pieces.append(st)
    tr_stats = pd.concat(pieces).sort_index().reset_index(drop=True)
    others = [target_stats(codes, days, fit_idx, y_fit, np.asarray(s)) for s in apply_sets]
    return tr_stats, others


# ==========================================================================
# 4. folds and feature matrices
# ==========================================================================
def outer_folds(seed=SEED, n_splits=N_FOLDS):
    n_train = build_base()[3]
    return list(KFold(n_splits, shuffle=True, random_state=seed).split(np.arange(n_train)))


def frame(idx, stats=None):
    d = build_base()[0]
    X = d.iloc[idx][CAT_COLS + BASE_NUM].reset_index(drop=True)
    if stats is not None:
        X = pd.concat([X, stats.reset_index(drop=True)], axis=1)
    return X


@lru_cache(maxsize=32)
def fold_data(k, seed=SEED):
    """Outer fold k: (Xtr, ytr, Xva, yva, Xte, va_idx), target statistics nested."""
    d, codes, y, n_train, _ = build_base()
    tr_idx, va_idx = outer_folds(seed)[k]
    te_idx = np.arange(n_train, len(d))
    days = d["days_left"].to_numpy()
    st_tr, (st_va, st_te) = nested_stats(codes, days, tr_idx, y[tr_idx],
                                         [va_idx, te_idx], seed=seed + k)
    return (frame(tr_idx, st_tr), y[tr_idx], frame(va_idx, st_va), y[va_idx],
            frame(te_idx, st_te), va_idx)


@lru_cache(maxsize=2)
def full_data(seed=SEED):
    """All training rows (inner-OOF statistics) and the test rows (statistics from all of train)."""
    d, codes, y, n_train, _ = build_base()
    tr_idx = np.arange(n_train)
    te_idx = np.arange(n_train, len(d))
    days = d["days_left"].to_numpy()
    st_tr, (st_te,) = nested_stats(codes, days, tr_idx, y, [te_idx], seed=seed)
    return frame(tr_idx, st_tr), y, frame(te_idx, st_te)


def feature_lists(X):
    cats = [c for c in CAT_COLS if c in X.columns]
    return cats, [c for c in X.columns if c not in cats]


# ==========================================================================
# 5. scoring and saved predictions
# ==========================================================================
@lru_cache(maxsize=1)
def segments():
    d, _, y, n_train, _ = build_base()
    t = d.iloc[:n_train]
    biz = t["is_business"].to_numpy() == 1
    foreign = t["is_foreign"].to_numpy() == 1
    return {"business": biz, "economy": ~biz, "biz_foreign": biz & foreign,
            "biz_domestic": biz & ~foreign}


def score(oof, y=None):
    y = build_base()[2] if y is None else y
    s = {"rmse": rmse(y, oof), "r2": r2(y, oof), "mae": mae(y, oof)}
    for name, m in segments().items():
        s[f"rmse_{name}"] = rmse(y[m], oof[m])
    return s


def fmt(s):
    return (f"RMSE {s['rmse']:9.1f}  R2 {s['r2']:.4f}  MAE {s['mae']:8.1f}  "
            f"| Economy {s['rmse_economy']:7.1f}  Dom.Biz {s['rmse_biz_domestic']:7.1f}  "
            f"For.Biz {s['rmse_biz_foreign']:8.1f}")


def save_preds(name, oof, test, info=None, extra=None):
    oof, test = np.asarray(oof, float), np.asarray(test, float)
    s = score(oof)
    np.savez(PRED_DIR / f"{name}.npz", oof=oof, test=test, **(extra or {}))
    (PRED_DIR / f"{name}.json").write_text(json.dumps({"name": name, **s, **(info or {})},
                                                      indent=1, default=str))
    return s


def load_preds(names=None):
    out = {}
    for p in sorted(PRED_DIR.glob("*.npz")):
        if names is not None and p.stem not in names:
            continue
        z = np.load(p)
        out[p.stem] = {"oof": z["oof"], "test": z["test"],
                       "meta": json.loads((PRED_DIR / f"{p.stem}.json").read_text())}
    return out


def is_saved(name):
    return (PRED_DIR / f"{name}.npz").exists()


def compare_with_saved(name, oof):
    """Max absolute difference between a fresh run and the saved predictions."""
    if not is_saved(name):
        return None
    return float(np.abs(np.load(PRED_DIR / f"{name}.npz")["oof"] - oof).max())


def write_predictions(pred, path):
    test_ids = build_base()[4]
    pred = np.clip(np.asarray(pred, float), PRICE_FLOOR, None)
    pd.DataFrame({ID_COL: test_ids, TARGET: pred}).to_csv(path, index=False)
    return path


# ==========================================================================
# 6. basic models: the team notebooks' feature set + scikit-learn pipelines
# ==========================================================================
RAW_CATS = ["airline", "flight", "source_city", "departure_time", "stops",
            "arrival_time", "destination_city", "flight_class"]
RAW_NUMS = ["duration", "days_left"]
ENG_NUMS = ["duration", "days_left", "days_left_squared", "days_left_cubed",
            "log_days_left", "duration_squared", "log_duration", "duration_days"]
ENG_CATS = ["airline", "source_city", "departure_time", "stops", "arrival_time",
            "destination_city", "flight_class", "route", "class_stops",
            "airline_class", "airline_route", "departure_arrival", "days_left_bucket"]


def team_features(df):
    """engineer_features() of the team's tuned-model notebooks."""
    df = df.copy()
    df["route"] = df["source_city"] + "_" + df["destination_city"]
    df["class_stops"] = df["flight_class"] + "_" + df["stops"]
    df["airline_class"] = df["airline"] + "_" + df["flight_class"]
    df["airline_route"] = df["airline"] + "_" + df["route"]
    df["departure_arrival"] = df["departure_time"] + "_" + df["arrival_time"]
    df["days_left_squared"] = df["days_left"] ** 2
    df["days_left_cubed"] = df["days_left"] ** 3
    df["log_days_left"] = np.log1p(df["days_left"])
    df["duration_squared"] = df["duration"] ** 2
    df["log_duration"] = np.log1p(df["duration"])
    df["duration_days"] = df["duration"] * df["days_left"]
    df["days_left_bucket"] = pd.cut(
        df["days_left"], bins=[-1, 3, 7, 15, 30, 45, 60, 1000],
        labels=["last_minute", "very_short", "short", "medium", "long",
                "very_long", "early"]).astype(str)
    return df


@lru_cache(maxsize=1)
def basic_data():
    """Cleaned training rows with the team's engineered columns, and y."""
    d, y, n_train, _ = load_clean()
    d = d.copy()
    d["days_left"] = d["days_left"].astype(float)
    return team_features(d.iloc[:n_train].reset_index(drop=True)), y


def onehot_pipe(model, cats, nums, poly=False, min_freq=None):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
    num = Pipeline([("poly", PolynomialFeatures(2, include_bias=False)),
                    ("sc", StandardScaler())]) if poly else StandardScaler()
    ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=min_freq)
    return Pipeline([("prep", ColumnTransformer([("num", num, nums), ("cat", ohe, cats)])),
                     ("model", model)])


def tree_pipe(model, cats, nums):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return Pipeline([("prep", ColumnTransformer([("num", "passthrough", nums), ("cat", enc, cats)])),
                     ("model", model)])


FEATURE_SETS = {"onehot": (RAW_CATS, RAW_NUMS, None), "engineered": (ENG_CATS, ENG_NUMS, 2)}


def cv_predict(make, X, y):
    """Out-of-fold predictions of a scikit-learn model on the shared folds."""
    oof = np.zeros(len(y))
    for tr, va in outer_folds():
        oof[va] = make().fit(X.iloc[tr], y[tr]).predict(X.iloc[va])
    return oof


# ==========================================================================
# 7. per-class Ridge on the L5 feature ladder (the linear project's model)
# ==========================================================================
LIN_CATS = ["airline", "source_city", "departure_time", "stops", "arrival_time",
            "destination_city", "flight_class"]
LIN_NUM_ENG = ["duration", "days_left", "log_duration", "sqrt_duration",
               "inv_days_left", "log_days_left", "stops_num", "dep_hour",
               "arr_hour", "duration_per_stop", "is_business",
               "is_last_minute", "is_very_early", "is_redeye", "duration_imputed"]


def _linear_clean():
    """The linear project's cleaning: airline from the prefix; duration from
    flight+stops -> flight -> route+stops -> stops -> global medians."""
    train, test = load_raw()
    train, test = add_prefix(train), add_prefix(test)
    pmap = build_prefix_map([train, test])
    train, test = recover_airline(train, pmap), recover_airline(test, pmap)
    known = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True).dropna(subset=["duration"])
    maps = {"flight_stops": known.groupby(["flight", "stops"])["duration"].median(),
            "flight": known.groupby("flight")["duration"].median(),
            "route_stops": known.groupby(["source_city", "destination_city", "stops"])["duration"].median(),
            "stops": known.groupby("stops")["duration"].median(),
            "global": known["duration"].median()}

    def imp(df):
        df = df.copy()
        df["duration_imputed"] = df["duration"].isna().astype(int)
        need = df["duration"].isna()
        if need.any():
            keys = list(zip(df.loc[need, "flight"], df.loc[need, "stops"]))
            df.loc[need, "duration"] = pd.Series(maps["flight_stops"].reindex(keys).to_numpy(),
                                                 index=df.index[need])
        need = df["duration"].isna()
        if need.any():
            df.loc[need, "duration"] = df.loc[need, "flight"].map(maps["flight"])
        need = df["duration"].isna()
        if need.any():
            keys = list(zip(df.loc[need, "source_city"], df.loc[need, "destination_city"],
                            df.loc[need, "stops"]))
            df.loc[need, "duration"] = pd.Series(maps["route_stops"].reindex(keys).to_numpy(),
                                                 index=df.index[need])
        need = df["duration"].isna()
        if need.any():
            df.loc[need, "duration"] = df.loc[need, "stops"].map(maps["stops"])
        df["duration"] = df["duration"].fillna(maps["global"])
        return df
    return imp(train), imp(test)


def _linear_engineer(df):
    d = df.copy()
    d["stops_num"] = d["stops"].map(STOPS_ORDER).fillna(0).astype(int)
    d["dep_hour"] = d["departure_time"].map(TIME_ORDER).fillna(2).astype(int)
    d["arr_hour"] = d["arrival_time"].map(TIME_ORDER).fillna(2).astype(int)
    d["is_business"] = (d["flight_class"] == "Business").astype(int)
    d["log_duration"] = np.log1p(d["duration"])
    d["sqrt_duration"] = np.sqrt(d["duration"])
    d["inv_days_left"] = 1.0 / (d["days_left"] + 1.0)
    d["log_days_left"] = np.log1p(d["days_left"])
    d["is_last_minute"] = (d["days_left"] <= 3).astype(int)
    d["is_very_early"] = (d["days_left"] >= 40).astype(int)
    d["route"] = d["source_city"] + "->" + d["destination_city"]
    d["is_redeye"] = ((d["dep_hour"] >= 4) & (d["arr_hour"] <= 1)).astype(int)
    d["time_pair"] = d["departure_time"] + "|" + d["arrival_time"]
    d["duration_per_stop"] = d["duration"] / (d["stops_num"] + 1)
    return d


def _class_interaction():
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.preprocessing import StandardScaler

    class ClassInteraction(BaseEstimator, TransformerMixin):
        """Every numeric column times the Business indicator (separate slopes per cabin)."""
        def fit(self, X, y=None):
            self.scaler_ = StandardScaler().fit(self._interact(X))
            return self

        def transform(self, X):
            return self.scaler_.transform(self._interact(X))

        @staticmethod
        def _interact(X):
            X = pd.DataFrame(X)
            return X.to_numpy(dtype=float) * X["is_business"].to_numpy(dtype=float).reshape(-1, 1)
    return ClassInteraction


def l5_pipeline(estimator):
    """One-hot + engineered numerics + class interactions + splines + target-encoded flight."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (OneHotEncoder, SplineTransformer, StandardScaler,
                                       TargetEncoder)
    ct = ColumnTransformer([
        ("num", StandardScaler(), LIN_NUM_ENG),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5),
         LIN_CATS + ["route", "time_pair"]),
        ("class_x", _class_interaction()(), LIN_NUM_ENG),
        ("spline", SplineTransformer(n_knots=12, degree=3, include_bias=False),
         ["duration", "days_left"]),
        ("te", TargetEncoder(smooth="auto", random_state=SEED, target_type="continuous"),
         ["flight", "flight_prefix", "route"]),
    ])
    return Pipeline([("features", ct), ("model", estimator)])


class PerClassRidge:
    """One Ridge for Economy and one for Business (plus a global fallback)."""

    def __init__(self, alpha=0.1, min_rows=50):
        self.alpha, self.min_rows = alpha, min_rows

    def fit(self, X, y):
        from sklearn.linear_model import Ridge
        y = np.asarray(y, dtype=float)
        self.global_ = l5_pipeline(Ridge(alpha=self.alpha)).fit(X, y)
        self.models_ = {}
        for c in (0, 1):
            m = X["is_business"].to_numpy() == c
            if m.sum() >= self.min_rows:
                self.models_[c] = l5_pipeline(Ridge(alpha=self.alpha)).fit(X[m], y[m])
        return self

    def predict(self, X):
        out = self.global_.predict(X)
        for c, mdl in self.models_.items():
            m = X["is_business"].to_numpy() == c
            if m.any():
                out[m] = mdl.predict(X[m])
        return out


def run_perclass_ridge(name="linear_ridge_perclass", save=True):
    train, test = _linear_clean()
    train, test = _linear_engineer(train), _linear_engineer(test)
    y = train[TARGET].to_numpy(float)
    X = train.drop(columns=[TARGET])
    oof = np.zeros(len(y))
    for tr, va in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(y):
        oof[va] = PerClassRidge().fit(X.iloc[tr], y[tr]).predict(X.iloc[va])
    pred = PerClassRidge().fit(X, y).predict(test)
    if save:
        save_preds(name, oof, pred, {"kind": "ridge"})
    return oof, pred


# ==========================================================================
# 8. gradient boosting: LightGBM, XGBoost, CatBoost
# ==========================================================================
LOW_CATS = ["airline", "flight_prefix", "source_city", "destination_city",
            "route", "departure_time", "arrival_time", "stops",
            "flight_class", "time_pair", "airline_c", "route_c"]
HIGH_CATS = ["flight", "itin", "itin_c", "flight_c"]
REFIT_SCALE = 1.15          # the refit sees 25% more rows -> a few more rounds
_CATEGORIES = {}


def categories(col):
    if col not in _CATEGORIES:
        _CATEGORIES[col] = pd.Index(sorted(build_base()[0][col].astype(str).unique()))
    return _CATEGORIES[col]


def prepare(X, cats, drop=()):
    """Numeric columns + the chosen categoricals as pandas Categorical."""
    nums = [c for c in feature_lists(X)[1] if c not in drop]
    out = X[nums].astype(float).copy()
    for c in cats:
        out[c] = pd.Categorical(X[c].astype(str), categories=categories(c))
    return out


def fit_lgb(params, Xtr, ytr, Xva, yva, n_rounds, seed):
    import lightgbm as lgb
    p = dict(objective="regression", metric="l2", learning_rate=0.03, num_leaves=31,
             min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
             reg_lambda=1.0, max_cat_to_onehot=8, cat_smooth=10, min_data_per_group=20,
             n_jobs=8, verbose=-1, random_state=seed)
    p.update(params)
    ev = {}
    m = lgb.train(p, lgb.Dataset(Xtr, ytr), num_boost_round=n_rounds,
                  valid_sets=[lgb.Dataset(Xva, yva)], valid_names=["va"],
                  callbacks=[lgb.record_evaluation(ev)])
    return np.asarray(ev["va"]["l2"]), (lambda X, n: m.predict(X, num_iteration=n))


def fit_xgb(params, Xtr, ytr, Xva, yva, n_rounds, seed):
    import xgboost as xgb
    p = dict(objective="reg:squarederror", eta=0.03, max_depth=6, min_child_weight=5,
             subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist",
             device="cpu", nthread=8, max_cat_to_onehot=8, max_bin=256, seed=seed,
             eval_metric="rmse")
    p.update(params)
    dtr = xgb.DMatrix(Xtr, ytr, enable_categorical=True)
    dva = xgb.DMatrix(Xva, yva, enable_categorical=True)
    ev = {}
    m = xgb.train(p, dtr, num_boost_round=n_rounds, evals=[(dva, "va")],
                  evals_result=ev, verbose_eval=False)

    def pred(X, n):
        return m.predict(xgb.DMatrix(X, enable_categorical=True), iteration_range=(0, n))
    return np.asarray(ev["va"]["rmse"]) ** 2, pred


def fit_cat(params, Xtr, ytr, Xva, yva, n_rounds, seed):
    from catboost import CatBoostRegressor, Pool
    cats = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]

    def pool(X, y=None):
        X = X.copy()
        for c in cats:
            X[c] = X[c].astype(str)
        return Pool(X, y, cat_features=cats)
    p = dict(loss_function="RMSE", learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
             iterations=n_rounds, random_seed=seed, task_type="CPU", thread_count=8,
             border_count=254, verbose=False, allow_writing_files=False)
    p.update(params)
    m = CatBoostRegressor(**p)
    m.fit(pool(Xtr, ytr), eval_set=pool(Xva, yva), use_best_model=False)
    curve = np.asarray(m.get_evals_result()["validation"]["RMSE"]) ** 2
    return curve, (lambda X, n: m.predict(pool(X), ntree_end=n))


FITTERS = {"lgb": fit_lgb, "xgb": fit_xgb, "cat": fit_cat}


def run_gbdt(name, kind, params, cats=LOW_CATS, n_rounds=1000, seeds=(0,), drop=(),
             refit=True, save=True, verbose=True):
    """5-fold CV with ONE shared number of rounds (the argmin of the
    fold-pooled validation curve - no per-fold early stopping), then a refit
    on all training rows for the test predictions."""
    fit = FITTERS[kind]
    y_all = build_base()[2]
    n_test = len(build_base()[4])
    t0 = time.perf_counter()
    curves, store = [], []
    for k in range(N_FOLDS):
        Xtr, ytr, Xva, yva, Xte, va_idx = fold_data(k)
        A, B, T = prepare(Xtr, cats, drop), prepare(Xva, cats, drop), prepare(Xte, cats, drop)
        for s in seeds:
            curve, predict = fit(params, A, ytr, B, yva, n_rounds, seed=s)
            curves.append(curve * len(yva))
            store.append((va_idx, predict, B, T))
    L = min(len(c) for c in curves)
    best_n = int(np.argmin(np.sum([c[:L] for c in curves], axis=0))) + 1
    oof, cnt, test = np.zeros(len(y_all)), np.zeros(len(y_all)), np.zeros(n_test)
    for va_idx, predict, B, T in store:
        oof[va_idx] += predict(B, best_n)
        cnt[va_idx] += 1
        test += predict(T, best_n)
    oof /= np.maximum(cnt, 1)
    test /= len(store)
    test_fold_avg = test
    if refit:
        Xall, yall, Xtest = full_data()
        A, T = prepare(Xall, cats, drop), prepare(Xtest, cats, drop)
        n_full = int(round(best_n * REFIT_SCALE))
        test = np.mean([fit(params, A, yall, A.iloc[:200], yall[:200], n_full, seed=s)[1](T, n_full)
                        for s in seeds], axis=0)
    s = score(oof)
    if verbose:
        log(f"{name:28s} rounds={best_n:5d}  {fmt(s)}  ({time.perf_counter() - t0:.0f}s)")
    if save:
        save_preds(name, oof, test, {"kind": kind, "params": params, "target": "raw",
                                     "n_iter": best_n, "cats": list(cats), "seeds": list(seeds),
                                     "refit": refit}, extra={"test_fold_avg": test_fold_avg})
    return {"oof": oof, "test": test, "score": s, "n_iter": best_n}


# ==========================================================================
# 9. bagged trees: Random Forest, Extra Trees
# ==========================================================================
ORD_CATS = ["airline", "flight_prefix", "source_city", "destination_city",
            "route", "departure_time", "arrival_time", "stops", "flight_class", "time_pair"]


def _ord_matrix(X, codebook):
    M = X[feature_lists(X)[1]].astype(float).copy()
    for c in ORD_CATS:
        M[c] = X[c].map(codebook[c]).fillna(-1).astype(float)
    return M.fillna(-1.0).to_numpy(np.float32)        # missing statistics -> -1


def run_bagged(name, kind, params, refit=True, save=True):
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    d = build_base()[0]
    codebook = {c: {v: i for i, v in enumerate(sorted(d[c].unique()))} for c in ORD_CATS}
    Est = {"rf": RandomForestRegressor, "et": ExtraTreesRegressor}[kind]
    y_all = build_base()[2]
    oof, test_avg = np.zeros(len(y_all)), np.zeros(len(build_base()[4]))
    t0 = time.perf_counter()
    for k in range(N_FOLDS):
        Xtr, ytr, Xva, yva, Xte, va_idx = fold_data(k)
        m = Est(**params, n_jobs=8, random_state=k).fit(_ord_matrix(Xtr, codebook), ytr)
        oof[va_idx] = m.predict(_ord_matrix(Xva, codebook))
        test_avg += m.predict(_ord_matrix(Xte, codebook)) / N_FOLDS
    test = test_avg
    if refit:
        Xall, yall, Xtest = full_data()
        m = Est(**params, n_jobs=8, random_state=99).fit(_ord_matrix(Xall, codebook), yall)
        test = m.predict(_ord_matrix(Xtest, codebook))
    s = score(oof)
    log(f"{name:28s} {fmt(s)}  ({time.perf_counter() - t0:.0f}s)")
    if save:
        save_preds(name, oof, test, {"kind": kind, "params": params, "refit": refit},
                   extra={"test_fold_avg": test_avg})
    return {"oof": oof, "test": test, "score": s}


# ==========================================================================
# 10. neural networks: MLP, TabM, FT-Transformer (PyTorch, GPU if present)
# ==========================================================================
try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:                                   # pragma: no cover
    HAVE_TORCH = False

PRICE_SUFFIX = ("_mean", "_med", "_min", "_max", "_std", "_xp")


def numeric_block(Xtr, Xva, Xte, drop=()):
    """log-transform prices and counts, standardise on the training fold,
    add missing-value indicators, drop columns constant on the training fold."""
    nums = [c for c in feature_lists(Xtr)[1] if c not in drop]

    def tf(X):
        out = {}
        for c in nums:
            v = X[c].to_numpy(float)
            if c.endswith(PRICE_SUFFIX) or c.startswith("sm_"):
                v = np.log(np.clip(v, 1.0, None))
            elif c.endswith("_n") or c.endswith("_w") or c.startswith("cnt_"):
                v = np.log1p(np.clip(v, 0.0, None))
            out[c] = v
        return pd.DataFrame(out)
    A, B, C = tf(Xtr), tf(Xva), tf(Xte)
    nan_cols = [c for c in nums if A[c].isna().any() or B[c].isna().any() or C[c].isna().any()]
    med = A.median()
    Af = A.fillna(med)
    mu, sd = Af.mean(), Af.std().replace(0, 1.0)

    def fin(D):
        Z = ((D.fillna(med) - mu) / sd).clip(-6, 6).to_numpy(np.float32)
        return np.hstack([Z, D[nan_cols].isna().to_numpy(np.float32)])
    A, B, C = fin(A), fin(B), fin(C)
    keep = np.array([len(np.unique(A[:, j])) > 1 for j in range(A.shape[1])])
    return A[:, keep], B[:, keep], C[:, keep]


def cat_block(Xtr, Xva, Xte, cols, min_count=2):
    """Integer codes; levels rare or unseen in the training fold share index 0."""
    cards, A, B, C = [], [], [], []
    for c in cols:
        vc = Xtr[c].value_counts()
        keep = vc[vc >= min_count].index
        mapping = pd.Series(np.arange(1, len(keep) + 1), index=keep)
        for D, out in ((Xtr, A), (Xva, B), (Xte, C)):
            out.append(D[c].map(mapping).fillna(0).astype(np.int64).to_numpy())
        cards.append(len(keep) + 1)
    return np.stack(A, 1), np.stack(B, 1), np.stack(C, 1), cards


if HAVE_TORCH:
    def emb_dim(card):
        return int(min(32, max(4, round(1.6 * card ** 0.56))))

    class MLP(nn.Module):
        def __init__(self, n_num, cards, d=256, n_layers=3, dropout=0.15, emb_dropout=0.05):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(c, emb_dim(c)) for c in cards])
            d_in = n_num + sum(emb_dim(c) for c in cards)
            layers = []
            for i in range(n_layers):
                layers += [nn.Linear(d_in if i == 0 else d, d), nn.LayerNorm(d),
                           nn.SiLU(), nn.Dropout(dropout)]
            self.body = nn.Sequential(*layers)
            self.head = nn.Linear(d, 1)
            self.emb_drop = nn.Dropout(emb_dropout)

        def forward(self, xn, xc):
            e = [emb(xc[:, i]) for i, emb in enumerate(self.embs)]
            h = torch.cat([xn, self.emb_drop(torch.cat(e, 1))], 1)
            return self.head(self.body(h)).squeeze(-1)

    class TabMNet(nn.Module):
        """TabM (Gorishniy et al., ICLR 2025): k MLPs trained as one efficient ensemble."""
        def __init__(self, n_num, cards, xn_train, k=32, n_blocks=3, d_block=512,
                     dropout=0.1, d_emb=16, n_bins=32):
            super().__init__()
            from tabm import TabM
            num_emb = None
            if d_emb:
                from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
                bins = compute_bins(torch.as_tensor(xn_train), n_bins=n_bins)
                num_emb = PiecewiseLinearEmbeddings(bins, d_emb, activation=False, version="B")
            self.m = TabM.make(n_num_features=n_num, cat_cardinalities=cards, d_out=1, k=k,
                               n_blocks=n_blocks, d_block=d_block, dropout=dropout,
                               num_embeddings=num_emb)

        def forward(self, xn, xc):
            return self.m(xn, xc).squeeze(-1)                   # (batch, k)

    class FTT(nn.Module):
        """FT-Transformer: every feature becomes a token; a [CLS] token is read out."""
        def __init__(self, n_num, cards, d=96, n_layers=3, heads=8, attn_drop=0.2, ffn_drop=0.1):
            super().__init__()
            self.num_w = nn.Parameter(torch.randn(n_num, d) / math.sqrt(d))
            self.num_b = nn.Parameter(torch.zeros(n_num, d))
            self.cat_emb = nn.ModuleList([nn.Embedding(c, d) for c in cards])
            self.cls = nn.Parameter(torch.randn(1, 1, d) / math.sqrt(d))
            layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=ffn_drop,
                                               activation="gelu", batch_first=True, norm_first=True)
            layer.self_attn.dropout = attn_drop
            self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

        def forward(self, xn, xc):
            t_num = xn.unsqueeze(-1) * self.num_w + self.num_b
            t_cat = torch.stack([e(xc[:, i]) for i, e in enumerate(self.cat_emb)], 1)
            x = torch.cat([self.cls.expand(len(xn), -1, -1), t_num, t_cat], 1)
            return self.head(self.enc(x)[:, 0]).squeeze(-1)

    def fit_predict(make, xn, xc, y, xv, cv, xt, ct, epochs, lr, wd, bs, seed,
                    warmup=3, total_epochs=None):
        """AdamW, warmup + cosine schedule; per-epoch predictions for (val, test)."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = make().to(DEV)
        decay, no_decay = [], []
        for n_, p in model.named_parameters():
            (no_decay if p.ndim < 2 or "emb" in n_ or "norm" in n_.lower() else decay).append(p)
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                                 {"params": no_decay, "weight_decay": 0.0}], lr=lr)
        steps_per = math.ceil(len(y) / bs)
        total = (total_epochs or epochs) * steps_per

        def lr_at(step):
            if step < warmup * steps_per:
                return (step + 1) / (warmup * steps_per)
            t = (step - warmup * steps_per) / max(1, total - warmup * steps_per)
            return 0.5 * (1 + math.cos(math.pi * t))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=DEV)
        Xn, Xc, Y = T(xn), T(xc, torch.long), T(y)
        Vn, Vc, Tn, Tc = T(xv), T(cv, torch.long), T(xt), T(ct, torch.long)
        chunk = 256 if isinstance(model, FTT) else 2048        # attention is memory-hungry

        def predict(an, ac):
            out = []
            for i in range(0, len(an), chunk):
                o = model(an[i:i + chunk], ac[i:i + chunk])
                out.append(o.mean(1) if o.ndim == 2 else o)
            return torch.cat(out).float().cpu().numpy()
        vals, tests = [], []
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(len(Y), device=DEV)
            for i in range(0, len(Y), bs):
                idx = perm[i:i + bs]
                if len(idx) < 2:
                    continue
                out = model(Xn[idx], Xc[idx])
                tgt = Y[idx]
                loss = ((out - tgt[:, None]) ** 2).mean() if out.ndim == 2 else ((out - tgt) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
            model.eval()
            with torch.no_grad():
                vals.append(predict(Vn, Vc))
                tests.append(predict(Tn, Tc))
        del model, opt
        torch.cuda.empty_cache()
        return np.stack(vals), np.stack(tests)

    def _nn_prepare(arch, model_kw, Xtr, ytr, Xva, Xte, drop, cats):
        xn, xv, xt = numeric_block(Xtr, Xva, Xte, drop)
        xc, cv, ct, cards = cat_block(Xtr, Xva, Xte, cols=cats)
        mu, sd = ytr.mean(), ytr.std()
        fs = ((ytr.astype(float) - mu) / sd).astype(np.float32)

        def make():
            if arch == "mlp":
                return MLP(xn.shape[1], cards, **model_kw)
            if arch == "tabm":
                return TabMNet(xn.shape[1], cards, xn, **model_kw)
            return FTT(xn.shape[1], cards, **model_kw)
        return make, (xn, xc, fs), (xv, cv), (xt, ct), (lambda p: p * sd + mu)

    def run_nn(name, arch, seeds=(0,), epochs=30, lr=1e-3, wd=1e-4, bs=256, model_kw=None,
               cats=None, drop=(), refit=True, save=True):
        """5-fold CV with one shared epoch (argmin of the pooled validation curve),
        then a refit on all rows that stops at that epoch."""
        model_kw = model_kw or {}
        cats = cats or LOW_CATS
        y_all = build_base()[2]
        n_test = len(build_base()[4])
        t0 = time.perf_counter()
        curves = np.zeros(epochs)
        store = []
        for k in range(N_FOLDS):
            Xtr, ytr, Xva, yva, Xte, va_idx = fold_data(k)
            make, (xn, xc, fs), (xv, cv), (xt, ct), back = _nn_prepare(
                arch, model_kw, Xtr, ytr, Xva, Xte, drop, cats)
            for s in seeds:
                pv, pt = fit_predict(make, xn, xc, fs, xv, cv, xt, ct, epochs, lr, wd, bs,
                                     seed=1000 * s + k)
                pv, pt = back(pv), back(pt)
                curves += ((pv - yva[None, :]) ** 2).sum(1)
                store.append((va_idx, pv, pt))
        best = int(np.argmin(curves))
        oof, cnt, test = np.zeros(len(y_all)), np.zeros(len(y_all)), np.zeros(n_test)
        for va_idx, pv, pt in store:
            oof[va_idx] += pv[best]
            cnt[va_idx] += 1
            test += pt[best]
        oof /= np.maximum(cnt, 1)
        test /= len(store)
        test_fold_avg = test
        if refit:
            Xall, yall, Xtest = full_data()
            make, (xn, xc, fs), _, (xt, ct), back = _nn_prepare(
                arch, model_kw, Xall, yall, Xtest, Xtest, drop, cats)
            test = np.mean([back(fit_predict(make, xn, xc, fs, xt[:16], ct[:16], xt, ct, best + 1,
                                             lr, wd, bs, seed=7777 + s_, total_epochs=epochs)[1][best])
                            for s_ in seeds], axis=0)
        s = score(oof)
        log(f"{name:28s} epoch={best + 1:3d}  {fmt(s)}  ({time.perf_counter() - t0:.0f}s)")
        if save:
            save_preds(name, oof, test, {"kind": arch, "epochs": best + 1, "lr": lr, "wd": wd,
                                         "bs": bs, "model_kw": model_kw, "seeds": list(seeds),
                                         "target": "raw", "cats": cats, "refit": refit},
                       extra={"test_fold_avg": test_fold_avg})
        return {"oof": oof, "test": test, "score": s}


# ==========================================================================
# 11. TabPFN v2 (pretrained tabular transformer), one model per cabin
# ==========================================================================
TABPFN_CATS = ["airline", "source_city", "destination_city", "departure_time",
               "arrival_time", "stops"]
# 43 curated numeric features (all ~126 would not fit the 8 GB GPU)
TABPFN_NUMS = ["duration", "days_left", "dow", "stops_num", "dep_slot", "arr_slot",
               "overnight", "is_foreign", "dur_imputed", "flight_number",
               "cnt_itin_c", "cnt_flight_c", "cnt_twin_all",
               "sm_icd", "sm_ic", "sm_fc", "sm_ac", "ic_med", "ic_min", "ic_max",
               "ic_std", "ic_n", "icd_mean", "icd_n", "fc_mean", "fc_std", "arc_mean",
               "tw_ic_mean", "tw_ic_n", "nb2_mean", "nb6_mean", "nb2x_mean", "nb2_w",
               "d_seg_eff", "d_air_eff", "d_route_eff", "d_dst_eff", "d_flt_eff",
               "d_fltx_eff", "d_itx_eff", "w_seg_eff", "w_air_eff", "w_route_eff"]
# the same plus composite "itinerary level x day effect" prices
TABPFN_NUMS_XP = TABPFN_NUMS + ["d_seg_xp", "d_air_xp", "d_flt_xp", "d_itx_xp", "d_route_xp",
                                "w_seg_xp", "w_air_xp", "fc_med", "arc_med", "ac_mean"]


def _tabpfn(seed, n_estimators):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    return TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, device="cuda" if HAVE_TORCH and torch.cuda.is_available() else "cpu",
        n_estimators=n_estimators, memory_saving_mode=True, fit_mode="fit_preprocessors",
        random_state=seed, ignore_pretraining_limits=True)


def _parts(X, split):
    biz = X["is_business"].to_numpy() == 1
    if split == "class":
        return [~biz, biz]
    foreign = X["is_foreign"].to_numpy() == 1
    return [~biz, biz & ~foreign, biz & foreign]


def _tabpfn_matrix(X, nums, cat_maps):
    cats = np.stack([X[c].map(cat_maps[c]).fillna(-1).to_numpy(np.float32)
                     for c in TABPFN_CATS], 1)
    return np.hstack([X[nums].to_numpy(np.float32), cats])


def run_tabpfn(name, n_estimators=8, seed=0, nums=None, split="class", refit=True, save=True):
    """TabPFN v2 per cabin (or per segment); predicts the conditional MEAN."""
    nums = nums or TABPFN_NUMS
    d = build_base()[0]
    cat_maps = {c: {v: i for i, v in enumerate(sorted(d[c].unique()))} for c in TABPFN_CATS}
    cat_idx = list(range(len(nums), len(nums) + len(TABPFN_CATS)))
    y_all = build_base()[2]
    oof, test = np.zeros(len(y_all)), np.zeros(len(build_base()[4]))
    t0 = time.perf_counter()

    def predict(m, M):
        return np.concatenate([m.predict(M[i:i + 1000], output_type="mean")
                               for i in range(0, len(M), 1000)])
    for k in range(N_FOLDS):
        Xtr, ytr, Xva, yva, Xte, va_idx = fold_data(k)
        A, B, T = (_tabpfn_matrix(X, nums, cat_maps) for X in (Xtr, Xva, Xte))
        for mtr, mva, mte in zip(_parts(Xtr, split), _parts(Xva, split), _parts(Xte, split)):
            m = _tabpfn(seed, n_estimators)
            m.categorical_features_indices = cat_idx
            m.fit(A[mtr], ytr[mtr])
            oof[va_idx[mva]] = predict(m, B[mva])
            test[mte] += predict(m, T[mte]) / N_FOLDS
    test_fold_avg = test
    if refit:
        Xall, yall, Xtest = full_data()
        A, T = _tabpfn_matrix(Xall, nums, cat_maps), _tabpfn_matrix(Xtest, nums, cat_maps)
        test = np.zeros(len(Xtest))
        for mtr, mte in zip(_parts(Xall, split), _parts(Xtest, split)):
            m = _tabpfn(seed, n_estimators)
            m.categorical_features_indices = cat_idx
            m.fit(A[mtr], yall[mtr])
            test[mte] = predict(m, T[mte])
    s = score(oof)
    log(f"{name:28s} {fmt(s)}  ({time.perf_counter() - t0:.0f}s)")
    if save:
        save_preds(name, oof, test, {"kind": "tabpfn_v2", "split": split,
                                     "n_estimators": n_estimators, "target": "raw",
                                     "seed": seed, "refit": refit},
                   extra={"test_fold_avg": test_fold_avg})
    return {"oof": oof, "test": test, "score": s}


# ==========================================================================
# 12. blending and the twin pull
# ==========================================================================
def family(name):
    for f in ("lgb", "xgb", "cat", "mlp", "tabm", "ftt", "tabpfn", "ridge",
              "linear", "rf", "et"):
        if name.startswith(f):
            return f
    return "other"


def fit_nnls(P, y):
    from scipy.optimize import nnls
    return nnls(P, y)[0]


def fit_convex(P, y):
    """Weights >= 0 that sum to 1 (solved in units of the price SD)."""
    from scipy.optimize import minimize
    m = P.shape[1]
    s = y.std()
    G = (P / s).T @ (P / s) / len(y)
    b = (P / s).T @ (y / s) / len(y)
    res = minimize(lambda w: w @ G @ w - 2 * b @ w, np.full(m, 1.0 / m),
                   jac=lambda w: 2 * G @ w - 2 * b, method="SLSQP", bounds=[(0.0, 1.0)] * m,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x


def nested_blend(P, y, fitter):
    """Honest out-of-fold score of a blend: weights fitted on the other folds."""
    out = np.zeros(len(y))
    for tr_idx, va_idx in outer_folds():
        out[va_idx] = P[va_idx] @ fitter(P[tr_idx], y[tr_idx])
    return out


def twin_arrays():
    """Mean price of exact twins (same itinerary, cabin and days_left) in the
    training data: per validation row from its training fold, per test row
    from all of train. NaN where no twin exists."""
    y = build_base()[2]
    oof_tw = np.full(len(y), np.nan)
    for k in range(N_FOLDS):
        fd = fold_data(k)
        oof_tw[fd[5]] = fd[2]["tw_ic_mean"].to_numpy()
    return oof_tw, full_data()[2]["tw_ic_mean"].to_numpy()


TWIN_W = (0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0)


def twin_pull(oof, test, oof_tw, test_tw):
    """Move predictions toward the twin price; the weight is chosen by nested CV."""
    y = build_base()[2]
    has, has_t = ~np.isnan(oof_tw), ~np.isnan(test_tw)

    def apply(p, tw, m, w):
        q = p.copy()
        q[m] = (1 - w) * p[m] + w * tw[m]
        return q
    out = oof.copy()
    for tr_idx, va_idx in outer_folds():
        errs = [np.mean((apply(oof[tr_idx], oof_tw[tr_idx], has[tr_idx], w) - y[tr_idx]) ** 2)
                for w in TWIN_W]
        out[va_idx] = apply(oof[va_idx], oof_tw[va_idx], has[va_idx], TWIN_W[int(np.argmin(errs))])
    errs = [np.mean((apply(oof, oof_tw, has, w) - y) ** 2) for w in TWIN_W]
    w_all = TWIN_W[int(np.argmin(errs))]
    return out, apply(test, test_tw, has_t, w_all), w_all


def family_strong(names, preds):
    """Average each family's runs within 1% of its best, then equally weight
    the families whose best model is within 3% of the best single model."""
    best = preds[names[0]]["meta"]["rmse"]
    fam = {}
    for n in names:
        fam.setdefault(family(n), []).append(n)
    oofs, tests, used = [], [], []
    for f, members in fam.items():
        if f == "linear":
            continue
        fb = preds[members[0]]["meta"]["rmse"]
        keep = [n for n in members if preds[n]["meta"]["rmse"] <= 1.01 * fb]
        if fb <= 1.03 * best:
            oofs.append(np.mean([preds[n]["oof"] for n in keep], axis=0))
            tests.append(np.mean([preds[n]["test"] for n in keep], axis=0))
            used.append(f"{f}({len(keep)})")
    return np.mean(oofs, axis=0), np.mean(tests, axis=0), used


# ==========================================================================
# 13. experiment: size of the training set
# ==========================================================================
def learning_curve(fracs=(0.6, 0.8, 1.0)):
    """LightGBM on a share of each training fold, scored on the same validation rows."""
    import lightgbm as lgb
    d, codes, y, n_train, _ = build_base()
    days = d["days_left"].to_numpy()
    p = dict(objective="regression", metric="l2", learning_rate=0.05, num_leaves=63,
             min_child_samples=5, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
             reg_lambda=1.0, max_cat_to_onehot=8, cat_smooth=10, min_data_per_group=20,
             n_jobs=8, verbose=-1, random_state=0)
    rows = []
    for frac in fracs:
        oof = np.zeros(n_train)
        for k, (tr, va) in enumerate(outer_folds()):
            rng = np.random.default_rng(k)
            sub = np.sort(rng.choice(tr, int(len(tr) * frac), replace=False)) if frac < 1 else tr
            st_tr, (st_va,) = nested_stats(codes, days, sub, y[sub], [va], seed=SEED + k)
            A = prepare(frame(sub, st_tr), LOW_CATS)
            B = prepare(frame(va, st_va), LOW_CATS)
            m = lgb.train(p, lgb.Dataset(A, y[sub]), num_boost_round=int(832 * frac ** 0.5))
            oof[va] = m.predict(B)
        rows.append({"share_of_rows": 0.8 * frac, "cv_rmse": rmse(y, oof)})
    return pd.DataFrame(rows)
