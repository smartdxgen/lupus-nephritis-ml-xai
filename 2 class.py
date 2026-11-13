"""
Lupus nephritis (biopsy class) - full pipeline in one file.

Steps
  0-4  read, clean, engineer features, handle missing values, filter columns (with statistics tables)
  5    feature table: whole data + each class (count, %, mean, variance) + ANOVA / Welch t-tests
  6    leakage removal (+ single-feature AUC screen)
  7    ML-ready matrix export
  8    evaluation, binary (III/V vs IV) first, then 3-class:
       repeated stratified CV; imputation, scaling, feature selection and SMOTE are fitted INSIDE each
       training fold; strategies none / class_weight / smote; several simple models;
       bootstrap CI, label-permutation test, verdict, train-vs-test per fold, confusion matrix,
       ROC, calibration, feature stability / importance of the best configuration

Reads  : <BASE_DIR>\\data.xlsx
Writes : preprocess_log.txt, cleaned_data*.xlsx, preprocessed_ml_ready.xlsx, group_comparison.xlsx,
         results_<mode>.xlsx and PNG figures in the same folder

Run:  python preprocess_ml.py                  (uses TARGET below)
      python preprocess_ml.py "Biopsy Class"
Requires: numpy, pandas, scipy, scikit-learn, matplotlib, openpyxl
"""
import os
import platform
import re
import sys
import traceback
import warnings
from collections import Counter
from itertools import combinations

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------------------ settings
BASE_DIR = os.environ.get("NEPHRIT_DIR", r"C:\Users\MHC\Downloads\nephrit\2 class")
DATA_FILE = "data.xlsx"
TARGET = "Biopsy Class"                 # ISN/RPS class (3, 4, 5); command-line argument overrides
RUN_MODES = ["binary", "three_class"]   # binary: class III/V (=1) vs class IV (=0)
SHOW_PLOTS = False                      # True -> also open plot windows; PNGs are always saved

# preprocessing
DRUG_NAN_AS_ZERO = True                 # ASSUMPTION: blank drug-days = drug not given (confirm with clinicians)
MISSING_DROP_THRESHOLD = 0.5            # drop columns with more than 50% missing
MIN_CLASS_COUNT = 5                     # drop columns with fewer than 5 values differing from the most frequent
FILL_PROTEIN_IF_FLAG_NEGATIVE = True
PROTEIN_NEGATIVE_FILL_MG = 150
CREATININE_UNIT_DIVISOR = 1             # 1000 if values are creatinine x1000
PROTEIN_THRESHOLD_MG = 500
LOG_COLS = ["Protein_Value_mg", "Creatinine_Value_mg_dL"]
DROP_ALWAYS = ["Case_Number", "Date_of_First_Visit", "Year_of_Birth", "24h_Creatinine_Value_mg_dL"]

# leakage
LEAKAGE_EXACT = ["Has_Biopsy"]
LEAKAGE_SUFFIX = ["_used", "_log1p_days"]       # treatment given after the visit / biopsy result
REDUNDANT_DERIVED_PREFIX = ["Has_Protein"]      # flag derived from Protein_Value
LEAK_AUC_THRESHOLD = 0.95
LEAK_AUTO_DROP = False

# modelling
N_REPEATS = 20
K_FEATURES = 5                          # features kept per training fold (SelectKBest)
STRATEGIES = ["none", "class_weight", "smote"]
SMOTE_K = 5
N_BOOT = 500
N_LABEL_PERM = 200

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", 80)

if not SHOW_PLOTS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PREFIX = "model"


# ------------------------------------------------------------ utilities
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, x):
        for s in self.streams:
            s.write(x)

    def flush(self):
        for s in self.streams:
            s.flush()


def slug(name):
    return re.sub(r"[^0-9A-Za-z]+", "_", str(name).strip()).strip("_")


def banner(title):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def save_fig(fig, name):
    path = os.path.join(BASE_DIR, f"{PREFIX}_{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[figure saved] {path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def stats_table(d, title):
    banner(title + f"\nshape: {d.shape[0]} rows x {d.shape[1]} columns")
    rows = []
    for c in d.columns:
        s = d[c]
        miss = int(s.isna().sum())
        v = s.dropna()
        if not pd.api.types.is_numeric_dtype(s):
            kind, summ = "text", f"{v.nunique()} unique values"
        elif len(v) and set(v.unique()) <= {0, 1}:
            n1 = int((v == 1).sum())
            kind = "binary"
            summ = f"1: {n1} ({100 * n1 / len(v):.1f}%) | 0: {len(v) - n1} ({100 * (len(v) - n1) / len(v):.1f}%)"
        elif v.nunique() <= 6 and (v == v.round()).all():
            vc = v.value_counts().sort_index()
            kind = "categorical"
            summ = " | ".join(f"{k:g}: {n} ({100 * n / len(v):.1f}%)" for k, n in vc.items())
        elif len(v):
            kind = "numeric"
            summ = (f"mean={v.mean():.3f} var={v.var(ddof=1):.3f} sd={v.std(ddof=1):.3f} "
                    f"min={v.min():g} median={v.median():g} max={v.max():g}")
        else:
            kind, summ = "empty", "-"
        rows.append({"column": c, "type": kind, "n_valid": len(v), "missing_n": miss,
                     "missing_%": round(100 * miss / len(d), 1), "summary": summ})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nTotal missing cells: {int(d.isna().sum().sum())}")


# ------------------------------------------------------------ cleaning / engineering
def clean(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    first = df.columns[0]
    n_hdr = int((df[first].astype(str) == first).sum())
    df = df[df[first].astype(str) != first]
    print(f"Removed {n_hdr} repeated header row(s)")
    df.columns = [slug(c) for c in df.columns]
    for c in df.columns:
        if c != "Date_of_First_Visit":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    n0 = len(df)
    dup = df[df.duplicated(subset="Case_Number", keep=False)]
    if len(dup):
        print("Duplicate Case_Number rows found:\n", dup.iloc[:, :6].to_string())
    df = df.drop_duplicates(subset="Case_Number", keep="first")
    print(f"Removed {n0 - len(df)} duplicate Case Number row(s); {len(df)} patients remain")
    return df.reset_index(drop=True)


def engineer(df):
    d = df.copy()
    hp = [c for c in d.columns if c.startswith("Has_Protein")]
    if hp and "Protein_Value_mg" in d:
        m = d[hp[0]].isna() & d["Protein_Value_mg"].notna()
        d.loc[m, hp[0]] = (d.loc[m, "Protein_Value_mg"] > PROTEIN_THRESHOLD_MG).astype(float)
        print(f"Derived {int(m.sum())} missing '{hp[0]}' value(s) from Protein Value (> {PROTEIN_THRESHOLD_MG} mg)")
    if FILL_PROTEIN_IF_FLAG_NEGATIVE and hp and "Protein_Value_mg" in d:
        m = d["Protein_Value_mg"].isna() & (d[hp[0]] == 0)
        d.loc[m, "Protein_Value_mg"] = PROTEIN_NEGATIVE_FILL_MG
        print(f"Filled {int(m.sum())} blank Protein Value(s) with {PROTEIN_NEGATIVE_FILL_MG} mg (flag says no proteinuria)")
    if "Creatinine_Value_mg_dL" in d and CREATININE_UNIT_DIVISOR != 1:
        d["Creatinine_Value_mg_dL"] = d["Creatinine_Value_mg_dL"] / CREATININE_UNIT_DIVISOR
        print(f"Creatinine value divided by {CREATININE_UNIT_DIVISOR}")
    if "Creatinine_Value_mg_dL" in d:
        d["Creatinine_measured"] = d["Creatinine_Value_mg_dL"].notna().astype(float)
        print("Added Creatinine_measured (1 = creatinine value present)")

    drug_cols = [c for c in d.columns if c.endswith("_Days")]
    if DRUG_NAN_AS_ZERO:
        print("Drug-days: blank -> 0 for", drug_cols)
        d[drug_cols] = d[drug_cols].fillna(0)
    for c in drug_cols:
        base = c[: -len("_Days")]
        d[base + "_used"] = (d[c] > 0).astype(float).where(d[c].notna())
        d[base + "_log1p_days"] = np.log1p(d[c])
    d = d.drop(columns=drug_cols)

    sex_cr = [c for c in d.columns if c.startswith("Has_Creatinine") and ("Women" in c or "Men" in c)]
    if sex_cr:
        d["Creatinine_high_sex_adj"] = d[sex_cr].max(axis=1, skipna=True)
        print("Merged", sex_cr, "-> Creatinine_high_sex_adj")
        d = d.drop(columns=sex_cr)

    for c in LOG_COLS:
        if c in d:
            d[c + "_log1p"] = np.log1p(d[c])
            d = d.drop(columns=c)
    print("log1p applied to", LOG_COLS)

    dropped = [c for c in DROP_ALWAYS if c in d.columns]
    print("Dropped (id/date/redundant/too sparse):", dropped)
    return d.drop(columns=dropped)


def drop_high_missing(d, protect=()):
    frac = d.isna().mean()
    drop = [c for c in d.columns if frac[c] > MISSING_DROP_THRESHOLD and c not in protect]
    print(f"Dropped columns with >{int(MISSING_DROP_THRESHOLD * 100)}% missing: {drop}")
    return d.drop(columns=drop)


def drop_near_constant(d, protect=()):
    drop = []
    for c in d.columns:
        if c in protect:
            continue
        v = d[c].dropna()
        if len(v) == 0 or (len(v) - v.value_counts().iloc[0]) < MIN_CLASS_COUNT:
            drop.append(c)
    print(f"Dropped near-constant columns (<{MIN_CLASS_COUNT} values differ from the most frequent): {drop}")
    return d.drop(columns=drop)


def missingness_diagnostics(d):
    bins = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c]) and set(d[c].dropna().unique()) <= {0, 1}]
    banner("MISSINGNESS DIAGNOSTICS (is a value missing because of another variable?)")
    hp = [c for c in d.columns if c.startswith("Has_Protein")]
    if hp and "Protein_Value_mg" in d:
        print(f"Crosstab {hp[0]} (rows) vs Protein_Value_mg missing (columns):")
        print(pd.crosstab(d[hp[0]], d["Protein_Value_mg"].isna(), margins=True, dropna=False).to_string())
    for c in d.columns:
        m = d[c].isna()
        if not m.any():
            continue
        found = []
        for b in bins:
            if b == c:
                continue
            r_miss, r_pres = d.loc[m, b].mean(), d.loc[~m, b].mean()
            if pd.notna(r_miss) and pd.notna(r_pres) and abs(r_miss - r_pres) >= 0.30:
                found.append(f"{b}: 1-rate {r_miss:.2f} when missing vs {r_pres:.2f} when present")
        if found:
            print(f"\n{c} (missing in {int(m.sum())} rows):")
            for f in found:
                print("   ", f)


def _find(d, prefix):
    m = [c for c in d.columns if c.startswith(prefix)]
    return m[0] if m else None


def reference_range_report(d):
    banner("CLINICAL REFERENCE-RANGE CHECK (typical adult ranges; replace with your laboratory's ranges)")
    p = "Protein_Value_mg"
    if p in d:
        bands = pd.cut(d[p], [-np.inf, 150, 500, 3500, np.inf],
                       labels=["<=150 normal", "150-500 mild", "500-3500 proteinuria", ">3500 nephrotic-range"])
        print("\n24h urine protein (mg/24h):\n" + bands.value_counts(dropna=False).sort_index().to_string())
        hp = _find(d, "Has_Protein")
        if hp:
            k = d[p].notna()
            print(f"\nConsistency: reported {hp} (rows) vs value > 500 mg (columns):")
            print(pd.crosstab(d.loc[k, hp], d.loc[k, p] > 500).to_string())
    c24 = "24h_Creatinine_Value_mg_dL"
    if c24 in d:
        bands = pd.cut(d[c24], [-np.inf, 800, 2000, np.inf], labels=["<800 low", "800-2000 normal", ">2000 high"])
        print("\n24h urine creatinine (mg/24h):\n" + bands.value_counts(dropna=False).sort_index().to_string())
    cs, sex = "Creatinine_Value_mg_dL", "Sex"
    if cs in d and sex in d:
        v = d[cs] / CREATININE_UNIT_DIVISOR
        print(f"\nCreatinine value / {CREATININE_UNIT_DIVISOR}: min={v.min():g} median={v.median():g} max={v.max():g} "
              "(adult serum creatinine ~0.6-1.3 mg/dL; if ~1000x larger set CREATININE_UNIT_DIVISOR = 1000)")


def impute_all(d, protect=()):
    d = d.copy()
    for c in d.columns:
        if c in protect or not d[c].isna().any():
            continue
        v = d[c].dropna()
        is_cat = set(v.unique()) <= {0, 1}
        fill = v.mode().iloc[0] if is_cat else v.median()
        print(f"  imputed {int(d[c].isna().sum())} value(s) in {c} with {'mode' if is_cat else 'median'} = {fill:g}")
        d[c] = d[c].fillna(fill)
    return d


# ------------------------------------------------------------ STEP 5: group table + tests
def _welch(a, b):
    if len(a) < 2 or len(b) < 2 or (np.ptp(a) == 0 and np.ptp(b) == 0):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return stats.ttest_ind(a, b, equal_var=False).pvalue


def _anova(samples):
    samples = [s for s in samples if len(s) >= 2]
    if len(samples) < 2 or all(np.ptp(s) == 0 for s in samples):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return stats.f_oneway(*samples).pvalue


def _fp(p):
    return "-" if pd.isna(p) else f"{p:.3f}{'*' if p < 0.05 else ''}"


def group_comparison(d, target):
    banner("STEP 5: FEATURE TABLE - whole data vs each biopsy class, with significance tests")
    groups = sorted(d[target].dropna().unique())
    gname = {g: f"Class {int(g)} (n={int((d[target] == g).sum())})" for g in groups}
    rows = []
    for c in d.columns:
        if c == target:
            continue
        s = d[c]
        v = s.dropna()
        is_bin = set(v.unique()) <= {0, 1}

        def cell(x):
            x = x.dropna()
            if len(x) == 0:
                return "-"
            if is_bin:
                n1 = int((x == 1).sum())
                return f"{n1}/{len(x)} ({100 * n1 / len(x):.1f}%)"
            if len(x) == 1:
                return f"mean {x.mean():.2f}, n=1"
            return f"mean {x.mean():.2f}, var {x.var(ddof=1):.2f}, n={len(x)}"

        row = {"feature": c, "type": "binary: n/N (%)" if is_bin else "numeric: mean, var, n",
               f"Whole data (n={len(d)})": cell(s)}
        samples = []
        for g in groups:
            sg = s[d[target] == g]
            row[gname[g]] = cell(sg)
            samples.append(sg.dropna().values.astype(float))
        row["ANOVA p"] = _fp(_anova(samples))
        for (i, gi), (j, gj) in combinations(list(enumerate(groups)), 2):
            row[f"t-test p ({int(gi)} vs {int(gj)})"] = _fp(_welch(samples[i], samples[j]))
        rows.append(row)
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    print("\n* p < 0.05 (Welch t-test, unequal variances; on 0/1 features it compares proportions). Classes 3 and 5 "
          "have only 4-5 patients and many tests are shown without correction: exploratory only.")
    tab.to_excel(os.path.join(BASE_DIR, "group_comparison.xlsx"), index=False)
    return tab


# ------------------------------------------------------------ STEP 6: leakage
def leakage_screen(d, t):
    y = d[t].astype(int).values
    rows = []
    for c in d.columns:
        if c == t:
            continue
        x = d[c].fillna(d[c].median()).values
        if len(np.unique(x)) < 2:
            continue
        best, bc = 0.0, None
        for k in np.unique(y):
            a = roc_auc_score(y == k, x)
            a = max(a, 1 - a)
            if a > best:
                best, bc = a, k
        rows.append({"feature": c, "max one-vs-rest AUC (single feature)": round(best, 3), "for class": bc})
    return pd.DataFrame(rows).sort_values("max one-vs-rest AUC (single feature)", ascending=False)


def remove_leakage(d, t):
    banner("STEP 6: LEAKAGE REMOVAL (features not known when predicting the biopsy class)")
    manual = [c for c in d.columns if c != t and (c in LEAKAGE_EXACT or any(c.endswith(s) for s in LEAKAGE_SUFFIX))]
    redundant = [c for c in d.columns if c != t and any(c.startswith(p) for p in REDUNDANT_DERIVED_PREFIX)]
    print("Removed - biopsy-related / treatment given after the visit:", manual)
    print("Removed - flag derived from Protein_Value (redundant):", redundant)
    d = d.drop(columns=manual + redundant)
    scr = leakage_screen(d, t)
    print("\nSingle-feature AUC screen (~1.0 = suspiciously perfect):")
    print(scr.head(10).to_string(index=False))
    flagged = scr.loc[scr.iloc[:, 1] >= LEAK_AUC_THRESHOLD, "feature"].tolist()
    if flagged:
        print(f"\nWARNING: features with AUC >= {LEAK_AUC_THRESHOLD}: {flagged}")
        if LEAK_AUTO_DROP:
            d = d.drop(columns=flagged)
            print("Dropped automatically (LEAK_AUTO_DROP=True).")
        else:
            print("Kept (LEAK_AUTO_DROP=False). Review them clinically.")
    else:
        print(f"\nNo remaining feature reaches AUC >= {LEAK_AUC_THRESHOLD}.")
    print(f"Remaining predictors: {d.shape[1] - 1}: {[c for c in d.columns if c != t]}")
    return d


# ------------------------------------------------------------ preprocessing / SMOTE
def build_preprocessor(X):
    num = [c for c in X.columns if X[c].nunique() > 2]
    binary = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        # error_on_new=False: a test fold may have NaN in a column that was complete in training
        ("miss", MissingIndicator(features="missing-only", error_on_new=False), num),
        ("bin", SimpleImputer(strategy="most_frequent"), binary),
    ], verbose_feature_names_out=False)


def smote_resample(X, y, seed, k=SMOTE_K):
    """Minimal SMOTE: interpolate between a minority sample and one of its k nearest same-class neighbours.
    0/1 columns are rounded back. Classes with a single sample are duplicated."""
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    target_n = counts.max()
    bin_cols = np.array([np.isin(np.unique(X[:, j]), [0, 1]).all() for j in range(X.shape[1])])
    Xs, ys = [X], [y]
    for c, n in zip(classes, counts):
        need = target_n - n
        if need <= 0:
            continue
        Xc = X[y == c]
        if n == 1:
            new = np.repeat(Xc, need, axis=0)
        else:
            kk = min(k, n - 1)
            idx = NearestNeighbors(n_neighbors=kk + 1).fit(Xc).kneighbors(Xc, return_distance=False)[:, 1:]
            base = rng.randint(0, n, need)
            nb = idx[base, rng.randint(0, kk, need)]
            new = Xc[base] + rng.rand(need, 1) * (Xc[nb] - Xc[base])
            new[:, bin_cols] = np.round(new[:, bin_cols])
        Xs.append(new)
        ys.append(np.full(need, c))
    return np.vstack(Xs), np.concatenate(ys)


def export_ml_ready(d, target):
    X = d.drop(columns=[target])
    pre = build_preprocessor(X)
    pre.set_output(transform="pandas")
    Xt = pre.fit_transform(X).astype(float)
    Xt[target] = d[target].values
    Xt.to_excel(os.path.join(BASE_DIR, "preprocessed_ml_ready.xlsx"), index=False)
    stats_table(Xt, "STEP 7: ML-ready matrix after leakage removal (imputed, z-scored) - inspection only")


# ------------------------------------------------------------ STEP 8: evaluation
def make_models(cw):
    w = "balanced" if cw else None
    m = {
        "LR L2 (C=0.05)": LogisticRegression(C=0.05, class_weight=w, max_iter=5000),
        "LR L2 (C=0.3)": LogisticRegression(C=0.3, class_weight=w, max_iter=5000),
        "RF shallow": RandomForestClassifier(n_estimators=300, max_depth=2, min_samples_leaf=4, max_features="sqrt",
                                             random_state=0, class_weight="balanced_subsample" if cw else None),
    }
    if not cw:
        m["Gaussian NB"] = GaussianNB()
    return m


def safe_auc(y, P, classes):
    if len(np.unique(y)) < len(classes):
        return np.nan
    if len(classes) == 2:
        return roc_auc_score(y == classes[1], P[:, 1])
    return float(np.mean([roc_auc_score(y == c, P[:, i]) for i, c in enumerate(classes)]))


def brier_skill(y, P, classes):
    oh = (y[:, None] == classes[None, :]).astype(float)
    b = ((P - oh) ** 2).sum(axis=1).mean()
    b0 = ((oh.mean(axis=0)[None, :] - oh) ** 2).sum(axis=1).mean()
    return 1 - b / b0


def native_imp(clf, names):
    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=names)
    if hasattr(clf, "coef_"):
        return pd.Series(np.abs(clf.coef_).mean(axis=0), index=names)
    return None


def cv_predict(X, y, classes, proto, strategy, n_splits, repeats, k=K_FEATURES, seed0=0, light=False):
    n, K = len(y), len(classes)
    P = np.zeros((repeats, n, K))
    L = np.zeros((repeats, n), dtype=int)
    folds, sel_count, imps = [], Counter(), []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in range(repeats):
            for f, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed0 + r).split(X, y), 1):
                pre = build_preprocessor(X)
                Xtr, Xte = pre.fit_transform(X.iloc[tr]), pre.transform(X.iloc[te])
                names = np.array(pre.get_feature_names_out())
                sel = SelectKBest(f_classif, k=min(k, Xtr.shape[1])).fit(Xtr, y[tr])   # training fold only
                Xtr, Xte, names = sel.transform(Xtr), sel.transform(Xte), names[sel.get_support()]
                if strategy == "smote":
                    Xf, yf = smote_resample(Xtr, y[tr], seed=seed0 + r * 100 + f)
                else:
                    Xf, yf = Xtr, y[tr]
                clf = clone(proto).fit(Xf, yf)
                p = clf.predict_proba(Xte)
                # prior correction only when the model saw the natural class balance
                prior = np.array([(y[tr] == c).mean() for c in classes]) if strategy == "none" else np.ones(K)
                P[r, te] = p
                L[r, te] = classes[(p / prior).argmax(1)]
                if light:
                    continue
                ptr = clf.predict_proba(Xtr)
                folds.append({"repeat": r + 1, "fold": f, "n_train": len(tr), "n_test": len(te),
                              "train_auc": safe_auc(y[tr], ptr, classes), "test_auc": safe_auc(y[te], p, classes),
                              "train_bal_acc": balanced_accuracy_score(y[tr], classes[(ptr / prior).argmax(1)]),
                              "test_bal_acc": balanced_accuracy_score(y[te], L[r, te])})
                sel_count.update(names.tolist())
                ni = native_imp(clf, names)
                if ni is not None:
                    imps.append(ni)
    return {"P": P, "L": L, "folds": pd.DataFrame(folds), "sel": sel_count, "imps": imps}


def boot_ci(y, Pm, classes, rng, n=N_BOOT):
    by = [np.where(y == c)[0] for c in classes]
    v = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(i, len(i), replace=True) for i in by])
        v.append(safe_auc(y[idx], Pm[idx], classes))
    return np.nanpercentile(v, [2.5, 97.5])


def evaluate_mode(d, t, mode):
    global PREFIX
    PREFIX = f"{slug(t)}_{mode}"
    d = d.dropna(subset=[t]).reset_index(drop=True)
    X, yraw = d.drop(columns=[t]), d[t].astype(int).values
    if mode == "binary":
        y, desc = (yraw != 4).astype(int), "BINARY: Class III/V (=1) vs Class IV (=0)"
    else:
        y, desc = yraw, "THREE-CLASS: III vs IV vs V"
    classes = np.sort(np.unique(y))
    K = len(classes)
    counts = pd.Series(y).value_counts().sort_index()
    n_splits = min(5, int(counts.min()))
    banner(f"STEP 8: EVALUATION - {desc}")
    print("Target distribution:\n" + counts.to_string())
    print(f"{n_splits}-fold x {N_REPEATS} repeats; impute/scale/select top {K_FEATURES} features/SMOTE are fitted "
          f"inside each training fold; strategies: {STRATEGIES}")
    if n_splits < 2:
        print("Smallest class has <2 samples: cannot cross-validate.")
        return
    rng = np.random.RandomState(0)

    runs, rows = {}, []
    for strat in STRATEGIES:
        for nm, proto in make_models(strat == "class_weight").items():
            key = f"{nm} | {strat}"
            print(f"  running {key} ...")
            res = cv_predict(X, y, classes, proto, strat, n_splits, N_REPEATS)
            P, L = res["P"], res["L"]
            aucs = np.array([safe_auc(y, P[r], classes) for r in range(N_REPEATS)])
            ba = np.array([balanced_accuracy_score(y, L[r]) for r in range(N_REPEATS)])
            bs = np.array([brier_skill(y, P[r], classes) for r in range(N_REPEATS)])
            lo, hi = boot_ci(y, P.mean(0), classes, rng)
            runs[key] = dict(res=res, proto=proto, strat=strat, auc=aucs.mean(), lo=lo, hi=hi)
            rows.append({"config": key, "AUC (mean ± SD over repeats)": f"{aucs.mean():.3f} ± {aucs.std():.3f}",
                         "AUC 95% CI (bootstrap)": f"{lo:.2f} - {hi:.2f}",
                         "balanced acc": f"{ba.mean():.3f} ± {ba.std():.3f}",
                         "Brier skill": f"{bs.mean():.3f}", "_auc": aucs.mean()})
    tab = pd.DataFrame(rows).sort_values("_auc", ascending=False).reset_index(drop=True)
    banner(f"RESULTS ({mode}). No-skill reference: AUC 0.500, balanced acc {1 / K:.3f}, Brier skill 0.000")
    print(tab.drop(columns="_auc").to_string(index=False))

    best = tab.loc[0, "config"]
    B = runs[best]
    res = B["res"]
    Pm = res["P"].mean(0)

    # ---- label-permutation test
    print(f"\nLabel-permutation test for '{best}' ({N_LABEL_PERM} shuffles of the outcome, full CV each)...")
    null = []
    for i in range(N_LABEL_PERM):
        yp = rng.permutation(y)
        rp = cv_predict(X, yp, classes, B["proto"], B["strat"], n_splits, 1, seed0=1000 + i, light=True)
        null.append(safe_auc(yp, rp["P"][0], classes))
    null = np.array(null)
    pval = (1 + np.sum(null >= B["auc"])) / (len(null) + 1)
    print(f"observed AUC {B['auc']:.3f} | null AUC mean {np.nanmean(null):.3f}, 95th percentile "
          f"{np.nanpercentile(null, 95):.3f} | permutation p = {pval:.3f}")
    ok = (B["lo"] > 0.5) and (pval < 0.05)
    print("\nVERDICT: " + (
        f"'{best}' performs above chance (CI excludes 0.5 and p < 0.05). Still an internal estimate on {len(y)} "
        f"patients; the best of {len(runs)} configurations was selected, which inflates it slightly."
        if ok else
        "NO configuration is shown to beat chance. Do not claim these features predict the class. "
        "The feature table (STEP 5) is exploratory only."))

    # ---- train vs test
    f = res["folds"]
    print(f"\nTRAIN vs TEST per fold (repeat 1) for {best}:")
    print(f[f.repeat == 1].drop(columns="repeat").round(3).to_string(index=False))
    print("Mean ± SD over all folds:  " + "  ".join(
        f"{c}={f[c].mean():.3f}±{f[c].std():.3f}" for c in ["train_auc", "test_auc", "train_bal_acc", "test_bal_acc"]))
    print("(fold-level test AUC uses 1-3 minority patients, so it is very noisy; trust the pooled AUC)")

    # ---- confusion matrix
    cm = sum(confusion_matrix(y, res["L"][r], labels=classes) for r in range(N_REPEATS)) / N_REPEATS
    print(f"\nMean confusion matrix, {best} (rows = true, columns = predicted):")
    print(pd.DataFrame(cm, index=[f"true {c}" for c in classes],
                       columns=[f"pred {c}" for c in classes]).round(2).to_string())

    # ---- feature importance
    selfreq = pd.Series(res["sel"]).div(n_splits * N_REPEATS)
    out = pd.DataFrame({"selected_in_fraction_of_folds": selfreq})
    if res["imps"]:
        out["mean_native_importance"] = pd.concat(res["imps"], axis=1).fillna(0).mean(axis=1)
    out = out.sort_values("selected_in_fraction_of_folds", ascending=False)
    print(f"\nFEATURE STABILITY for {best} ({'validated model' if ok else 'NOT validated: exploratory'}):")
    print("selection fraction = share of folds where the feature was in the top-K; native importance = "
          "|coefficient| (LR) or impurity importance (RF)")
    print(out.round(3).to_string())

    # ---- figures
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    cfg = tab["config"].tolist()[::-1]
    a = np.array([runs[c]["auc"] for c in cfg])
    lo_a = np.array([runs[c]["lo"] for c in cfg])
    hi_a = np.array([runs[c]["hi"] for c in cfg])
    ax[0, 0].errorbar(a, range(len(cfg)), xerr=[np.clip(a - lo_a, 0, None), np.clip(hi_a - a, 0, None)],
                      fmt="o", capsize=3)
    ax[0, 0].axvline(0.5, color="k", ls=":")
    ax[0, 0].set_yticks(range(len(cfg)))
    ax[0, 0].set_yticklabels(cfg, fontsize=7)
    ax[0, 0].set(title="AUC with bootstrap 95% CI (dotted = chance)", xlabel="AUC")
    for i in ([1] if K == 2 else range(K)):
        yy = (y == classes[i]).astype(int)
        fpr, tpr, _ = roc_curve(yy, Pm[:, i])
        ax[0, 1].plot(fpr, tpr, label=f"class {classes[i]} (AUC {roc_auc_score(yy, Pm[:, i]):.2f})")
        try:
            fr, mp = calibration_curve(yy, Pm[:, i], n_bins=5, strategy="quantile")
            ax[1, 0].plot(mp, fr, marker="o", label=f"class {classes[i]}")
        except ValueError:
            pass
    ax[0, 1].plot([0, 1], [0, 1], "k:")
    ax[0, 1].set(title=f"ROC, {best}", xlabel="1 - specificity", ylabel="sensitivity")
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot([0, 1], [0, 1], "k:")
    ax[1, 0].set(title="Calibration (best config)", xlabel="mean predicted", ylabel="observed fraction")
    ax[1, 0].legend(fontsize=8)
    rn = cm / cm.sum(axis=1, keepdims=True)
    ax[1, 1].imshow(rn, vmin=0, vmax=1, cmap="Blues")
    for i in range(K):
        for j in range(K):
            ax[1, 1].text(j, i, f"{cm[i, j]:.1f}\n({rn[i, j]:.0%})", ha="center", va="center",
                          color="white" if rn[i, j] > 0.55 else "black")
    ax[1, 1].set_xticks(range(K))
    ax[1, 1].set_xticklabels([f"pred {c}" for c in classes])
    ax[1, 1].set_yticks(range(K))
    ax[1, 1].set_yticklabels([f"true {c}" for c in classes])
    ax[1, 1].set_title("Confusion matrix (mean over repeats)")
    fig.suptitle(desc)
    save_fig(fig, "summary")

    with pd.ExcelWriter(os.path.join(BASE_DIR, f"results_{mode}.xlsx")) as xw:
        tab.drop(columns="_auc").to_excel(xw, sheet_name="all_configs", index=False)
        f.to_excel(xw, sheet_name="best_per_fold", index=False)
        out.to_excel(xw, sheet_name="feature_stability")


# ------------------------------------------------------------ main
def run(target):
    path = os.path.join(BASE_DIR, DATA_FILE)
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")
    print(f"python {platform.python_version()} | pandas {pd.__version__} | numpy {np.__version__} | "
          f"scikit-learn {sklearn.__version__}")
    print(f"input : {path}\noutput: {BASE_DIR}")
    print(f"settings: DRUG_NAN_AS_ZERO={DRUG_NAN_AS_ZERO}, MISSING_DROP_THRESHOLD={MISSING_DROP_THRESHOLD}, "
          f"MIN_CLASS_COUNT={MIN_CLASS_COUNT}, K_FEATURES={K_FEATURES}, N_REPEATS={N_REPEATS}, "
          f"STRATEGIES={STRATEGIES}, modes={RUN_MODES}, target={target}")

    raw = pd.read_excel(path)
    print(f"\nSTEP 0: raw file read: {raw.shape[0]} rows x {raw.shape[1]} columns")
    d = clean(raw)
    stats_table(d, "STEP 1: after cleaning (repeated headers and duplicate cases removed, numeric conversion)")
    missingness_diagnostics(d)
    reference_range_report(d)
    t = slug(target)

    print("\n--- feature engineering ---")
    d = engineer(d)
    if t not in d.columns:
        sys.exit(f"Target '{t}' not found after engineering. Columns: {list(d.columns)}")
    stats_table(d, "STEP 2: after feature engineering (all candidate features, leakage not yet removed)")

    print("\n--- column filtering ---")
    d = drop_high_missing(d, protect=[t])
    d = drop_near_constant(d, protect=[t])
    stats_table(d, "STEP 3: after dropping high-missing and rare columns (NaN kept, imputed inside CV folds)")

    print("\n--- simple imputation (descriptive copy) ---")
    d_imp = impute_all(d, protect=[t])
    stats_table(d_imp, "STEP 4: imputed copy (median/mode), descriptive use only")
    d.to_excel(os.path.join(BASE_DIR, "cleaned_data.xlsx"), index=False)
    d_imp.to_excel(os.path.join(BASE_DIR, "cleaned_data_imputed.xlsx"), index=False)

    d = d.dropna(subset=[t]).reset_index(drop=True)
    group_comparison(d, t)              # STEP 5
    d_ml = remove_leakage(d, t)         # STEP 6
    export_ml_ready(d_ml, t)            # STEP 7
    for mode in RUN_MODES:              # STEP 8
        evaluate_mode(d_ml, t, mode)
    print("\nDONE. Files saved in", BASE_DIR)


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    logf = open(os.path.join(BASE_DIR, "preprocess_log.txt"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, logf)
    try:
        run(sys.argv[1] if len(sys.argv) > 1 else TARGET)
    except SystemExit as e:
        print(e)
    except Exception:
        traceback.print_exc(file=sys.stdout)
    finally:
        sys.stdout = sys.__stdout__
        logf.close()
        print("Log saved to", os.path.join(BASE_DIR, "preprocess_log.txt"))


if __name__ == "__main__":
    main()