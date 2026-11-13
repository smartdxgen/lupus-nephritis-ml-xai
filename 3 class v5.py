"""
Lupus nephritis (biopsy class) - preprocessing, statistics, leakage control, honest ML evaluation.

Pipeline
  1. cleaning + feature engineering
  2. STEP 5: statistics of ALL features per class (Fisher / permutation chi2 / Kruskal / Mann-Whitney, BH-FDR q-values)
  3. STEP 6: leakage removal (treatment, Has_Biopsy, derived flags) - models never see them
  4. STEP 8-9: repeated stratified CV for several TASKS (3-class and IV-vs-rest), preprocessing refitted per fold,
     class-weighting instead of SMOTE by default, bootstrap CIs, LABEL-PERMUTATION TEST (is there any signal at all?),
     and a sensitivity run without "test-ordering" proxy features
  5. STEP 10: feature importance of the best model (flagged as non-interpretable if no signal)
  6. STEP 11: fold-wise SHAP of the best-ranked tree model (held-out patients only)

Everything printed (stdout + stderr) is also written to preprocess_log.txt.

Reads  : <BASE_DIR>\\data.xlsx
Writes : EVERYTHING goes into BASE_DIR (the folder of the data and the script): preprocess_log.txt,
         cleaned_data*.xlsx, preprocessed_ml_ready.xlsx, group_comparison.xlsx, <target>_<task>_*.xlsx / *.png,
         <target>_<task>_shap_*.png / shap_importance.xlsx
Run    : python preprocess_ml_v4.py            (or: python preprocess_ml_v4.py "Biopsy Class")
"""
import os
import platform
import re
import sys
import traceback
import warnings
from itertools import combinations
from math import ceil

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             log_loss, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

try:                                   # optional: pip install xgboost
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False
try:                                   # optional: pip install shap
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False
    print("NOTE: SHAP is not installed (pip install shap) -> STEP 11 will be skipped.")

# ------------------------------------------------------------ settings
BASE_DIR = r"C:\Users\MHC\Downloads\nephrit\3 class"
DATA_FILE = "data.xlsx"
TARGET = "Biopsy Class"        # ISN/RPS class (3, 4, 5); command-line argument overrides

# (task name, classes treated as positive). None = original multi-class problem.
# Other clinically useful options: ("III_IV_vs_V", [3, 4])  -> proliferative vs membranous
TASKS = [("3class", None), ("IV_vs_rest", [4])]

N_REPEATS = 10                 # repeats of stratified k-fold (use 20 for final results)
SHOW_PLOTS = False             # True -> also open plot windows; PNGs are always saved

DRUG_NAN_AS_ZERO = True        # ASSUMPTION: blank drug-days = drug not given (confirm with clinicians)
MISSING_DROP_THRESHOLD = 0.5
MIN_CLASS_COUNT = 5
FILL_PROTEIN_IF_FLAG_NEGATIVE = True
PROTEIN_NEGATIVE_FILL_MG = 150
CREATININE_UNIT_DIVISOR = 1
PROTEIN_THRESHOLD_MG = 500
LOG_COLS = ["Protein_Value_mg", "Creatinine_Value_mg_dL"]
DROP_ALWAYS = ["Case_Number", "Date_of_First_Visit", "Year_of_Birth", "24h_Creatinine_Value_mg_dL"]

# ---- leakage settings
LEAKAGE_EXACT = ["Has_Biopsy"]
LEAKAGE_SUFFIX = ["_used", "_log1p_days"]           # treatment given AFTER the visit / after the biopsy result
REDUNDANT_DERIVED_PREFIX = ["Has_Protein"]          # flag derived from Protein_Value
# "Was a test ordered?" features: kept in the main analysis, dropped in the sensitivity analysis.
# If they were not a fixed pre-biopsy protocol, move them into LEAKAGE_EXACT.
ORDERING_PROXY_FEATURES = ["Creatinine_measured", "Has_Creatinine_Urine_1_8g_24h"]
LEAK_AUC_THRESHOLD = 0.95
LEAK_AUTO_DROP = False
CORR_THRESHOLD = 0.8

# ---- imbalance handling
USE_SMOTE = False              # default OFF: with 3-4 real patients in a class SMOTE only memorizes them.
SMOTE_K = 5                    # (class_weight="balanced" is used instead)
SMOTE_TARGET_RATIO = 1.0

# ---- inference on model performance
N_BOOT = 1000                  # bootstrap resamples for AUC confidence interval
N_LABEL_PERM = 100             # label permutations for the "is there any signal?" test (use 500+ for final)
PERM_REPEATS = 2               # CV repeats inside each permutation (observed AUC uses the same design)
N_PERM = 10                    # permutation-importance shuffles per fold
N_IMP_REPEATS = 5
# ---- SHAP explainability (STEP 11)
RUN_SHAP = True
SHAP_EVEN_IF_NO_SIGNAL = True  # True: still run SHAP when the permutation test finds no signal (plots are labelled)
SHAP_MAX_FEATURES = 15
SHAP_REPEATS = N_REPEATS
SHAP_SEED = 2026
N_PERM_TEST = 5000             # permutations for 3-group binary-feature test in STEP 5
RUN_SENSITIVITY = True         # repeat comparison without ORDERING_PROXY_FEATURES
# None = run every model below. Or list names to keep, e.g. ["Random forest", "XGBoost"] (baseline is always kept).
# More models = stricter Bonferroni correction in the signal check (p x number of models).
MODELS_TO_RUN = None

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", 80)

if not SHOW_PLOTS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PREFIX = "model"


# ------------------------------------------------------------ logging (terminal + file)
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, x):
        for s in self.streams:
            try:
                s.write(x)
            except UnicodeEncodeError:
                s.write(x.encode("ascii", "replace").decode())

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def slug(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(name).strip()).strip("_")


def banner(title):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def outpath(name):
    return os.path.join(BASE_DIR, f"{PREFIX}_{name}")


def save_fig(fig, name):
    path = outpath(f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[figure saved] {path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ------------------------------------------------------------ descriptive statistics printer
def stats_table(d: pd.DataFrame, title: str):
    print("\n" + "=" * 110)
    print(title)
    print(f"shape: {d.shape[0]} rows x {d.shape[1]} columns")
    print("=" * 110)
    rows = []
    for c in d.columns:
        s = d[c]
        miss = int(s.isna().sum())
        v = s.dropna()
        if not pd.api.types.is_numeric_dtype(s):
            kind, summ = "text", f"{v.nunique()} unique values"
        elif len(v) and set(v.unique()) <= {0, 1}:
            n1 = int((v == 1).sum())
            kind, summ = "binary", f"1: {n1} ({100 * n1 / len(v):.1f}%) | 0: {len(v) - n1} ({100 * (len(v) - n1) / len(v):.1f}%)"
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


# ------------------------------------------------------------ cleaning
def clean(df: pd.DataFrame) -> pd.DataFrame:
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


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    hp = [c for c in d.columns if c.startswith("Has_Protein")]
    if hp and "Protein_Value_mg" in d:
        m = d[hp[0]].isna() & d["Protein_Value_mg"].notna()
        d.loc[m, hp[0]] = (d.loc[m, "Protein_Value_mg"] > PROTEIN_THRESHOLD_MG).astype(float)
        print(f"Derived {int(m.sum())} missing '{hp[0]}' value(s) from Protein Value (> {PROTEIN_THRESHOLD_MG} mg)")

    if FILL_PROTEIN_IF_FLAG_NEGATIVE and hp and "Protein_Value_mg" in d:
        m = d["Protein_Value_mg"].isna() & (d[hp[0]] == 0)
        d.loc[m, "Protein_Value_mg"] = PROTEIN_NEGATIVE_FILL_MG
        print(f"Filled {int(m.sum())} blank Protein Value(s) with normal-range value {PROTEIN_NEGATIVE_FILL_MG} mg (flag says no proteinuria)")

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


# ------------------------------------------------------------ missing values / rare features
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
    print(f"Dropped near-constant columns (<{MIN_CLASS_COUNT} values differ from the most frequent value): {drop}")
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
            print(f"\nConsistency, rows with a value: reported {hp} (rows) vs value > 500 mg (columns):")
            print(pd.crosstab(d.loc[k, hp], d.loc[k, p] > 500).to_string())

    c24 = "24h_Creatinine_Value_mg_dL"
    if c24 in d:
        bands = pd.cut(d[c24], [-np.inf, 800, 2000, np.inf], labels=["<800 low", "800-2000 normal", ">2000 high"])
        print("\n24h urine creatinine (mg/24h):\n" + bands.value_counts(dropna=False).sort_index().to_string())
        fu = _find(d, "Has_Creatinine_Urine")
        if fu:
            k = d[c24].notna()
            print(f"\nConsistency, rows with a value: reported {fu} (rows) vs value > 1800 mg (columns):")
            print(pd.crosstab(d.loc[k, fu], d.loc[k, c24] > 1800).to_string())

    cs, sex = "Creatinine_Value_mg_dL", "Sex"
    wf, wm = _find(d, "Has_Creatinine_Women"), _find(d, "Has_Creatinine_Men")
    if cs in d and sex in d:
        v = d[cs] / CREATININE_UNIT_DIVISOR
        print(f"\nCreatinine value / {CREATININE_UNIT_DIVISOR}: min={v.min():g} median={v.median():g} max={v.max():g} "
              "(adult serum creatinine is normally ~0.6-1.3 mg/dL; if ~1000x larger set CREATININE_UNIT_DIVISOR = 1000)")
        if wf and wm:
            k = v.notna().values
            for code in (1, 0):
                female = (d[sex] == code).values
                implied = pd.Series(v.values > np.where(female, 1.05, 1.35), name="implied (F>1.05, M>1.35)")[k]
                reported = pd.Series(np.where(female, d[wf], d[wm]), name="reported flag")[k]
                print(f"\nAssuming Sex=={code} is female: reported flag (rows) vs implied flag (columns):")
                print(pd.crosstab(reported, implied).to_string())


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


# ------------------------------------------------------------ STEP 5: statistics of ALL features per class
def _counts(a):
    return [int((a == 1).sum()), int((a == 0).sum())]


def _perm_chi2(samples, n_perm=N_PERM_TEST, seed=0):
    """Permutation chi-square test for a binary feature across >2 groups (valid for tiny groups)."""
    x = np.concatenate(samples)
    lab = np.concatenate([np.full(len(s), i) for i, s in enumerate(samples)])
    n_g = np.array([len(s) for s in samples], dtype=float)
    p = x.mean()
    if p in (0, 1):
        return np.nan

    def stat(l):
        ones = np.bincount(l[x == 1], minlength=len(samples)).astype(float)
        return ((ones - n_g * p) ** 2 / (n_g * p) + ((n_g - ones) - n_g * (1 - p)) ** 2 / (n_g * (1 - p))).sum()

    obs, rng = stat(lab), np.random.RandomState(seed)
    hits = sum(stat(rng.permutation(lab)) >= obs - 1e-12 for _ in range(n_perm))
    return (hits + 1) / (n_perm + 1)


def _p_overall(samples, is_bin):
    samples = [s for s in samples if len(s) > 0]
    if len(samples) < 2 or np.ptp(np.concatenate(samples)) == 0:
        return np.nan
    if is_bin:
        if len(samples) == 2:
            return stats.fisher_exact([_counts(samples[0]), _counts(samples[1])])[1]
        return _perm_chi2(samples)
    try:
        return stats.kruskal(*samples).pvalue
    except ValueError:
        return np.nan


def _p_pair(a, b, is_bin):
    if len(a) == 0 or len(b) == 0 or np.ptp(np.concatenate([a, b])) == 0:
        return np.nan
    if is_bin:
        return stats.fisher_exact([_counts(a), _counts(b)])[1]
    try:
        return stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    except ValueError:
        return np.nan


def bh_adjust(p):
    """Benjamini-Hochberg FDR-adjusted p-values (NaN kept)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1)
    out[ok] = adj
    return out


def _fp(p):
    return "-" if pd.isna(p) else f"{p:.3f}{'*' if p < 0.05 else ''}"


def group_comparison(d: pd.DataFrame, target: str, status: dict) -> pd.DataFrame:
    """Statistics for ALL features. `status` maps feature -> text shown in the in_model column."""
    banner("STEP 5: FEATURE TABLE - ALL features, whole data vs each biopsy class, with significance tests")
    groups = sorted(d[target].dropna().unique())
    gname = {g: f"Class {int(g)} (n={int((d[target] == g).sum())})" for g in groups}
    rows, p_raw = [], []
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

        row = {"feature": c, "in_model": status.get(c, "yes"),
               "type": "binary: n/N (%)" if is_bin else "numeric: mean, var, n",
               f"Whole data (n={len(d)})": cell(s)}
        samples = []
        for g in groups:
            sg = s[d[target] == g]
            row[gname[g]] = cell(sg)
            samples.append(sg.dropna().values.astype(float))
        p = _p_overall(samples, is_bin)
        p_raw.append(p)
        row["overall p"] = _fp(p)
        for (i, gi), (j, gj) in combinations(list(enumerate(groups)), 2):
            row[f"p ({int(gi)} vs {int(gj)})"] = _fp(_p_pair(samples[i], samples[j], is_bin))
        rows.append(row)
    tab = pd.DataFrame(rows)
    q = bh_adjust(p_raw)
    tab.insert(tab.columns.get_loc("overall p") + 1, "overall q (BH-FDR)", [_fp(x) for x in q])
    print(tab.to_string(index=False))
    n_sig_p = int(np.nansum(np.array(p_raw) < 0.05))
    n_sig_q = int(np.nansum(q < 0.05))
    print(f"\n* p < 0.05. Binary: Fisher exact (2 groups) / permutation chi-square (3 groups). Numeric: "
          f"Kruskal-Wallis / Mann-Whitney. Features with overall p < 0.05: {n_sig_p}; with FDR q < 0.05: {n_sig_q} "
          f"(of {len(p_raw)} tested). Classes 3 and 5 have only 4-5 patients, so all p-values are exploratory. "
          "Features not marked 'yes' are shown for description only and are NOT used by the models "
          "('ordering proxy' features are used in the main analysis and dropped in the sensitivity analysis).")
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


def leakage_lists(d, t):
    manual = [c for c in d.columns if c != t and (c in LEAKAGE_EXACT or any(c.endswith(s) for s in LEAKAGE_SUFFIX))]
    redundant = [c for c in d.columns if c != t and any(c.startswith(p) for p in REDUNDANT_DERIVED_PREFIX)]
    return manual, redundant


def feature_status(d, t):
    manual, redundant = leakage_lists(d, t)
    st = {c: "NO (leakage)" for c in manual}
    st.update({c: "NO (redundant)" for c in redundant})
    st.update({c: "yes (ordering proxy)" for c in ORDERING_PROXY_FEATURES if c in d.columns and c not in st})
    return st


def correlation_report(d, t, thr=CORR_THRESHOLD):
    X = d.drop(columns=[t])
    corr = X.fillna(X.median()).corr(method="spearman")
    pairs = [(a, b, corr.loc[a, b]) for a, b in combinations(corr.columns, 2) if abs(corr.loc[a, b]) >= thr]
    print(f"\nHighly correlated predictor pairs kept in the model (|Spearman| >= {thr}):")
    if pairs:
        for a, b, r in sorted(pairs, key=lambda x: -abs(x[2])):
            print(f"   {a}  <->  {b}: rho = {r:.2f}")
    else:
        print("   none")


def remove_leakage(d, t):
    banner("STEP 6: LEAKAGE REMOVAL (features that would not be known when predicting biopsy class)")
    manual, redundant = leakage_lists(d, t)
    print("Removed - biopsy-related / treatment given after the visit (leakage):", manual)
    print("Removed - flag derived from Protein_Value (redundant, also used to fill it):", redundant)
    d = d.drop(columns=manual + redundant)
    scr = leakage_screen(d, t)
    print("\nStatistical leakage screen on the remaining features (single-feature AUC; ~1.0 = suspiciously perfect):")
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
    correlation_report(d, t)
    feats = [c for c in d.columns if c != t]
    print(f"\nFEATURES USED BY THE MODELS ({len(feats)}): {feats}")
    print(f"Of these, 'ordering proxy' features (test ordered / not): "
          f"{[c for c in ORDERING_PROXY_FEATURES if c in feats]}")
    return d


# ------------------------------------------------------------ preprocessing / SMOTE / models
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = [c for c in X.columns if X[c].nunique() > 2]
    binary = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("miss", MissingIndicator(features="missing-only", error_on_new=False), num),
        ("bin", SimpleImputer(strategy="most_frequent"), binary),
    ], verbose_feature_names_out=False)


def smote_resample(X, y, seed, k=SMOTE_K, ratio=SMOTE_TARGET_RATIO):
    """Minimal SMOTE (optional; see USE_SMOTE). Binary columns are rounded back to 0/1."""
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    target_n = int(counts.max() * ratio)
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


class ImbalanceWrapper(ClassifierMixin, BaseEstimator):
    """Gives any classifier (a) arbitrary class labels (e.g. 3/4/5, needed by XGBoost) and (b) class balancing:
    mode="weights"    -> balanced sample weights (models with sample_weight: XGBoost, AdaBoost)
    mode="oversample" -> random duplication of minority patients inside the training fold (models without
                         sample_weight: neural network). No synthetic patients are created."""

    def __init__(self, base=None, mode="weights", seed=0):
        self.base = base
        self.mode = mode
        self.seed = seed

    def fit(self, X, y):
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        ye = self.le_.transform(y)
        self.base_ = clone(self.base)
        if self.mode == "oversample":
            rng = np.random.RandomState(self.seed)
            cls, cnt = np.unique(ye, return_counts=True)
            idx = [np.arange(len(ye))]
            for c, n in zip(cls, cnt):
                if n < cnt.max():
                    idx.append(rng.choice(np.where(ye == c)[0], cnt.max() - n, replace=True))
            idx = np.concatenate(idx)
            self.base_.fit(X[idx], ye[idx])
        else:
            self.base_.fit(X, ye, sample_weight=compute_sample_weight("balanced", ye))
        return self

    def predict_proba(self, X):
        return self.base_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.asarray(self.base_.predict(X)).astype(int)]

    @property
    def feature_importances_(self):
        return self.base_.feature_importances_


_XGB_WARNED = False


def get_models():
    """Model zoo. All are small/regularised because 52 patients cannot support complex models."""
    global _XGB_WARNED
    cw = "balanced"
    m = {
        "No-skill baseline": DummyClassifier(strategy="prior"),
        "Logistic regression (L2, C=0.1)": LogisticRegression(C=0.1, class_weight=cw, max_iter=5000),
        "Logistic regression (L2, C=1)": LogisticRegression(C=1.0, class_weight=cw, max_iter=5000),
        "SVM (RBF)": SVC(C=1.0, kernel="rbf", gamma="scale", class_weight=cw, probability=True, random_state=0),
        "Random forest": RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=3,
                                                class_weight="balanced_subsample", random_state=0),
        "Extra trees": ExtraTreesClassifier(n_estimators=200, max_depth=3, min_samples_leaf=3,
                                            class_weight=cw, random_state=0),
        "AdaBoost (stumps)": ImbalanceWrapper(AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, learning_rate=0.5, random_state=0)),
        "ANN (MLP 8 units)": ImbalanceWrapper(MLPClassifier(
            hidden_layer_sizes=(8,), alpha=1.0, max_iter=2000, random_state=0), mode="oversample"),
    }
    if XGB_OK:
        m["XGBoost"] = ImbalanceWrapper(XGBClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=5.0, n_jobs=1, random_state=0, verbosity=0))
    elif not _XGB_WARNED:
        print("NOTE: xgboost is not installed -> XGBoost skipped. Install with:  pip install xgboost")
        _XGB_WARNED = True
    if MODELS_TO_RUN:
        keep = set(MODELS_TO_RUN) | {"No-skill baseline"}
        m = {k: v for k, v in m.items() if k in keep}
    return m


def export_ml_ready(d, target):
    X = d.drop(columns=[target])
    pre = build_preprocessor(X)
    print("\nVariable groups used by the preprocessor:")
    for name, _, cols in pre.transformers:
        print(f"  {name}: {list(cols)}")
    pre.set_output(transform="pandas")
    Xt = pre.fit_transform(X).astype(float)
    Xt[target] = d[target].values
    Xt.to_excel(os.path.join(BASE_DIR, "preprocessed_ml_ready.xlsx"), index=False)
    stats_table(Xt, "STEP 7: ML-ready matrix after leakage removal (imputed, z-scored) - for inspection only")


# ------------------------------------------------------------ metrics
def auc_macro(y, P, classes):
    if len(classes) == 2:
        return roc_auc_score(y == classes[1], P[:, 1]) if len(np.unique(y)) > 1 else np.nan
    a = []
    for i, c in enumerate(classes):
        if 0 < (y == c).sum() < len(y):
            a.append(roc_auc_score(y == c, P[:, i]))
    return float(np.mean(a)) if a else np.nan


def norm(P):
    P = np.clip(P, 1e-6, 1)
    return P / P.sum(axis=1, keepdims=True)


def brier(y, P, classes):
    onehot = (y[:, None] == classes[None, :]).astype(float)
    return float(((P - onehot) ** 2).sum(axis=1).mean())


def ece_top(y, P, classes, bins=10):
    conf = P.max(axis=1)
    correct = (classes[P.argmax(axis=1)] == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def pooled_metrics(y, P, classes):
    pred = classes[P.argmax(axis=1)]
    out = {"bal_acc": balanced_accuracy_score(y, pred),
           "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
           "macro_auc": auc_macro(y, P, classes),
           "brier": brier(y, P, classes),
           "log_loss": log_loss(y, norm(P), labels=classes),
           "ece": ece_top(y, P, classes)}
    if len(classes) == 2:
        out["pr_auc"] = average_precision_score(y == classes[1], P[:, 1])
    for i, c in enumerate(classes):
        out[f"recall_{c}"] = float((pred[y == c] == c).mean())
        out[f"auc_{c}"] = roc_auc_score(y == c, P[:, i]) if len(classes) > 2 else np.nan
    return out


def ms(s):
    return f"{np.mean(s):.3f} ± {np.std(s):.3f}"


def bootstrap_auc_ci(y, P, classes, n_boot=N_BOOT, seed=0):
    """Patient-level bootstrap CI of the macro AUC of the repeat-averaged out-of-fold probabilities."""
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < len(classes):
            continue
        vals.append(auc_macro(y[idx], P[idx], classes))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ------------------------------------------------------------ cross-validation engine
def cv_oof(proto, X, y, classes, n_splits, repeats, use_smote, seed0=0, collect=False, name=""):
    """Repeated stratified CV; preprocessing (and SMOTE if enabled) refitted inside every training fold."""
    n, K = len(y), len(classes)
    oof = np.zeros((repeats, n, K))
    rows = []
    for r in range(repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed0 + r)
        for f, (tr, te) in enumerate(skf.split(X, y), 1):
            pre = build_preprocessor(X)
            Xtr = pre.fit_transform(X.iloc[tr])
            Xte = pre.transform(X.iloc[te])
            if use_smote and not isinstance(proto, DummyClassifier):
                Xfit, yfit = smote_resample(Xtr, y[tr], seed=(seed0 + r) * 100 + f)
            else:
                Xfit, yfit = Xtr, y[tr]
            clf = clone(proto).fit(Xfit, yfit)
            p_te = clf.predict_proba(Xte)
            oof[r, te] = p_te
            if collect:
                p_tr = clf.predict_proba(Xtr)
                rows.append({
                    "model": name, "repeat": r + 1, "fold": f, "n_train": len(tr), "n_test": len(te),
                    "train_bal_acc": balanced_accuracy_score(y[tr], classes[p_tr.argmax(1)]),
                    "test_bal_acc": balanced_accuracy_score(y[te], classes[p_te.argmax(1)]),
                    "train_auc": auc_macro(y[tr], p_tr, classes),
                    "test_auc": auc_macro(y[te], p_te, classes),
                    "test_f1": f1_score(y[te], classes[p_te.argmax(1)], average="macro", zero_division=0),
                    "test_log_loss": log_loss(y[te], norm(p_te), labels=classes)})
    return oof, rows


def label_permutation_test(proto, X, y, classes, n_splits, use_smote, observed, n_perm=N_LABEL_PERM, reps=PERM_REPEATS):
    """Null distribution of the CV macro AUC when labels carry no information (same pipeline, shuffled labels)."""
    rng = np.random.RandomState(12345)
    null = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n_perm):
            yp = rng.permutation(y)
            oof, _ = cv_oof(proto, X, yp, classes, n_splits, reps, use_smote, seed0=1000 + i * reps)
            null.append(np.mean([auc_macro(yp, oof[r], classes) for r in range(reps)]))
            if (i + 1) % 20 == 0:
                print(f"    permutation {i + 1}/{n_perm}")
    null = np.array(null)
    p = (1 + int((null >= observed - 1e-12).sum())) / (n_perm + 1)
    return p, null


# ------------------------------------------------------------ plots
def plot_roc(oof, y, classes, names):
    K = len(classes)
    idx = list(range(K)) if K > 2 else [1]
    fig, axes = plt.subplots(1, len(idx), figsize=(5.4 * len(idx), 5), squeeze=False)
    for ax, j in zip(axes[0], idx):
        c = classes[j]
        for nm in names:
            P = oof[nm]
            yy = np.tile((y == c).astype(int), P.shape[0])
            pp = P[:, :, j].ravel()
            fpr, tpr, _ = roc_curve(yy, pp)
            ax.plot(fpr, tpr, lw=1.6, label=f"{nm} (AUC {roc_auc_score(yy, pp):.2f})")
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set(title=f"ROC, class {c} vs rest (pooled out-of-fold)", xlabel="1 - specificity", ylabel="sensitivity")
        ax.legend(fontsize=7, loc="lower right")
    save_fig(fig, "roc")


def plot_confusions(cms, classes, names):
    n = len(names)
    cols = 3
    rows = ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, nm in zip(axes.ravel(), names):
        ax.axis("on")
        cm = cms[nm]
        rn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rn, vmin=0, vmax=1, cmap="Blues")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{cm[i, j]:.1f}\n({rn[i, j]:.0%})", ha="center", va="center",
                        color="white" if rn[i, j] > 0.55 else "black", fontsize=9)
        ax.set_xticks(range(len(classes)), [f"pred {c}" for c in classes])
        ax.set_yticks(range(len(classes)), [f"true {c}" for c in classes])
        ax.set_title(nm, fontsize=10)
    fig.suptitle("Confusion matrices (mean counts over repeats; % = share of the true class)")
    save_fig(fig, "confusion")


def plot_calibration(oof, y, classes, names):
    K = len(classes)
    idx = list(range(K)) if K > 2 else [1]
    fig, axes = plt.subplots(1, len(idx), figsize=(5.4 * len(idx), 5), squeeze=False)
    for ax, j in zip(axes[0], idx):
        c = classes[j]
        ax.plot([0, 1], [0, 1], "k:", label="perfect")
        for nm in names:
            P = oof[nm]
            yy = np.tile((y == c).astype(int), P.shape[0])
            pp = P[:, :, j].ravel()
            try:
                frac, mean = calibration_curve(yy, pp, n_bins=6, strategy="uniform")
                ax.plot(mean, frac, marker="o", ms=4, lw=1.3, label=nm)
            except ValueError:
                pass
        ax.set(title=f"Calibration, class {c} vs rest", xlabel="mean predicted probability",
               ylabel="observed fraction", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=7, loc="upper left")
    save_fig(fig, "calibration")


def plot_perm_null(null, observed, best):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null, bins=20, color="#9aa5b1")
    ax.axvline(observed, color="crimson", lw=2, label=f"observed {observed:.3f}")
    ax.axvline(np.percentile(null, 95), color="k", ls=":", label="95th percentile of null")
    ax.set(title=f"{best}: CV macro AUC with shuffled labels", xlabel="macro AUC", ylabel="count")
    ax.legend(fontsize=8)
    save_fig(fig, "permutation_null")


# ------------------------------------------------------------ STEP 8-9: evaluation of all models
def evaluate_all(d, target, title, full=True, drop_features=()):
    d = d.dropna(subset=[target]).reset_index(drop=True)
    X = d.drop(columns=[target] + [c for c in drop_features if c in d.columns])
    y = d[target].astype(int).values
    classes = np.sort(np.unique(y))
    K, n = len(classes), len(y)
    counts = pd.Series(y).value_counts().sort_index()
    banner(f"STEP 8: {title} - REPEATED STRATIFIED CV (preprocessing refitted per training fold; "
           f"{'SMOTE' if USE_SMOTE else 'class weighting, no SMOTE'})")
    print("Target distribution:\n" + counts.to_string())
    print(f"Predictors ({X.shape[1]}): {list(X.columns)}")
    n_splits = min(5, int(counts.min()))
    if n_splits < 2:
        print("Smallest class has <2 samples: cannot cross-validate.")
        return None
    print(f"{n_splits}-fold x {N_REPEATS} repeats = {n_splits * N_REPEATS} train/test splits per model.")

    models = get_models()
    names = list(models)
    oof, fold_rows = {}, []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for nm, proto in models.items():
            print(f"  running {nm} ...")
            oof[nm], rows = cv_oof(proto, X, y, classes, n_splits, N_REPEATS, USE_SMOTE, collect=full, name=nm)
            fold_rows += rows

    rep = {nm: pd.DataFrame([pooled_metrics(y, oof[nm][r], classes) for r in range(N_REPEATS)]) for nm in names}
    cand = [m for m in names if m != "No-skill baseline"]

    # ---- AUC with bootstrap CI (repeat-averaged out-of-fold probabilities)
    ci_rows = {}
    for nm in names:
        Pm = oof[nm].mean(axis=0)
        lo, hi = bootstrap_auc_ci(y, Pm, classes)
        ci_rows[nm] = {"macro AUC (mean ± SD over repeats)": ms(rep[nm].macro_auc),
                       "macro AUC of averaged probs": round(auc_macro(y, Pm, classes), 3),
                       "95% bootstrap CI": f"[{lo:.3f}, {hi:.3f}]",
                       "bal_acc": ms(rep[nm].bal_acc)}
    ci_tab = pd.DataFrame(ci_rows).T
    banner(f"{title}: discrimination with 95% bootstrap CI (no-skill baseline included as reference)")
    print(ci_tab.to_string())
    print("Note: with tiny classes the no-skill baseline can score slightly BELOW 0.5 in cross-validation; "
          "the label-permutation test (full run) is the correct comparison.")

    if not full:
        return {"ci": ci_tab}

    folds = pd.DataFrame(fold_rows)

    banner("TABLE 1: train vs test metrics for each fold (repeat 1)")
    print(folds[folds.repeat == 1].drop(columns="repeat").round(3).to_string(index=False))

    banner(f"TABLE 2: train vs test, mean ± SD over all {n_splits * N_REPEATS} folds (large train-test gap = overfitting)")
    t2 = folds.groupby("model", sort=False).agg(lambda s: ms(s))[
        ["train_bal_acc", "test_bal_acc", "train_auc", "test_auc", "test_f1", "test_log_loss"]]
    print(t2.to_string())

    cols = ["bal_acc", "macro_f1", "macro_auc", "brier", "log_loss", "ece"] + (["pr_auc"] if K == 2 else []) \
        + [f"recall_{c}" for c in classes] + ([f"auc_{c}" for c in classes] if K > 2 else [])
    t3 = pd.DataFrame({nm: {c: ms(rep[nm][c]) for c in cols} for nm in names}).T
    banner("TABLE 3: pooled out-of-fold performance (mean ± SD over repeats). brier/log_loss/ece: lower = better")
    print(t3.to_string())

    cms = {nm: sum(confusion_matrix(y, classes[oof[nm][r].argmax(1)], labels=classes)
                   for r in range(N_REPEATS)) / N_REPEATS for nm in names}
    print("\nTABLE 4: mean confusion matrices (rows = true, columns = predicted)")
    for nm in names:
        print(f"\n{nm}")
        print(pd.DataFrame(cms[nm], index=[f"true {c}" for c in classes],
                           columns=[f"pred {c}" for c in classes]).round(2).to_string())
    plot_roc(oof, y, classes, names)
    plot_confusions(cms, classes, names)
    plot_calibration(oof, y, classes, names)

    # ---- comparison and ranking
    banner("STEP 9: MODEL COMPARISON (no-skill baseline excluded from ranking)")
    gap = folds.groupby("model").apply(lambda g: g.train_auc.mean() - g.test_auc.mean(), include_groups=False)
    comp = pd.DataFrame({
        "bal_acc": {m: rep[m].bal_acc.mean() for m in cand},
        "macro_auc": {m: rep[m].macro_auc.mean() for m in cand},
        "macro_f1": {m: rep[m].macro_f1.mean() for m in cand},
        "brier": {m: rep[m].brier.mean() for m in cand},
        "log_loss": {m: rep[m].log_loss.mean() for m in cand},
        "train-test AUC gap": {m: gap[m] for m in cand}})
    ranks = pd.DataFrame({c: comp[c].rank(ascending=(c in ["brier", "log_loss", "train-test AUC gap"]))
                          for c in comp.columns})
    comp["mean rank (1 = best)"] = ranks.mean(axis=1)
    comp = comp.sort_values("mean rank (1 = best)")
    print(comp.round(3).to_string())
    best = comp.index[0]
    print(f"\nTop-ranked model: {best}")

    # ---- is there any signal at all?  (label-permutation test on the top-ranked model)
    banner(f"SIGNAL CHECK: label-permutation test for '{best}' ({N_LABEL_PERM} permutations x {PERM_REPEATS} CV repeats)")
    observed = float(np.mean(rep[best].macro_auc.values[:PERM_REPEATS]))
    p, null = label_permutation_test(models[best], X, y, classes, n_splits, USE_SMOTE, observed)
    p_adj = min(1.0, p * len(cand))
    print(f"observed CV macro AUC = {observed:.3f} | AUC with shuffled labels: mean {null.mean():.3f}, "
          f"95th percentile {np.percentile(null, 95):.3f}")
    print(f"permutation p = {p:.3f}; Bonferroni-adjusted for choosing among {len(cand)} models: p = {p_adj:.3f}")
    plot_perm_null(null, observed, best)
    signal = p_adj < 0.05
    if signal:
        print("VERDICT: the model performs better than chance under the same CV pipeline (adjusted p < 0.05). "
              "Still an internal estimate with very small classes: external validation is required.")
    else:
        print("VERDICT: NO RELIABLE SIGNAL. Performance is not distinguishable from what shuffled labels give. "
              "Do not interpret feature importance or report this model as predictive; report the group "
              "comparison (STEP 5) as exploratory and state the sample-size limitation.")
    print("CAUTION: 52 patients, 4-5 in class 3/5; per-class results are unstable.")

    with pd.ExcelWriter(outpath("cv_results.xlsx")) as xw:
        folds.to_excel(xw, sheet_name="per_fold", index=False)
        t2.to_excel(xw, sheet_name="train_vs_test_summary")
        t3.to_excel(xw, sheet_name="pooled_oof")
        ci_tab.to_excel(xw, sheet_name="auc_ci")
        comp.to_excel(xw, sheet_name="comparison")
        pd.DataFrame({"null_auc": null}).to_excel(xw, sheet_name="permutation_null", index=False)
    return dict(best=best, models=models, X=X, y=y, classes=classes, signal=signal, ci=ci_tab,
                ranking=list(comp.index))


# ------------------------------------------------------------ STEP 10: feature importance
def native_importance(clf, names):
    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=names)
    if hasattr(clf, "coef_"):
        return pd.Series(np.abs(clf.coef_).mean(axis=0), index=names)
    return None


def importance_best(res):
    best, X, y, classes = res["best"], res["X"], res["y"], res["classes"]
    proto = res["models"][best]
    banner(f"STEP 10: FEATURE IMPORTANCE of the top-ranked model: {best}")
    if not res["signal"]:
        print("WARNING: the signal check found no reliable signal; the importances below are NOT interpretable "
              "(they describe noise the model fitted). Shown only for completeness.")
    n_splits = min(5, int(pd.Series(y).value_counts().min()))
    nat, perm = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in range(N_IMP_REPEATS):
            for f, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=r).split(X, y), 1):
                pre = build_preprocessor(X)
                Xtr, Xte = pre.fit_transform(X.iloc[tr]), pre.transform(X.iloc[te])
                names = list(pre.get_feature_names_out())
                Xfit, yfit = smote_resample(Xtr, y[tr], seed=r * 100 + f) if USE_SMOTE else (Xtr, y[tr])
                clf = clone(proto).fit(Xfit, yfit)
                ni = native_importance(clf, names)
                if ni is not None:
                    nat.append(ni)
                base = log_loss(y[te], norm(clf.predict_proba(Xte)), labels=classes)
                rng = np.random.RandomState(r * 100 + f)
                imp = {}
                for j, nm in enumerate(names):
                    ds = []
                    for _ in range(N_PERM):
                        Xp = Xte.copy()
                        Xp[:, j] = rng.permutation(Xp[:, j])
                        ds.append(log_loss(y[te], norm(clf.predict_proba(Xp)), labels=classes) - base)
                    imp[nm] = np.mean(ds)
                perm.append(pd.Series(imp))
    P = pd.concat(perm, axis=1).fillna(0)
    out = pd.DataFrame({"permutation_importance_mean": P.mean(axis=1), "permutation_sd": P.std(axis=1)})
    if nat:
        Nn = pd.concat(nat, axis=1).fillna(0)
        out["native_importance_mean"] = Nn.mean(axis=1)
        out["native_sd"] = Nn.std(axis=1)
    out = out.sort_values("permutation_importance_mean", ascending=False)
    print("Permutation importance = increase in test-fold log-loss when the feature is shuffled; "
          "native = |coefficient| (logistic) or impurity importance (trees). Averaged over all test folds:")
    print(out.round(4).to_string())
    out.to_excel(outpath("feature_importance_best.xlsx"))

    top = out.head(15)[::-1]
    ncol = 2 if nat else 1
    fig, axes = plt.subplots(1, ncol, figsize=(6.5 * ncol, 6), squeeze=False)
    axes[0, 0].barh(top.index, top.permutation_importance_mean, xerr=top.permutation_sd, color="#3b7ddd")
    axes[0, 0].set_title(f"{best}: permutation importance\n(log-loss increase, mean ± SD over folds)"
                         + ("" if res["signal"] else "\nNO RELIABLE SIGNAL - not interpretable"))
    if nat:
        tn = out.sort_values("native_importance_mean", ascending=False).head(15)[::-1]
        axes[0, 1].barh(tn.index, tn.native_importance_mean, xerr=tn.native_sd, color="#2a9d8f")
        axes[0, 1].set_title(f"{best}: model-native importance")
    save_fig(fig, "feature_importance")


# ------------------------------------------------------------ STEP 11: SHAP explainability
SHAP_TREE_MODELS = ["XGBoost", "Random forest", "Extra trees"]     # models TreeExplainer supports here


def normalize_shap_values(shap_values, n_features, n_classes):
    """Return a list with one (n_samples, n_features) matrix per class, for old and new SHAP output formats."""
    if isinstance(shap_values, list):
        out = [np.asarray(v) for v in shap_values]
        if len(out) == n_classes:
            return out
        raise ValueError(f"Unexpected number of SHAP outputs: {len(out)}")
    v = np.asarray(shap_values)
    if v.ndim == 2:                                  # binary XGBoost: one matrix = positive class
        if v.shape[1] == n_features and n_classes == 2:
            return [np.zeros_like(v), v]
        raise ValueError(f"Unexpected 2-D SHAP shape {v.shape} for {n_classes} classes")
    if v.ndim == 3:
        if v.shape[1] == n_features and v.shape[2] == n_classes:      # samples x features x classes
            return [v[:, :, k] for k in range(n_classes)]
        if v.shape[1] == n_classes and v.shape[2] == n_features:      # samples x classes x features
            return [v[:, k, :] for k in range(n_classes)]
    raise ValueError(f"Unsupported SHAP output shape: {v.shape}")


def shap_importance_cv(res):
    """Fold-wise SHAP for the best-ranked TREE model (XGBoost / random forest / extra trees).
    SHAP values are computed only for held-out patients of each CV fold, then pooled over folds and repeats.
    Plots are coloured by the ORIGINAL (unscaled) feature values. All files go to BASE_DIR with the task prefix."""
    if not RUN_SHAP:
        print("SHAP analysis disabled (RUN_SHAP=False).")
        return
    if not SHAP_OK:
        print("SHAP unavailable. Install with: pip install shap")
        return
    if not res["signal"] and not SHAP_EVEN_IF_NO_SIGNAL:
        print("SHAP skipped: no reliable signal (set SHAP_EVEN_IF_NO_SIGNAL=True to run anyway).")
        return
    cand = [m for m in res["ranking"] if m in SHAP_TREE_MODELS and m in res["models"]]
    if not cand:
        print(f"SHAP skipped: none of the models {SHAP_TREE_MODELS} was run.")
        return
    for best in cand:                      # if SHAP fails for one model (e.g. XGBoost/SHAP version clash) try the next
        try:
            _shap_one(res, best)
            return
        except Exception as e:
            print(f"SHAP failed for {best}: {e}\n  -> trying the next tree model" if best != cand[-1]
                  else f"SHAP failed for {best}: {e}")
    print("SHAP analysis could not be completed for any tree model.")


def _shap_one(res, best):
    proto, X, y, classes = res["models"][best], res["X"], res["y"], res["classes"]
    n_classes = len(classes)
    cols = list(X.columns)
    n_splits = min(5, int(pd.Series(y).value_counts().min()))
    if n_splits < 2:
        print("SHAP skipped: too few patients in the smallest class.")
        return
    warn = "" if res["signal"] else "\nNO RELIABLE SIGNAL - not interpretable"
    banner(f"STEP 11: FOLD-WISE SHAP - {best} (top-ranked tree model)")
    if not res["signal"]:
        print("WARNING: the signal check found no reliable signal; SHAP only describes what the model fitted "
              "(mostly noise). Shown for completeness, not for interpretation.")
    print(f"{n_splits}-fold x {SHAP_REPEATS} repeats; {len(y)} patients; classes {list(classes)}; "
          "explanations are for held-out patients only.")

    shap_parts = [[] for _ in range(n_classes)]
    feat_parts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in range(SHAP_REPEATS):
            skf = StratifiedKFold(n_splits, shuffle=True, random_state=SHAP_SEED + r)
            for f, (tr, te) in enumerate(skf.split(X, y), 1):
                pre = build_preprocessor(X)
                Xtr, Xte = pre.fit_transform(X.iloc[tr]), pre.transform(X.iloc[te])
                names = list(pre.get_feature_names_out())
                # The missing-indicator column exists only in some folds -> keep the original features only,
                # so every fold has the same columns and can be stacked.
                keep = [names.index(c) for c in cols]
                if USE_SMOTE:
                    Xfit, yfit = smote_resample(Xtr, y[tr], seed=SHAP_SEED + r * 100 + f)
                else:
                    Xfit, yfit = Xtr, y[tr]
                clf = clone(proto).fit(Xfit, yfit)
                tree_model = clf.base_ if hasattr(clf, "base_") else clf
                try:
                    values = shap.TreeExplainer(tree_model).shap_values(Xte, check_additivity=False)
                    fold = normalize_shap_values(values, len(names), n_classes)
                except Exception as e:
                    raise RuntimeError(f"SHAP failed at repeat {r + 1}, fold {f}: {e}") from e
                for k in range(n_classes):
                    shap_parts[k].append(fold[k][:, keep])
                feat_parts.append(X.iloc[te][cols].reset_index(drop=True))
            print(f"  SHAP repeat {r + 1}/{SHAP_REPEATS} done")

    X_all = pd.concat(feat_parts, ignore_index=True)
    tables = []
    for k, cval in enumerate(classes):
        if n_classes == 2 and k == 0:        # binary: SHAP explains the positive class only
            continue
        S = np.vstack(shap_parts[k])
        label = f"class_{cval}"
        plt.figure(figsize=(10, 7))
        shap.summary_plot(S, X_all, plot_type="dot", max_display=SHAP_MAX_FEATURES, show=False)
        plt.title(f"SHAP summary - {best} - {label}{warn}")
        plt.tight_layout()
        path = outpath(f"shap_beeswarm_{label}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[figure saved] {path}")

        imp = pd.DataFrame({"class": cval, "feature": cols,
                            "mean_abs_shap": np.abs(S).mean(axis=0),
                            "sd_abs_shap": np.abs(S).std(axis=0)}).sort_values("mean_abs_shap", ascending=False)
        tables.append(imp)
        top = imp.head(SHAP_MAX_FEATURES).sort_values("mean_abs_shap")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top["feature"], top["mean_abs_shap"], xerr=top["sd_abs_shap"], capsize=3, color="#e76f51")
        ax.set_xlabel("Mean |SHAP| (held-out patients)")
        ax.set_title(f"SHAP importance - {best} - {label}{warn}")
        save_fig(fig, f"shap_bar_{label}")
    if tables:
        tab = pd.concat(tables, ignore_index=True)
        print("\nSHAP feature importance (mean |SHAP| over held-out patients, all folds and repeats):")
        print(tab.round(4).to_string(index=False))
        xl = outpath("shap_importance.xlsx")
        tab.to_excel(xl, index=False)
        print(f"[excel saved] {xl}")


# ------------------------------------------------------------ main
def run(target):
    global PREFIX
    path = os.path.join(BASE_DIR, DATA_FILE)
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")
    print(f"python {platform.python_version()} | pandas {pd.__version__} | numpy {np.__version__} | scikit-learn {sklearn.__version__}")
    print(f"input : {path}\noutput: {BASE_DIR}")
    print(f"settings: DRUG_NAN_AS_ZERO={DRUG_NAN_AS_ZERO}, MISSING_DROP_THRESHOLD={MISSING_DROP_THRESHOLD}, "
          f"MIN_CLASS_COUNT={MIN_CLASS_COUNT}, PROTEIN_THRESHOLD_MG={PROTEIN_THRESHOLD_MG}, USE_SMOTE={USE_SMOTE}, "
          f"N_REPEATS={N_REPEATS}, N_LABEL_PERM={N_LABEL_PERM}, TASKS={[t[0] for t in TASKS]}, target={target}")

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
    stats_table(d, "STEP 2: after feature engineering (ALL candidate features, leakage not yet removed)")

    d_stat = d.dropna(subset=[t]).reset_index(drop=True)
    group_comparison(d_stat, t, feature_status(d_stat, t))                 # STEP 5, before any column is dropped

    print("\n--- column filtering (missing values / rare features) ---")
    d = drop_high_missing(d, protect=[t])
    d = drop_near_constant(d, protect=[t])
    stats_table(d, "STEP 3: after dropping high-missing and rare columns (NaN kept, imputed inside CV folds)")

    print("\n--- simple imputation (descriptive copy) ---")
    d_imp = impute_all(d, protect=[t])
    stats_table(d_imp, "STEP 4: imputed copy (median/mode), descriptive use only")
    d.to_excel(os.path.join(BASE_DIR, "cleaned_data.xlsx"), index=False)
    d_imp.to_excel(os.path.join(BASE_DIR, "cleaned_data_imputed.xlsx"), index=False)

    d = d.dropna(subset=[t]).reset_index(drop=True)
    d_ml = remove_leakage(d, t)            # STEP 6 (models never see leakage features)
    export_ml_ready(d_ml, t)               # STEP 7

    summary = []
    for task, positive in TASKS:
        d_task = d_ml.copy()
        if positive is not None:
            d_task[t] = d_task[t].isin(positive).astype(int)
        PREFIX = f"{t}_{task}"
        res = evaluate_all(d_task, t, f"TASK {task}", full=True)
        if not res:
            continue
        importance_best(res)
        shap_importance_cv(res)               # STEP 11
        summary.append((task, res["best"], res["signal"]))
        if RUN_SENSITIVITY:
            PREFIX = f"{t}_{task}_no_ordering_proxies"
            print(f"\n--- sensitivity: {task} without ordering-proxy features {ORDERING_PROXY_FEATURES} ---")
            evaluate_all(d_task, t, f"SENSITIVITY {task} (no ordering proxies)", full=False,
                         drop_features=ORDERING_PROXY_FEATURES)

    banner("FINAL SUMMARY")
    for task, best, signal in summary:
        print(f"  {task}: top model = {best} | reliable signal vs shuffled labels: {'YES' if signal else 'NO'}")
    print("\nDONE. Files saved in", BASE_DIR)


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    log_path = os.path.join(BASE_DIR, "preprocess_log.txt")
    logf = open(log_path, "w", encoding="utf-8", buffering=1)
    try:
        sys.__stdout__.reconfigure(encoding="utf-8", errors="replace")
        sys.__stderr__.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.stdout = Tee(sys.__stdout__, logf)
    sys.stderr = Tee(sys.__stderr__, logf)      # warnings and errors are logged too
    try:
        run(sys.argv[1] if len(sys.argv) > 1 else TARGET)
    except SystemExit as e:
        print(e)
    except Exception:
        traceback.print_exc(file=sys.stdout)
    finally:
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        logf.close()
        print("Log saved to", log_path)


if __name__ == "__main__":
    main()