import os
import re
import math
import statistics
import numpy as np
import pickle
from typing import Dict, List, Set
from scipy.optimize import curve_fit
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit


# Plots
def normalize_cdi_label(w: str) -> str:
    w = w.lower().strip()
    w = re.sub(r"\s*\(.*?\)\s*", "", w)
    w = re.sub(r"\s+", " ", w)
    return w


def load_wordbank_aoa(csv_path: str):
    df = pd.read_csv(csv_path)
    aoa_cols = [c for c in df.columns if c.isdigit()]
    aoa_cols_sorted = sorted(aoa_cols, key=lambda x: int(x))
    word_to_aoa = {}
    for _, row in df.iterrows():
        word = normalize_cdi_label(str(row["item_definition"]))
        if not word:
            continue
        aoa = math.nan
        for col in aoa_cols_sorted:
            try:
                v = float(row[col])
            except (TypeError, ValueError):
                continue
            if v >= 0.5:
                aoa = float(col)
                break
        if not math.isnan(aoa):
            if word in word_to_aoa:
                word_to_aoa[word] = statistics.mean([word_to_aoa[word], aoa])
            else:
                word_to_aoa[word] = aoa
    return word_to_aoa


def load_wordbank_curves(csv_path: str):
    df = pd.read_csv(csv_path)
    aoa_cols = [c for c in df.columns if c.isdigit()]
    aoa_cols_sorted = sorted(aoa_cols, key=lambda x: int(x))
    months = [int(c) for c in aoa_cols_sorted]
    word_to_curve: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        word = normalize_cdi_label(str(row["item_definition"]))
        if not word:
            continue
        vals = []
        for col in aoa_cols_sorted:
            v = row[col]
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(math.nan)
        vals_arr = np.array(vals, dtype=float)
        if word in word_to_curve:
            prev = word_to_curve[word]
            stacked = np.vstack([prev, vals_arr])
            word_to_curve[word] = np.nanmean(stacked, axis=0)
        else:
            word_to_curve[word] = vals_arr
    return months, word_to_curve


def parse_step_from_fname(fname: str):
    m_label = re.search(r"idx\d+_([A-Za-z]+)(\d+)\.pkl$", fname)
    if not m_label:
        raise ValueError(f"Cannot parse label from {fname}")
    step = int(m_label.group(2))
    return step


def load_results_dir(results_dir: str):
    entries = []
    for fname in os.listdir(results_dir):
        if not fname.endswith(".pkl"):
            continue
        step = parse_step_from_fname(fname)
        path = os.path.join(results_dir, fname)
        with open(path, "rb") as f:
            data = pickle.load(f)
        word_surprisal = {}
        word_act = {}
        for w, info in data.items():
            w_norm = normalize_cdi_label(w)
            word_surprisal[w_norm] = float(info["avg_surprisal"])
            layer_attn = info.get("avg_layer_attn", None)
            if layer_attn is not None:
                layer_arr = np.array(layer_attn, dtype=float)
                word_act[w_norm] = float(np.nanmean(layer_arr))
        entries.append((step, word_surprisal, word_act))
    entries.sort(key=lambda x: x[0])
    steps = [e[0] for e in entries]
    all_words_surprisal: Set[str] = set()
    all_words_act: Set[str] = set()
    for _, ws, wa in entries:
        all_words_surprisal.update(ws.keys())
        all_words_act.update(wa.keys())
    all_words = all_words_surprisal | all_words_act
    word_to_surprisal: Dict[str, List[float]] = {}
    word_to_act: Dict[str, List[float]] = {}
    for w in all_words:
        s_series = []
        a_series = []
        for _, ws, wa in entries:
            s_series.append(float(ws.get(w, math.nan)))
            a_series.append(float(wa.get(w, math.nan)))
        word_to_surprisal[w] = s_series
        word_to_act[w] = a_series
    return steps, word_to_surprisal, word_to_act


def get_simple_ranking(word_aoa: Dict[str, float], available_words: set[str], max_n: int):
    items = [(w, aoa) for w, aoa in word_aoa.items() if w in available_words and not math.isnan(aoa)]
    items.sort(key=lambda x: (x[1], x[0]))
    return [w for w, _ in items[:max_n]]


def compute_avg_series(word_to_series: Dict[str, List[float]], words: List[str]):
    if not words:
        return np.array([], dtype=float)
    arr = []
    for w in words:
        arr.append(np.array(word_to_series[w], dtype=float))
    arr = np.stack(arr, axis=0)
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr, axis=0)


def compute_thresholds_per_word(
    word_to_series: Dict[str, List[float]],
    baseline_bits: float,
    words: List[str],
    steps: List[int],
):
    aoa_log10 = compute_llm_aoa_steps(
        word_to_series=word_to_series,
        steps=steps,
        baseline_bits=baseline_bits,
        words=words,
    )
    aoa_steps = {}
    for w, x_star in aoa_log10.items():
        aoa_steps[w] = float(10.0 ** x_star)
    return aoa_steps


def logistic4(x, L, k, x0, b):
    x = np.asarray(x, dtype=float)
    z = k * (x - x0)
    z = np.clip(z, -60, 60)
    return L / (1.0 + np.exp(z)) + b


def compute_llm_aoa_steps(
    word_to_series: Dict[str, List[float]],
    steps: List[int],
    baseline_bits: float,
    words: List[str],
):
    aoa_log10: Dict[str, float] = {}
    if not words:
        return aoa_log10

    step_arr = np.array(steps, dtype=float)
    safe_steps = np.where(step_arr > 0, step_arr, 1.0)
    log_steps = np.log10(safe_steps)
    mask_log = np.isfinite(log_steps)

    for w in words:
        s = np.array(word_to_series[w], dtype=float)
        mask = mask_log & np.isfinite(s)
        x = log_steps[mask]
        y = s[mask]
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max == y_min:
            continue

        thr = 0.5 * (baseline_bits + y_min)

        L0 = max(y_max - y_min, 1e-3)
        b0 = y_min
        x0_0 = float(np.median(x))
        k0 = 1.0

        x_star = None

        try:
            popt, _ = curve_fit(
                logistic4,
                x,
                y,
                p0=[L0, k0, x0_0, b0],
                maxfev=10000,
            )
            L, k, x0, b = popt

            if L * k == 0:
                raise RuntimeError("flat logistic fit")

            f_lo = float(logistic4(x[0], *popt))
            f_hi = float(logistic4(x[-1], *popt))

            lo_val = min(f_lo, f_hi)
            hi_val = max(f_lo, f_hi)

            if not (lo_val <= thr <= hi_val):
                raise RuntimeError("threshold outside fit range")

            lo_x = x[0]
            hi_x = x[-1]

            if f_lo <= f_hi:
                for _ in range(60):
                    mid_x = 0.5 * (lo_x + hi_x)
                    val = float(logistic4(mid_x, *popt))
                    if val < thr:
                        lo_x = mid_x
                    else:
                        hi_x = mid_x
            else:
                for _ in range(60):
                    mid_x = 0.5 * (lo_x + hi_x)
                    val = float(logistic4(mid_x, *popt))
                    if val > thr:
                        lo_x = mid_x
                    else:
                        hi_x = mid_x

            x_star = 0.5 * (lo_x + hi_x)

        except Exception:
            idx_val = None

            for j in range(1, len(y)):
                y_prev, y_curr = y[j - 1], y[j]
                if not (np.isfinite(y_prev) and np.isfinite(y_curr)):
                    continue

                if (y_prev - thr) * (y_curr - thr) <= 0:
                    idx_val = j
                    break

            if idx_val is None:
                idx_val = int(np.argmin(np.abs(y - thr)))
            x_star = float(x[idx_val])

        if x_star is not None and np.isfinite(x_star):
            aoa_log10[w] = x_star

    return aoa_log10


def normalize_x(xs: np.ndarray):
    xs = np.array(xs, dtype=float)
    if xs.size == 0:
        return xs
    denom = xs[-1] - xs[0]
    if denom <= 0:
        return np.zeros_like(xs)
    return (xs - xs[0]) / denom


def compute_child_interp_aoa(wordbank_csv: str):
    months, word_to_curve = load_wordbank_curves(wordbank_csv)
    months_arr = np.array(months, dtype=float)

    child_interp_aoa = {}
    for w, curve in word_to_curve.items():
        curve_arr = np.array(curve, dtype=float)
        if not np.isfinite(curve_arr).any():
            continue

        mask = np.isfinite(curve_arr)
        m = months_arr[mask]
        y = curve_arr[mask]
        if m.size < 2:
            continue

        idx_cross = None
        for j in range(1, y.size):
            y_prev = y[j - 1]
            y_curr = y[j]
            if not (np.isfinite(y_prev) and np.isfinite(y_curr)):
                continue
            if (y_prev - 0.5) * (y_curr - 0.5) <= 0:
                idx_cross = j
                break

        if idx_cross is None:
            continue

        m1, m2 = m[idx_cross - 1], m[idx_cross]
        y1, y2 = y[idx_cross - 1], y[idx_cross]
        if not (np.isfinite(y1) and np.isfinite(y2)) or m2 == m1:
            continue

        if y2 == y1:
            aoa_month = m2
        else:
            aoa_month = m1 + (0.5 - y1) * (m2 - m1) / (y2 - y1)

        if np.isfinite(aoa_month) and aoa_month > 0:
            child_interp_aoa[w] = float(aoa_month)

    return child_interp_aoa


def compute_log_freq(freq_dict):
    return {w: np.log10(freq) for w, freq in freq_dict.items() if freq > 0}


# Regression
def bootstrap_correlation(x_arr, y_arr, n_bootstrap=10000):
    x_arr = np.asarray(x_arr, dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)

    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]

    r_obs, _ = pearsonr(x_arr, y_arr)

    y_mean = np.mean(y_arr)
    resid = y_arr - y_mean

    rng = np.random.default_rng(42)
    n = x_arr.shape[0]
    r_null = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        res_boot = resid[rng.integers(0, n, size=n)]
        y_boot = y_mean + res_boot
        r_null[i], _ = pearsonr(x_arr, y_boot)

    pval = np.mean(np.abs(r_null) >= np.abs(r_obs))
    return r_obs, pval, r_null


def fit_regression_xy(x_arr, y_arr):
    x_arr = np.asarray(x_arr, dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 2:
        return None, None, None, None

    X = x_arr.reshape(-1, 1)
    model = LinearRegression().fit(X, y_arr)
    
    # Use bootstrap for p-value
    r, pval, _ = bootstrap_correlation(x_arr, y_arr)
    
    return model, r, pval


# Correlation
def fit_logistic_normalized(x_raw: np.ndarray, y_raw: np.ndarray):
    x_raw = np.asarray(x_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    mask = np.isfinite(x_raw) & np.isfinite(y_raw)
    x_raw = x_raw[mask]
    y_raw = y_raw[mask]
    if x_raw.size < 4:
        return None
    x = normalize_x(x_raw)
    y_min = float(np.min(y_raw))
    y_max = float(np.max(y_raw))
    if not (np.isfinite(y_min) and np.isfinite(y_max)) or y_max <= y_min:
        return None
    L0 = y_max - y_min
    b0 = y_min
    x0_0 = 0.5
    if x.size > 1:
        corr = np.corrcoef(x, y_raw)[0, 1]
    else:
        corr = 0.0
    k0 = 1.0 if corr < 0 else -1.0
    p0 = [L0, k0, x0_0, b0]
    try:
        popt, _ = curve_fit(logistic4, x, y_raw, p0=p0, maxfev=10000)
    except Exception:
        return None
    return popt


def zscore(vec: np.ndarray):
    v = np.asarray(vec, dtype=float)
    mask = np.isfinite(v)
    if np.sum(mask) < 2:
        return v
    m = np.mean(v[mask])
    s = np.std(v[mask])
    if s > 0:
        v[mask] = (v[mask] - m) / s
    else:
        v[mask] = 0.0
    return v