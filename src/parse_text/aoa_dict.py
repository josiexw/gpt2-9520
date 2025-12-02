import os
import glob
import argparse
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def logistic4(x, L, k, x0, b):
    return L / (1.0 + np.exp(-k * (x - x0))) + b


def compute_llm_aoa_steps(
    word_to_series: Dict[str, List[float]],
    steps: List[int],
    baseline_bits: float,
    words: List[str],
) -> Dict[str, float]:
    aoa_log10: Dict[str, float] = {}
    if not words:
        return aoa_log10

    step_arr = np.array(steps, dtype=float)
    log_steps = np.log10(step_arr + 1.0)
    positive_mask = step_arr > 0
    log_steps[positive_mask] = np.log10(step_arr[positive_mask])

    for w in words:
        s = np.array(word_to_series[w], dtype=float)
        mask = np.isfinite(log_steps) & np.isfinite(s)
        x = log_steps[mask]
        y = s[mask]
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        if x.size < 2:
            continue

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


def load_child_aoa_from_wordbank(csv_path: str) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    cols = ["16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"]
    aoa_cols = [c for c in cols if c in df.columns]
    aoa_cols_sorted = sorted(aoa_cols, key=lambda x: int(x))
    word_to_aoa: Dict[str, float] = {}
    for _, row in df.iterrows():
        word = str(row["item_definition"]).strip().lower()
        if not word:
            continue
        aoa = np.nan
        for col in aoa_cols_sorted:
            try:
                v = float(row[col])
            except (TypeError, ValueError):
                continue
            if v >= 0.5:
                aoa = float(col)
                break
        if not np.isnan(aoa):
            word_to_aoa[word] = aoa
    return word_to_aoa


def build_word_series_from_results(results_dir: str, value_key: str = "avg_surprisal_per_token"):
    pattern = os.path.join(results_dir, "results_ckpt_idx*.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        return {}, []

    common_words = None
    for path in files:
        with open(path, "rb") as f:
            output_dict = pickle.load(f)
        words_here = set(output_dict.keys())
        if common_words is None:
            common_words = words_here
        else:
            common_words &= words_here

    if not common_words:
        return {}, []

    words = sorted(common_words)
    word_to_series: Dict[str, List[float]] = {w: [] for w in words}

    for path in files:
        with open(path, "rb") as f:
            output_dict = pickle.load(f)
        for w in words:
            d = output_dict[w]
            v = d.get(value_key, float("nan"))
            word_to_series[w].append(float(v))

    steps = list(range(1, len(files) + 1))
    return word_to_series, steps


def main():
    parser = argparse.ArgumentParser(description="Generate child and GPT-2 AoA dictionaries from Wordbank and mult_ckpt outputs.")
    parser.add_argument("--wordbank_csv_path", type=str, default="data/wordbank_item_data.csv")
    parser.add_argument("--child_aoa_out_path", type=str, default="data/child_aoa.pkl")
    parser.add_argument("--results_small_dir", type=str, required=False)
    parser.add_argument("--results_medium_dir", type=str, required=False)
    parser.add_argument("--gpt2_small_aoa_out_path", type=str, default="data/gpt2_small_aoa.pkl")
    parser.add_argument("--gpt2_medium_aoa_out_path", type=str, default="data/gpt2_medium_aoa.pkl")
    parser.add_argument("--baseline_bits_small", type=float, default=14.9)
    parser.add_argument("--baseline_bits_medium", type=float, default=14.9)
    parser.add_argument("--value_key", type=str, default="avg_surprisal_per_token")
    args = parser.parse_args()

    child_aoa_dict = load_child_aoa_from_wordbank(args.wordbank_csv_path)
    with open(args.child_aoa_out_path, "wb") as f:
        pickle.dump(child_aoa_dict, f)

    if args.results_small_dir:
        word_to_series_small, steps_small = build_word_series_from_results(args.results_small_dir, args.value_key)
        if steps_small:
            words_small = list(word_to_series_small.keys())
            gpt2_small_aoa_dict = compute_llm_aoa_steps(
                word_to_series=word_to_series_small,
                steps=steps_small,
                baseline_bits=args.baseline_bits_small,
                words=words_small,
            )
            with open(args.gpt2_small_aoa_out_path, "wb") as f:
                pickle.dump(gpt2_small_aoa_dict, f)

    if args.results_medium_dir:
        word_to_series_medium, steps_medium = build_word_series_from_results(args.results_medium_dir, args.value_key)
        if steps_medium:
            words_medium = list(word_to_series_medium.keys())
            gpt2_medium_aoa_dict = compute_llm_aoa_steps(
                word_to_series=word_to_series_medium,
                steps=steps_medium,
                baseline_bits=args.baseline_bits_medium,
                words=words_medium,
            )
            with open(args.gpt2_medium_aoa_out_path, "wb") as f:
                pickle.dump(gpt2_medium_aoa_dict, f)


if __name__ == "__main__":
    main()
