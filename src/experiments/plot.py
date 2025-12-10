import os
import re
import argparse
import math
import statistics
import pickle
from typing import Dict, List, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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


def normalize_x_aligned(xs, align_idx, child_aoa_x):
    xs = np.array(xs, dtype=float)
    if xs.size == 0 or align_idx >= len(xs) or align_idx < 0:
        return normalize_x(xs)
    
    x_align = xs[align_idx]
    x_start = xs[0]
    x_end = xs[-1]
    xs_norm = (xs - x_start) / (x_end - x_start)
    align_norm = (x_align - x_start) / (x_end - x_start)

    if align_norm == 0 or not np.isfinite(child_aoa_x) or child_aoa_x == 0:
        return xs_norm

    xs_aligned = xs_norm * (child_aoa_x / align_norm)
    
    return xs_aligned


def compute_threshold_crossing_idx(s, baseline_bits):
    s = np.array(s, dtype=float)
    if not np.isfinite(s).any():
        return None
    s_min = float(np.nanmin(s))
    thr = 0.5 * (baseline_bits + s_min)
    
    for j, v in enumerate(s):
        if np.isfinite(v) and v <= thr:
            return j
    return len(s) - 1


def crop_with_threshold(s, steps_arr, baseline_bits, margin_idx):
    s = np.array(s, dtype=float)
    if not np.isfinite(s).any():
        return None, None, None, None
    s_min = float(np.nanmin(s))
    thr = 0.5 * (baseline_bits + s_min)
    idx_cross = None
    for j, v in enumerate(s):
        if np.isfinite(v) and v <= thr:
            idx_cross = j
            break
    if idx_cross is None:
        idx_cross = len(s) - 1
    end_idx = min(idx_cross + margin_idx, len(s) - 1)
    s_crop = s[: end_idx + 1]
    steps_crop = steps_arr[: end_idx + 1]
    return s_crop, steps_crop, thr, idx_cross


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_dir", type=str, default="stanford-gpt2-small-a_results")
    parser.add_argument("--medium_dir", type=str, default="stanford-gpt2-medium-a_results")
    parser.add_argument("--wordbank_csv", type=str, default="data/wordbank_item_data.csv")
    parser.add_argument("--out_dir", type=str, default="figs")
    parser.add_argument("--max_simple", type=int, default=500)
    parser.add_argument("--ks", type=str, default="10,100,500")
    parser.add_argument("--baseline_bits", type=float, default=15.6)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    word_aoa = load_wordbank_aoa(args.wordbank_csv)
    months, word_to_curve = load_wordbank_curves(args.wordbank_csv)
    child_interp_aoa = compute_child_interp_aoa(args.wordbank_csv)

    steps_small, small_surpr, _ = load_results_dir(args.small_dir)
    steps_medium, medium_surpr, _ = load_results_dir(args.medium_dir)

    words_small = set(small_surpr.keys())
    words_medium = set(medium_surpr.keys())
    available_words = words_small & words_medium

    simple_ranking = get_simple_ranking(word_aoa, available_words, args.max_simple)

    if simple_ranking:
        margin_idx = 3
        months_arr = np.array(months, dtype=float)
        if months_arr.size > 1:
            months_norm = (months_arr - months_arr.min()) / (months_arr.max() - months_arr.min())
        else:
            months_norm = np.zeros_like(months_arr)

        for w in simple_ranking[150:]:
            if w not in word_to_curve:
                continue
            child_curve = word_to_curve[w]
            if not np.isfinite(child_curve).any():
                continue
            s_small = np.array(small_surpr[w], dtype=float)
            s_medium = np.array(medium_surpr[w], dtype=float)
            if not (np.isfinite(s_small).any() or np.isfinite(s_medium).any()):
                continue

            steps_small_arr = np.array(steps_small, dtype=float)
            steps_medium_arr = np.array(steps_medium, dtype=float)

            s_small_crop, steps_small_crop, thr_small, idx_cross_small = crop_with_threshold(
                s_small, steps_small_arr, args.baseline_bits, margin_idx
            )
            s_medium_crop, steps_medium_crop, thr_medium, idx_cross_medium = crop_with_threshold(
                s_medium, steps_medium_arr, args.baseline_bits, margin_idx
            )

            if s_small_crop is None and s_medium_crop is None:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(12, 3))

            if s_small_crop is not None and len(s_small_crop) > 0:
                axes[0].plot(steps_small_crop, s_small_crop, label="gpt2-small")
                if thr_small is not None:
                    axes[0].axhline(thr_small, linestyle="--", linewidth=1, color="cornflowerblue")
            if s_medium_crop is not None and len(s_medium_crop) > 0:
                axes[0].plot(steps_medium_crop, s_medium_crop, label="gpt2-medium")
                if thr_medium is not None:
                    axes[0].axhline(thr_medium, linestyle="--", linewidth=1, color="orange")

            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Surprisal (bits)")
            axes[0].set_title(f"{w} - LLM surprisal")
            axes[0].legend()
            axes[0].invert_yaxis()

            axes[1].plot(months, child_curve, marker="o")
            axes[1].axhline(0.5, linestyle="--", linewidth=1, color="cornflowerblue")
            axes[1].set_xlabel("Age (months)")
            axes[1].set_ylabel("Proportion producing")
            axes[1].set_ylim(0.0, 1.0)
            axes[1].set_title(f"{w} - Children")

            ax3 = axes[2]
            ax3b = ax3.twinx()

            # Align AoA
            child_aoa_x = None
            if w in child_interp_aoa:
                child_aoa_month = child_interp_aoa[w]
                if months_arr.size > 1:
                    child_aoa_x = (child_aoa_month - months_arr.min()) / (months_arr.max() - months_arr.min())
                    child_aoa_x = np.clip(child_aoa_x, 0.0, 1.0)
            else:
                child_mask = np.isfinite(child_curve)
                if child_mask.any():
                    first_valid_val = child_curve[child_mask][0]
                    if first_valid_val >= 0.5:
                        child_aoa_x = 0.0
                    else:
                        plt.close(fig)
                        continue

            if child_aoa_x is None:
                plt.close(fig)
                continue

            child_mask = np.isfinite(child_curve)
            if child_mask.any():
                ax3b.plot(months_norm[child_mask], child_curve[child_mask], marker="o", color="green", label="children")

            # Normalize surprisal curves so threshold is at 0.5
            if thr_small is not None and np.isfinite(s_small).any():
                s_small_min = np.nanmin(s_small)
                s_small_range = np.nanmax(s_small) - s_small_min
                if s_small_range > 0:
                    s_small_norm = (s_small - thr_small) / s_small_range + 0.5
                else:
                    s_small_norm = np.full_like(s_small, 0.5)
            else:
                s_small_norm = None

            if thr_medium is not None and np.isfinite(s_medium).any():
                s_medium_min = np.nanmin(s_medium)
                s_medium_range = np.nanmax(s_medium) - s_medium_min
                if s_medium_range > 0:
                    s_medium_norm = (s_medium - thr_medium) / s_medium_range + 0.5
                else:
                    s_medium_norm = np.full_like(s_medium, 0.5)
            else:
                s_medium_norm = None

            # Plot normalized surprisal
            x_small_norm = normalize_x_aligned(steps_small_arr, idx_cross_small, child_aoa_x)
            ax3.plot(x_small_norm, s_small_norm, label="gpt2-small")
            x_medium_norm = normalize_x_aligned(steps_medium_arr, idx_cross_medium, child_aoa_x)
            ax3.plot(x_medium_norm, s_medium_norm, label="gpt2-medium")
            ax3.set_ylim(1.0, 0.0)

            ax3.set_xlim(0.0, 1.0)
            ax3.set_xlabel("Normalized timeline")
            ax3.set_ylabel("Surprisal (bits)")
            ax3b.set_ylabel("Proportion producing")
            ax3b.set_ylim(0.0, 1.0)
            ax3.set_title(f"{w} - Normalized aligned overlay")

            handles1, labels1 = ax3.get_legend_handles_labels()
            handles2, labels2 = ax3b.get_legend_handles_labels()
            if handles1 or handles2:
                ax3.legend(handles1 + handles2, labels1 + labels2, loc="best")

            fig.tight_layout()
            safe_w = re.sub(r"[^A-Za-z0-9]+", "_", w).strip("_")
            out_path_word = os.path.join(args.out_dir, f"word_{safe_w}_child_vs_llm_surprisal.png")
            fig.savefig(out_path_word, bbox_inches="tight")
            plt.close(fig)


            # MSE
            mse_results = []
                
            # Create interpolation functions and evaluate at 100 points
            x_eval = np.linspace(0, 1, 100)
            
            # Interpolate small model
            if s_small_norm is not None:
                x_small_norm_interp = normalize_x_aligned(steps_small_arr, idx_cross_small, child_aoa_x)
                mask_small = np.isfinite(s_small_norm) & np.isfinite(x_small_norm_interp)
                if mask_small.sum() >= 2:
                    s_small_interp = np.interp(x_eval, x_small_norm_interp[mask_small], s_small_norm[mask_small])
                else:
                    s_small_interp = None
            else:
                s_small_interp = None
            
            # Interpolate medium model
            if s_medium_norm is not None:
                x_medium_norm_interp = normalize_x_aligned(steps_medium_arr, idx_cross_medium, child_aoa_x)
                mask_medium = np.isfinite(s_medium_norm) & np.isfinite(x_medium_norm_interp)
                if mask_medium.sum() >= 2:
                    s_medium_interp = np.interp(x_eval, x_medium_norm_interp[mask_medium], s_medium_norm[mask_medium])
                else:
                    s_medium_interp = None
            else:
                s_medium_interp = None
            
            # Interpolate child curve
            if child_mask.sum() >= 2:
                child_interp = np.interp(x_eval, months_norm[child_mask], child_curve[child_mask])
            else:
                child_interp = None
            
            # Calculate MSEs
            mse_small_child = None
            mse_medium_child = None
            mse_small_medium = None

            # Invert surprisal so it increases
            if s_small_interp is not None and child_interp is not None:
                s_small_interp_inverted = 1.0 - s_small_interp
                mse_small_child = np.mean((s_small_interp_inverted - child_interp) ** 2)

            if s_medium_interp is not None and child_interp is not None:
                s_medium_interp_inverted = 1.0 - s_medium_interp
                mse_medium_child = np.mean((s_medium_interp_inverted - child_interp) ** 2)

            if s_small_interp is not None and s_medium_interp is not None:
                mse_small_medium = np.mean((s_small_interp - s_medium_interp) ** 2)

            mse_results.append({
                'word': w,
                'mse_small_child': mse_small_child,
                'mse_medium_child': mse_medium_child,
                'mse_small_medium': mse_small_medium
            })

            # Save MSE csv
            if mse_results:
                mse_df = pd.DataFrame(mse_results)
                mse_csv_path = os.path.join(args.out_dir, "mse_results.csv")
                if os.path.exists(mse_csv_path):
                    existing_df = pd.read_csv(mse_csv_path)

                    if not existing_df.empty and not mse_df.empty:
                        mse_df = pd.concat([existing_df, mse_df], ignore_index=True)
                    elif not existing_df.empty:
                        mse_df = existing_df
                mse_df.to_csv(mse_csv_path, index=False)


if __name__ == "__main__":
    main()
