import argparse
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
from plot import *

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="stanford-gpt2-small-a_results")
    parser.add_argument("--wordbank_csv", type=str, default="data/wordbank_item_data.csv")
    parser.add_argument("--max_simple", type=int, default=600)
    parser.add_argument("--corr_type", type=str, choices=["pearson", "spearman"], default="pearson")
    args = parser.parse_args()

    word_aoa = load_wordbank_aoa(args.wordbank_csv)
    months, word_to_curve = load_wordbank_curves(args.wordbank_csv)
    months_arr = np.array(months, dtype=float)

    steps_small, small_surpr, small_act = load_results_dir(args.model_dir)
    steps_small_arr = np.array(steps_small, dtype=float)

    available_words = set(small_surpr.keys()) & set(small_act.keys()) & set(word_to_curve.keys())
    simple_ranking = get_simple_ranking(word_aoa, available_words, args.max_simple)

    x_shared = np.linspace(0.0, 1.0, 200)

    rows = []

    print(
        "word\t"
        "r_child_vs_surprisal\tp_child_vs_surprisal\t"
        "r_child_vs_attention\tp_child_vs_attention\t"
        "r_surprisal_vs_attention\tp_surprisal_vs_attention"
    )

    for w in simple_ranking:
        child_curve = word_to_curve[w]
        if not np.isfinite(child_curve).any():
            continue
        s_small = np.array(small_surpr[w], dtype=float)
        a_small = np.array(small_act[w], dtype=float)
        if not np.isfinite(s_small).any() or not np.isfinite(a_small).any():
            continue

        child_mask = np.isfinite(child_curve)
        if not child_mask.any():
            continue
        x_child_raw = months_arr[child_mask]
        y_child_raw = child_curve[child_mask]

        s_mask = np.isfinite(s_small) & np.isfinite(steps_small_arr)
        if not s_mask.any():
            continue
        x_s_raw = steps_small_arr[s_mask]
        y_s_raw = s_small[s_mask]

        a_mask = np.isfinite(a_small) & np.isfinite(steps_small_arr)
        if not a_mask.any():
            continue
        x_a_raw = steps_small_arr[a_mask]
        y_a_raw = a_small[a_mask]

        params_child = fit_logistic_normalized(x_child_raw, y_child_raw)
        params_s = fit_logistic_normalized(x_s_raw, y_s_raw)
        params_a = fit_logistic_normalized(x_a_raw, y_a_raw)
        if params_child is None or params_s is None or params_a is None:
            continue

        y_child_fit = logistic4(x_shared, *params_child)
        y_s_fit = logistic4(x_shared, *params_s)
        y_a_fit = logistic4(x_shared, *params_a)

        if not (np.isfinite(y_child_fit).any() and np.isfinite(y_s_fit).any() and np.isfinite(y_a_fit).any()):
            continue

        y_child_norm = zscore(y_child_fit)
        y_s_norm = zscore(y_s_fit)
        y_a_norm = zscore(y_a_fit)

        if args.corr_type == "pearson":
            r_cs, p_cs = pearsonr(y_child_norm, y_s_norm)
            r_ca, p_ca = pearsonr(y_child_norm, y_a_norm)
            r_sa, p_sa = pearsonr(y_s_norm, y_a_norm)
        else:
            r_cs, p_cs = spearmanr(y_child_norm, y_s_norm)
            r_ca, p_ca = spearmanr(y_child_norm, y_a_norm)
            r_sa, p_sa = spearmanr(y_s_norm, y_a_norm)

        print(f"{w}\t{r_cs:.4f}\t{p_cs:.4e}\t{r_ca:.4f}\t{p_ca:.4e}\t{r_sa:.4f}\t{p_sa:.4e}")

        rows.append(
            {
                "word": w,
                "corr_type": args.corr_type,
                "r_child_surprisal": r_cs,
                "p_child_surprisal": p_cs,
                "r_child_attention": r_ca,
                "p_child_attention": p_ca,
                "r_surprisal_attention": r_sa,
                "p_surprisal_attention": p_sa,
                "child_L": params_child[0],
                "child_k": params_child[1],
                "child_x0": params_child[2],
                "child_b": params_child[3],
                "surprisal_L": params_s[0],
                "surprisal_k": params_s[1],
                "surprisal_x0": params_s[2],
                "surprisal_b": params_s[3],
                "attention_L": params_a[0],
                "attention_k": params_a[1],
                "attention_x0": params_a[2],
                "attention_b": params_a[3],
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(f"{args.corr_type}_results.csv", index=False)

if __name__ == "__main__":
    main()

# python src/experiments/dist_correlation.py --model_dir stanford-gpt2-medium-a_results --corr_type spearman
