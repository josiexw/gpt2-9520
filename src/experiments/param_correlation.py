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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="stanford-gpt2-small-a_results")
    parser.add_argument("--wordbank_csv", type=str, default="data/wordbank_item_data.csv")
    parser.add_argument("--max_simple", type=int, default=600)
    parser.add_argument("--out_csv", type=str, default="logistic_params.csv")
    parser.add_argument("--corr_type", type=str, choices=["pearson", "spearman"], default="pearson")
    args = parser.parse_args()

    word_aoa = load_wordbank_aoa(args.wordbank_csv)
    months, word_to_curve = load_wordbank_curves(args.wordbank_csv)
    months_arr = np.array(months, dtype=float)

    steps_small, small_surpr, small_act = load_results_dir(args.model_dir)
    steps_small_arr = np.array(steps_small, dtype=float)

    available_words = set(small_surpr.keys()) & set(small_act.keys()) & set(word_to_curve.keys())
    simple_ranking = get_simple_ranking(word_aoa, available_words, args.max_simple)

    rows = []

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

        rows.append(
            {
                "word": w,
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

    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    df_norm = df.copy()
    param_cols = [
        "child_L", "child_k", "child_x0", "child_b",
        "surprisal_L", "surprisal_k", "surprisal_x0", "surprisal_b",
        "attention_L", "attention_k", "attention_x0", "attention_b",
    ]
    for col in param_cols:
        if col not in df_norm.columns:
            continue
        col_vals = df_norm[col].to_numpy(dtype=float)
        mask = np.isfinite(col_vals)
        if np.sum(mask) > 1:
            mean = np.mean(col_vals[mask])
            std = np.std(col_vals[mask])
            if std > 0:
                col_vals[mask] = (col_vals[mask] - mean) / std
            else:
                col_vals[mask] = 0.0
            df_norm[col] = col_vals

    summary_rows = []
    systems = ["child", "surprisal", "attention"]
    param_names = ["L", "k", "x0", "b"]

    for param in param_names:
        for i in range(len(systems)):
            for j in range(i + 1, len(systems)):
                s1 = systems[i]
                s2 = systems[j]
                col1 = f"{s1}_{param}"
                col2 = f"{s2}_{param}"
                if col1 not in df_norm.columns or col2 not in df_norm.columns:
                    continue
                v1 = df_norm[col1].to_numpy(dtype=float)
                v2 = df_norm[col2].to_numpy(dtype=float)
                mask = np.isfinite(v1) & np.isfinite(v2)
                if np.sum(mask) < 2:
                    rho, p = np.nan, np.nan
                else:
                    if args.corr_type == "pearson":
                        rho, p = pearsonr(v1[mask], v2[mask])
                    else:
                        rho, p = spearmanr(v1[mask], v2[mask])
                summary_rows.append(
                    {
                        "param": param,
                        "pair": f"{s1}_{s2}",
                        "corr_type": args.corr_type,
                        "rho": rho,
                        "p": p,
                        "n": int(np.sum(mask)),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    summary_out = args.out_csv.replace(".csv", f"_{args.corr_type}_summary.csv")
    summary_df.to_csv(summary_out, index=False)


if __name__ == "__main__":
    main()

# python src/experiments/param_correlation.py --model_dir stanford-gpt2-medium-a_results --corr_type spearman