import argparse 
import math
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import pandas as pd 
import pickle
import seaborn as sns 

from scipy.stats import linregress, spearmanr

def load_wordbank_aoa(csv_path: str):
    df = pd.read_csv(csv_path)
    df['item_definition_clean'] = df['item_definition'].str.replace(r'\s*\(.*?\)', '', regex=True)

    aoa_cols = [c for c in df.columns if c.isdigit()]
    aoa_cols_sorted = sorted(aoa_cols, key=lambda x: int(x))
    word_to_aoa = {}
    for _, row in df.iterrows():
        word = str(row["item_definition_clean"]).split('*')[0].split('/')[0].strip().lower()
        if not word:
            continue
        aoa = math.nan
        for col in aoa_cols_sorted:
            try:
                v = float(col)
                v = float(row[col])
            except (TypeError, ValueError):
                try:
                    v = float(row[col])
                except (TypeError, ValueError):
                    continue
            try:
                v = float(row[col])
            except (TypeError, ValueError):
                continue
            if v >= 0.5:
                aoa = float(col)
                break
        if not math.isnan(aoa):
            word_to_aoa[word] = aoa
    return word_to_aoa

def parse_wordbank(wordbank_path):
    df = pd.read_csv(wordbank_path)
    df['item_definition_clean'] = df['item_definition'].str.replace(r'\s*\(.*?\)', '', regex=True)

    aoa_cols = [str(c) for c in range(16, 31)]
    word_to_proportions = {}
    for _, row in df.iterrows():
        word = str(row["item_definition_clean"]).split('*')[0].split('/')[0].strip().lower()
        word_to_proportions[word] = row[aoa_cols].tolist()
    return word_to_proportions 

def compute_aoa_surp_spearman(model_surprisal, wordbank_path, size):
    word_to_proportions = parse_wordbank(wordbank_path)
    months = list(range(16, 31))
    model_ckpts = sorted(model_surprisal.keys())

    word_surprisals = {}
    for _, ckpt_dict in model_surprisal.items():
        for word, word_dict in ckpt_dict.items():
            word_surprisals.setdefault(word, []).append(sum(word_dict[size]) / len(word_dict[size]))
    
    model_slopes, child_slopes = [], []
    model_slope_dict, child_slope_dict = {}, {}

    for word in word_surprisals:
        slope_model, _, _, _, _ = linregress(model_ckpts, word_surprisals[word])
        slope_child, _, _, _, _ = linregress(months, word_to_proportions[word])

        model_slopes.append(slope_model)
        child_slopes.append(slope_child)

        model_slope_dict[word] = slope_model
        child_slope_dict[word] = slope_child
    
    rho, pval = spearmanr(model_slopes, child_slopes)
    print(f'spearman correlation across words: {rho}')
    print(f'p-val: {pval}')
    return model_slope_dict, child_slope_dict

def compute_meanjsd(all_ckpt_jsd_dict):
    mean_jsd_dict = {}

    for ckpt, ckpt_dict in all_ckpt_jsd_dict.items():
        for word, jsd_w in ckpt_dict.items():
            mean_jsd_dict.setdefault(word, []).append(jsd_w)
    
    return {word: np.mean(jsd_l) for word, jsd_l in mean_jsd_dict.items()}

def jsd_analysis(small_slope_dict, medium_slope_dict, child_slope_dict, mean_jsd_dict, frequency_dict):
    words = list(small_slope_dict.keys())

    q1 = np.percentile(list(mean_jsd_dict.values()), 25)
    q3 = np.percentile(list(mean_jsd_dict.values()), 75)

    q1_jsd_words = [word for word in words if mean_jsd_dict[word] < q1]
    q3_jsd_words = [word for word in words if mean_jsd_dict[word] > q3]

    def corr(model_slope_dict, child_slope_dict, word_list):
        x = [model_slope_dict[word] for word in word_list]
        y = [child_slope_dict[word] for word in word_list]
        rho, p_val = spearmanr(x, y)
        return rho, p_val
    
    print('Is the correlation between change in surprisal & child learning dependent on the value of JSD?')
    print('Q1 JSD words:')
    small_rho, small_p_val = corr(small_slope_dict, child_slope_dict, q1_jsd_words)
    print(f'small vs. child: {small_rho}, p-val: {small_p_val}')
    medium_rho, medium_p_val = corr(medium_slope_dict, child_slope_dict, q1_jsd_words)
    print(f'medium vs. child: {medium_rho}, p-val: {medium_p_val}')
    print('Q3 JSD words:')
    small_rho, small_p_val = corr(small_slope_dict, child_slope_dict, q3_jsd_words)
    print(f'small vs. child: {small_rho}, p-val: {small_p_val}')
    medium_rho, medium_p_val = corr(medium_slope_dict, child_slope_dict, q3_jsd_words)
    print(f'medium vs. child: {medium_rho}, p-val: {medium_p_val}')

    slope_diff = [abs(small_slope_dict[word] - medium_slope_dict[word]) for word in words]
    owt_frequencies = [frequency_dict[word] for word in words]
    jsd_vals = [mean_jsd_dict[word] for word in words]

    print(f'Does JSD predict the difference in slopes between small and medium?')
    rho, p_val = spearmanr(slope_diff, jsd_vals)
    print(f'spearman between |small - medium slope| and JSD: {rho}, p-val: {p_val}')

    print(f'Does JSD correlate with the frequency in the OpenWebText?')
    rho, p_val = spearmanr(owt_frequencies, jsd_vals)
    print(f'spearman between OWT frequency and JSD: {rho}, p-val: {p_val}')

def main():
    parser = argparse.ArgumentParser(description='Analyze JSD and surprisal across training checkpoints.')
    parser.add_argument('--wordbank_path', type=str, required=True)
    parser.add_argument('--owt_frequency_path', type=str, required=True)
    parser.add_argument('--all_ckpt_jsd_path', type=str, required=True)
    parser.add_argument('--all_ckpt_surprisal_path', type=str, required=True)
    args = parser.parse_args()

    with open(args.owt_frequency_path, 'rb') as f:
        owt_frequency_dict = pickle.load(f)
    
    with open(args.all_ckpt_jsd_path, 'rb') as f:
        all_ckpt_jsd_dict = pickle.load(f) 

    with open(args.all_ckpt_surprisal_path, 'rb') as f:
        all_ckpt_surprisal_dict = pickle.load(f) 
    
    mean_jsd_dict = compute_meanjsd(all_ckpt_jsd_dict)
    print('How does change in surprisal correlate with how fast children learn words?')
    print(f'GPT-2 Small:')
    small_slope_dict, child_slope_dict = compute_aoa_surp_spearman(all_ckpt_surprisal_dict, args.wordbank_path, 'small')
    print(f'GPT-2 Medium:')
    medium_slope_dict, child_slope_dict = compute_aoa_surp_spearman(all_ckpt_surprisal_dict, args.wordbank_path, 'medium')

    jsd_analysis(small_slope_dict, medium_slope_dict, child_slope_dict, mean_jsd_dict, owt_frequency_dict)

if __name__ == '__main__':
    main()