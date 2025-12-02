import argparse 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import pickle

from sklearn.linear_model import LinearRegression

def compute_log_freq(freq_dict):
    return {w: np.log10(freq) for w, freq in freq_dict.items() if freq > 0}

def compile_df(aoa, log_freq):
    return pd.DataFrame([
        {'word': w, 'aoa': aoa[w], 'log_freq': log_freq[w]}
        for w in aoa.keys() if w in log_freq
    ])

def fit_regression(df):
    x = df['log_freq'].values.reshape(-1, 1)
    y = df['aoa'].values

    model = LinearRegression().fit(x, y)

    r2 = model.score(x, y)
    n = len(df)
    p = 1 
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return model, r2, adj_r2

def plot_freq_vs_aoa(df, model, adj_r2, fig_path):
    plt.scatter(df['log_freq'], df['aoa'], alpha=0.2)

    x_line = np.linspace(df['log_freq'].min(), df['log_freq'].max(), 200)
    y_line = model.predict(x_line.reshape(-1, 1))

    plt.plot(x_line, y_line)

    plt.xlabel('Word log-frequency')
    plt.ylabel('GPT-2 AoA (steps, log10)')
    plt.title(f'GPT-2 Small: R2={adj_r2:.3f}')
    plt.savefig(fig_path)

def main():
    parser = argparse.ArgumentParser(description='Plot regression between log-frequency and AoA.')
    parser.add_argument('--fig_path', type=str, required=True)
    parser.add_argument('--freq_dict_path', type=str, required=True)
    parser.add_argument('--aoa_dict_path', type=str, required=True)
    args = parser.parse_args()

    with open(args.freq_dict_path, 'rb') as f:
        freq_dict = pickle.load(f)

    with open(args.aoa_dict_path, 'rb') as g:
        aoa_dict = pickle.load(g)

    log_freq = compute_log_freq(freq_dict)
    compiled_df = compile_df(aoa_dict, log_freq)
    model, _, adj_r2 = fit_regression(compiled_df)
    plot_freq_vs_aoa(compiled_df, model, adj_r2, args.fig_path)

if __name__ == '__main__':
    main()