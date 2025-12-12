import argparse 
import matplotlib.pyplot as plt 
import pandas as pd 

from scipy.stats import ttest_rel

def test_mse(mse_csv, output_png):
    mse_df = pd.read_csv(mse_csv)
    mse_df = mse_df.dropna(subset=['mse_small_child', 'mse_medium_child'])
    diff = mse_df['mse_medium_child'] - mse_df['mse_small_child']

    # verify that the distribution of MSE differences is approximately normal 
    plt.hist(diff, bins=15)
    plt.xlabel('MSE Difference (medium - small)')
    plt.ylabel('Count')
    plt.title('Distribution of MSE Differences')
    plt.savefig(output_png)

    # conduct paired t-test 
    t_test_res = ttest_rel(
        mse_df['mse_medium_child'],
        mse_df['mse_small_child'],
        alternative='greater'
    )

    print(f'MSE mean difference: {diff.mean()}')
    print(f'Paired t-test results: {t_test_res}')

def main():
    parser = argparse.ArgumentParser(description='Analyze MSE results.')
    parser.add_argument('--mse_csv_path', type=str, required=True)
    parser.add_argument('--output_png', type=str, required=True)
    args = parser.parse_args()

    test_mse(args.mse_csv_path, args.output_png)

if __name__ == '__main__':
    main()