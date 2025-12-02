import argparse
import re
import pandas as pd 
import pickle 
import warnings
from collections import Counter
import urllib3
from urllib3.exceptions import NotOpenSSLWarning
from datasets import load_dataset
from tqdm import tqdm
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
urllib3.disable_warnings(NotOpenSSLWarning)

def parse_wordbank(wordbank_path):
    word_set = set()

    df = pd.read_csv(wordbank_path)
    df['item_definition_clean'] = df['item_definition'].str.replace(r'\s*\(.*?\)', '', regex=True)
    for _, row in df.iterrows():
        word = str(row['item_definition_clean']).split('*')[0].split('/')[0].strip().lower()
        if not word:
            continue
        word_set.add(word)
    return word_set

def get_word_frequencies(max_docs: int, word_set: set):
    dataset = load_dataset('openwebtext', split='train', streaming=True, trust_remote_code=True)
    docs_scanned = 0

    word_counter = Counter()
    for item in tqdm(dataset):
        text = item.get('text') or ''
        words = re.findall(r'[A-Za-z]+', text.lower())

        filtered_words = [w for w in words if w in word_set]
        if filtered_words:
            word_counter.update(filtered_words)

        docs_scanned += 1
        if docs_scanned >= max_docs:
            break

    return word_counter

def main():
    parser = argparse.ArgumentParser(description='Parse OWT for common children words.')
    parser.add_argument('--max_docs', type=int, default=10_000_000)
    parser.add_argument('--wordbank_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    word_set = parse_wordbank(args.wordbank_path)
    word_counter = get_word_frequencies(args.max_docs, word_set)

    with open(args.output_path, 'wb') as f:
        pickle.dump(word_counter, f)

if __name__ == "__main__":
    main()
