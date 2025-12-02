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
    parser.add_argument('--max_docs', type=int, default=10000000)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    with open('C:\\Users\\olivi\\OneDrive\\Desktop\\PythonPrograms\\gpt2-9520\\data\\contexts_cosmopedia.pkl', 'rb') as f:
        cosmopedia_dict = pickle.load(f)

    word_set = set(cosmopedia_dict.keys())
    word_counter = get_word_frequencies(args.max_docs, word_set)

    with open(args.output_path, 'wb') as g:
        pickle.dump(word_counter, g)

if __name__ == "__main__":
    main()