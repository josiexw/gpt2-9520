import argparse
import json
import math
import torch
import torch.nn.functional as F

from transformer_lens import HookedTransformer
from tqdm import tqdm
from typing import Dict, List


class Experiment():
    def __init__(self,
                 model_name: str,
                 output_path: str,
                 batch_size: int):
        self.model_name = model_name 
        self.output_path = output_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size

        self.model = HookedTransformer.from_pretrained(self.model_name, device=self.device)
        self.model.eval()
        print(f'loaded model {self.model_name} on {self.device}')
        self.model.tokenizer.padding_side = 'right'

    def compute_batches(self,
                        prefixes: List[str],
                        word: str):
        full_text_l = [prefix.rstrip() + ' ' + word for prefix in prefixes]
        for i in range(0, len(full_text_l), self.batch_size):
            yield {
                'prefixes': prefixes[i:i + self.batch_size],
                'full_texts': full_text_l[i:i + self.batch_size]
            }

    def compute_surprisal(self,
                          batch: Dict[str, List[str]]):
        prefixes, full_texts = batch['prefixes'], batch['full_texts']
        batch_size = len(prefixes)

        prefix_lens = [len(self.model.to_tokens(prefix, prepend_bos=False)[0]) for prefix in prefixes]
        full_lens = [len(self.model.to_tokens(full_text, prepend_bos=False)[0]) for full_text in full_texts]
        num_word_tokens = [full - prefix for prefix, full in zip(prefix_lens, full_lens)]

        full_tokens = self.model.to_tokens(full_texts, prepend_bos=False)
        with torch.no_grad():
            logits, _ = self.model.run_with_cache(full_tokens) # [B, seq_len, d_vocab]

        total_logprob_e = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            word_token_positions = range(prefix_lens[i], full_lens[i])
            for position in word_token_positions:
                logits_pos = logits[i, position - 1]
                log_probs = F.log_softmax(logits_pos, dim=-1)
                token_id = int(full_tokens[i, position].item())
                total_logprob_e[i] += log_probs[token_id]
        
        surprisal_bits = (- total_logprob_e / math.log(2.0)).tolist()
        return surprisal_bits, num_word_tokens 

    def compute_surprisal_dict(self, 
                               contexts: Dict[str, List[str]]):
        surprisal_dict = {}
        words = list(contexts.keys())

        for word in tqdm(words):
            prefixes = contexts[word]
            surprisals, token_counts = [], []
            for batch in self.compute_batches(prefixes, word):
                surprisal_bits, num_word_tokens = self.compute_surprisal(batch)
                surprisals.extend(surprisal_bits)
                token_counts.extend(num_word_tokens)
            
            avg = sum(surprisals) / len(surprisals) 
            per_token_vals = [s / t for s, t in zip(surprisals, token_counts)]
            avg_per_token = sum(per_token_vals) / len(per_token_vals) 

            surprisal_dict[word] = {
                'avg_surprisal': avg, 
                'avg_surprisal_per_token': avg_per_token, 
                'surprisals_list': surprisals 
            }
        return surprisal_dict 

def main():
    parser = argparse.ArgumentParser(description='Compute average surprisal per simple word using GPT-2.')
    parser.add_argument('--model_name', type=str, default='gpt2-small', help='Name of pretrained GPT-2 model to use.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save surprisal dict as JSON.')
    parser.add_argument('--contexts_json', type=str, required=True, help='Path to contexts JSON.')
    args = parser.parse_args()

    with open(args.contexts_json, 'r') as f:
        contexts = json.load(f)
    experiment = Experiment(model_name=args.model_name, output_path=args.output_path)
    surprisal_dict = experiment.compute_surprisal_dict(contexts)

    with open(args.output_path, 'w') as f:
        json.dump(surprisal_dict, f)

if __name__ == '__main__':
    main()