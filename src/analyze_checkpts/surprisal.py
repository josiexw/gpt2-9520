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
                 output_path: str):
        self.model_name = model_name 
        self.output_path = output_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # TODO: Batching 

        self.model = HookedTransformer.from_pretrained(self.model_name, device=self.device)
        self.model.eval()
        print(f'loaded model {self.model_name} on {self.device}')

    def compute_surprisal(self, 
                          prefix: str,
                          word: str):
        full_text = prefix.rstrip() + ' ' + word

        prefix_tokens = self.model.to_tokens(prefix)
        full_tokens = self.model.to_tokens(full_text)
        prefix_len, full_len = prefix_tokens.shape[1], full_tokens.shape[1]
        num_word_tokens = full_len - prefix_len

        logits, _ = self.model.run_with_cache(full_tokens) # (1, seq_len, d_vocab)

        word_token_positions = list(range(prefix_len, full_len))

        total_logprob_e = 0.0
        for position in word_token_positions:
            logits = logits[0, position - 1]
            log_probs = F.log_softmax(logits, dim=-1)
            token_id = int(full_tokens[0, position].item())
            total_logprob_e += float(log_probs[token_id].item())
        
        surprisal_bits = - total_logprob_e / math.log(2.0)
        return surprisal_bits, num_word_tokens
    
    def compute_surprisal_dict(self,
                               contexts: Dict[str, List[str]]):
        surprisal_dict = {}
        words = list(contexts.keys())

        for word in tqdm(words):
            prefixes = contexts[word]
            surprisals, token_counts = [], []
            for prefix in prefixes:
                surprisal_bits, num_word_tokens = self.compute_surprisal(prefix, word)
                surprisals.append(surprisal_bits)
                token_counts.append(num_word_tokens)
            
            avg = sum(surprisals) / len(surprisals)
            per_token_vals = [s / t for s, t in zip(surprisals, token_counts)]
            avg_per_token = sum(per_token_vals) / len(per_token_vals)

            surprisal_dict[word] = {
                'avg_surprisal': avg,
                'avg_surprisal_per_token_bits': avg_per_token,
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