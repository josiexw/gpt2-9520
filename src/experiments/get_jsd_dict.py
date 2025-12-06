import argparse
import numpy as np 
import os
import pickle 

from scipy.spatial.distance import jensenshannon

def convert_to_dist(surprisal_list):
    prob_dist = np.exp(-np.array(surprisal_list))
    return prob_dist / prob_dist.sum()

def compute_jsd(small_surp_l, medium_surp_l):
    small_dist, medium_dist = convert_to_dist(small_surp_l), convert_to_dist(medium_surp_l)
    js_distance = jensenshannon(small_dist, medium_dist)
    return js_distance ** 2

def get_dir_contents(dir):
    return {file for _, _, files in os.walk(dir) for file in files}

def compute_jsd_dict(small_dir, medium_dir, joint_ckpt_set):
    all_ckpt_jsd_dict = {}

    for ckpt in joint_ckpt_set:
        ckpt_jsd_dict = {}
        small_ckpt_path, medium_ckpt_path = os.path.join(small_dir, ckpt), os.path.join(medium_dir, ckpt)

        with open(small_ckpt_path, 'rb') as f:
            small_ckpt_dict = pickle.load(f)
        
        with open(medium_ckpt_path, 'rb') as f:
            medium_ckpt_dict = pickle.load(f)
        
        word_l = list(small_ckpt_dict.keys())
        for word in word_l:
            jsd_w = compute_jsd(small_ckpt_dict[word]['surprisals_list'], medium_ckpt_dict[word]['surprisals_list'])
            ckpt_jsd_dict[word] = jsd_w 
        
        all_ckpt_jsd_dict[ckpt] = ckpt_jsd_dict
    return all_ckpt_jsd_dict

def get_jsd_dict(small_dir, medium_dir, output_path):
    small_ckpt_set, medium_ckpt_set = get_dir_contents(small_dir), get_dir_contents(medium_dir)
    joint_ckpt_set = small_ckpt_set.intersection(medium_ckpt_set)
    all_ckpt_jsd_dict = compute_jsd_dict(small_dir, medium_dir, joint_ckpt_set)

    with open(output_path, 'wb') as f:
        pickle.dump(all_ckpt_jsd_dict, f)

def main():
    parser = argparse.ArgumentParser(description='Compute JSD dict for small vs. medium GPT-2.')
    parser.add_argument('--small_dir', type=str, required=True)
    parser.add_argument('--medium_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    get_jsd_dict(args.small_dir, args.medium_dir, args.output_path)

if __name__ == '__main__':
    main()