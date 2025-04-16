import pandas as pd
import os
import sys
import json
import argparse
from utils.graded_algo import sal, cal, ours_graded, LLMOnly, run_random_experiment, pool_trec8, move_to_front_pooling, maxmean, LLMOnly, run_naive_experiment
from utils.utils import (save_results, calculate_reljudge_metrics)
import numpy as np
from typing import Optional

def uniform_smooth_normalize(data, column='prob_1', smoothness=1.0):
    """
    Transform the distribution to be more uniform with controllable smoothness.
    smoothness: 0 means no change, 1 means fully uniform distribution
    Values between 0 and 1 give intermediate results
    """
    x = data[column].values
    
    # Calculate empirical CDF
    sorted_idx = np.argsort(x)
    ranks = np.empty_like(sorted_idx, dtype=float)
    ranks[sorted_idx] = np.linspace(0, 1, len(x))
    
    # Interpolate between original values and uniform distribution
    smoothed_values = (1 - smoothness) * x + smoothness * ranks
    
    # Min-max normalize the result
    normalized = (smoothed_values - smoothed_values.min()) / (smoothed_values.max() - smoothed_values.min())
    
    data[column] = normalized
    return data

def load_previous_results(output_dir):
    json_path = os.path.join(output_dir, 'all_metrics.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return {}

def smooth_normalize(data, column='pi', smoothness=1):
    x = data[column].values
    sigmoid = 1 / (1 + np.exp(-smoothness * (x - np.mean(x))))
    normalized = (sigmoid - sigmoid.min()) / (sigmoid.max() - sigmoid.min())
    data[column] = normalized
    return data

def run_experiment(data, method:str, B: int, random_state: int, learning_rate: float, batch_size: int, ours_lambda: int, runs_path: Optional[str], track: str, doc_path: Optional[str] = None):
    if method.startswith('Ours') and method.endswith('Groups'):
        n_groups = int(method.replace('Ours', '').replace('Groups', ''))
        _, ours_group_data = ours_graded(data=data, annotation_budget=B, learning_rate=learning_rate, 
                                            random_state=random_state, 
                                            n_groups=n_groups)
        return calculate_reljudge_metrics(metrics=['map', 'ndcg', 'recall_1000'], model='Llama', 
                                            root=f"../results/{track}", df=ours_group_data, track = track)
    elif method == 'OursIndiv':
        n_groups = len([f for f in os.listdir(runs_path) if os.path.isfile(os.path.join(runs_path, f))])
        _, ours_indiv_data = ours_graded(data=data, annotation_budget=B, learning_rate=learning_rate, 
                                            random_state=random_state, 
                                            n_groups=n_groups)
        return calculate_reljudge_metrics(metrics=['map', 'ndcg', 'recall_1000'], model='Llama', 
                                            root=f"../results/{track}", df=ours_indiv_data, track = track)
    elif method == 'Ours':
        _, ours_data = ours_graded(data=data, annotation_budget=B, learning_rate=learning_rate, 
                            random_state=random_state)
        return calculate_reljudge_metrics(metrics=['map', 'ndcg', 'recall_1000'], model='Llama', 
                                            root=f"../results/{track}", df=ours_data, track = track)
    elif method == 'Random':
        _, random_data = run_random_experiment(data, B, random_state=random_state)
        return calculate_reljudge_metrics(metrics=['map', 'ndcg', 'recall_1000'], model='Llama', 
                                            root=f"../results/{track}", df=random_data, track = track)
    elif method == 'Naive':
        _, naive_data = run_naive_experiment(data, B, random_state=random_state)
        return calculate_reljudge_metrics(metrics=['map', 'ndcg', 'recall_1000'], model='Llama', 
                                            root=f"../results/{track}", df=naive_data, track = track)
    elif method == 'Pool':
        pool_data, _ = pool_trec8(data=data, budget=B, runs_path=runs_path)
        return calculate_reljudge_metrics(metrics=['map', 'ndcg', 'recall_1000'], model='Llama', 
                                            root=f"../results/{track}", df=pool_data, track = track)
    elif method == 'MTF':
        mtf_data, _ = move_to_front_pooling(data=data, budget=B, runs_path=runs_path)
        return calculate_reljudge_metrics(metrics=['map', 'ndcg', 'recall_1000'], model='Llama', 
                                            root=f"../results/{track}", df=mtf_data, track = track)
    elif method == 'MaxMean':
        mm_data, _ = maxmean(data=data, budget=B, runs_path=runs_path)
        print(mm_data.head())
        print(mm_data.shape)
        return calculate_reljudge_metrics(metrics=['map', 'ndcg', 'recall_1000'], model='Llama', 
                                            root=f"../results/{track}", df=mm_data, track = track)
    elif method == 'CAL-human':
        cal_data, cal_data_llm = cal(data=data, budget=B, doc_path=doc_path)
        print(cal_data.head())
        print(cal_data.shape)
        return calculate_reljudge_metrics(metrics=['map', 'ndcg', 'recall_1000'], model='Llama', 
                                            root=f"../results/{track}", df=cal_data, track = track)
    elif method == 'CAL-hybrid':
        cal_data, cal_data_llm = cal(data=data, budget=B, doc_path=doc_path )
        print(cal_data_llm.head())
        print(cal_data_llm.shape)
        return calculate_reljudge_metrics(metrics=['map', 'ndcg', 'recall_1000'], model='Llama', 
                                            root=f"../results/{track}", df=cal_data_llm, track = track)
    elif method == 'SAL-human':
        sal_data, sal_data_llm = sal(data=data, budget=B, doc_path=doc_path)
        print(sal_data.head())
        print(sal_data.shape)
        return calculate_reljudge_metrics(metrics=['map', 'ndcg', 'recall_1000'], model='Llama', 
                                            root=f"../results/{track}", df=sal_data, track = track)
    elif method == 'SAL-hybrid':
        sal_data, sal_data_llm = sal(data=data, budget=B, doc_path=doc_path)
        print(sal_data_llm.head())
        print(sal_data_llm.shape)
        return calculate_reljudge_metrics(metrics=['map', 'ndcg', 'recall_1000'], model='Llama', 
                                            root=f"../results/{track}", df=sal_data_llm, track = track)
    elif method == 'LLMOnly':
        llm_data, _ = LLMOnly(data=data)
        return calculate_reljudge_metrics(metrics=['map', 'ndcg', 'recall_1000'], model='Llama', 
                                            root=f"../results/{track}", df=llm_data, track = track)
    elif method == 'Ours-llm':
        ours_data, ours_plusllm = ours_graded(data=data, annotation_budget=B, learning_rate=learning_rate, 
                            random_state=random_state)
        if B < len(data)-1:
            predicted_only = ours_plusllm[~ours_plusllm.index.isin(ours_data.index)]
        else: 
            predicted_only = ours_plusllm
        return calculate_reljudge_metrics(metrics=['map', 'ndcg', 'recall_1000'], model='Llama', 
                                            root=f"../results/{track}", df=predicted_only, track = track)
    elif method == 'Random-llm':
        random_data, random_plusllm = run_random_experiment(data, B, random_state=random_state)
        if B < len(data)-1:
            predicted_only_random = random_plusllm[~random_plusllm.index.isin(random_data.index)]
        else:
            predicted_only_random = random_plusllm
        return calculate_reljudge_metrics(metrics=['map', 'ndcg', 'recall_1000'], model='Llama', 
                                            root=f"../results/{track}", df=predicted_only_random, track = track)
    elif method == 'Naive-llm':
        naive_data, naive_plusllm = run_naive_experiment(data, B, random_state=random_state)
        # Get the predicted annotations only
        if B < len(data)-1:
            predicted_only_naive = naive_plusllm[~naive_plusllm.index.isin(naive_data.index)]
        else:
            predicted_only_naive = naive_plusllm
        return calculate_reljudge_metrics(metrics=['map', 'ndcg', 'recall_1000'], model='Llama', 
                                            root=f"../results/{track}", df=predicted_only_naive, track = track)
    else:
        raise ValueError(f"Unknown method: {method}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with specific methods.")
    parser.add_argument('methods', nargs='+', type=str,
                        help="Methods to run (Ours, Random, Naive, Pool, MTF, OursGroup[N])")

    parser.add_argument('--drop', action='store_true',
                        help="If set, only update 'drop_*' metrics in the results.")
    
    parser.add_argument('--drop5', action='store_true',
                        help="If set, only update 'drop_*' metrics in existing results.")
    
    # parser.add_argument("--prompt_type", type=str, choices=["utility", "rational", "simple"], required=True, 
    #                     help="Specify the type of prompt to use: 'utility', 'rational', or 'simple'.")

    args = parser.parse_args()

    track = "gcovid5" # dataset name
    prompt_type= "simple" # base prompt

    input_path = f"../results/{track}/{prompt_type}-Llama-3.1-70B-Instruct.csv" # The path to the LLM predictions (CSV file in the form of <topic_id,doc_id,annotation,prob_0,...,prob_k>)
    data = pd.read_csv(input_path)
    data['annotation'] = data['annotation']

    
    if track == "dl2020":
        data['annotation'] = (data['annotation'] >= 2).astype(int)
    
    if track == "covid5":
        data['annotation'] = (data['annotation'] >= 1).astype(int)

    doc_len = len(data)
    b_list = [
        int(1 / 512 * doc_len), int(1 / 256 * doc_len), int(1 / 128 * doc_len),
        int(1 / 64 * doc_len), int(1 / 32 * doc_len), int(1 / 16 * doc_len),
        int(1 / 8 * doc_len), int(1 / 4 * doc_len), int(1 / 2 * doc_len),
        doc_len - 1
    ]

    random_state = 12345
    learning_rate = 0.01
    batch_size = 10
    ours_lambda = 200
    labels = [
        'Ours', 'Ours3Groups', 'OursIndiv', 'Random', 'MTF', 'Naive',
        'Pool', 'MaxMean', 'CAL-human', 'CAL-hybrid', 'SAL-human', 
        'SAL-hybrid', 'Ours-llm', 'Random-llm', 'Naive-llm', 'LLMOnly'
    ]
    output_dir = f"OUTPUT_PATH" # The path to save the results
    runs_path = f"RUNS_PATH" # The path to the runs submitted
    doc_path = f"PATH_TO_DOC_PKL_FILE" # If you use SAL/CAL methods, you need the unique feature indices of documents here

    # Load previous results
    previous_results = load_previous_results(output_dir)
    
    # Run the specified methods and update results
    for method in args.methods:
        print(f"Running experiments for method: {method}")
        for i, B in enumerate(b_list):
            print(f"  Running {method} experiment with B={B}")
            result = run_experiment(
                data, method=method, B=B, random_state=random_state,
                learning_rate=learning_rate, batch_size=batch_size,
                ours_lambda=ours_lambda, runs_path=runs_path,
                track=track, doc_path=doc_path
            )

            # Update previous_results
            if method not in previous_results:
                previous_results[method] = {}

            # Existing results for this method and budget
            old_dict = previous_results[method].get(str(B), {})

            if args.drop:
                # Only update metrics starting with "drop_"
                for metric_name, val in result.items():
                    if metric_name.startswith("drop_"):
                        old_dict[metric_name] = val
                previous_results[method][str(B)] = old_dict
            else:
                # Overwrite all metrics
                previous_results[method][str(B)] = result

        # Save results after each method completes
        all_metric_scores = []
        for label in labels:
            method_scores = []
            for B in b_list:
                if label in previous_results and str(B) in previous_results[label]:
                    method_scores.append(previous_results[label][str(B)])
                else:
                    method_scores.append({}) 
            all_metric_scores.append(method_scores)
        
        save_results(b_list, all_metric_scores, labels, output_dir)
        print(f"Saved results after completing method: {method}")
    
    print(f"All experiments completed. Final results saved in {output_dir}")
