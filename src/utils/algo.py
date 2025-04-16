import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from scipy.interpolate import interp1d
from scipy.stats import beta, gaussian_kde
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from scipy.stats import gaussian_kde
from .utils import count_in_ranges
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from typing import Tuple
import random

import mmh3 
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from collections import defaultdict
import os
from typing import Tuple, Set, List

import pickle

class GradualMultinomialLogisticRegression:
    def __init__(self, learning_rate, n_classes, random_state=None):
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Initialize parameters
        self.beta0 = np.zeros(n_classes)  # Bias terms for each class
        self.beta = np.zeros((n_classes, n_classes))  # Coefficients for each class
        
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
        return exp_z / exp_z.sum(axis=1, keepdims=True)
    
    def predict_proba(self, pi):
        """
        Predict calibrated probabilities based on LLM predictions.
        """
        # pi: shape (n_samples, n_classes)
        z = self.beta0 + np.dot(pi, self.beta.T)  # Shape (n_samples, n_classes)
        probs = self.softmax(z)
        return probs  # Shape (n_samples, n_classes)
    
    def update_params(self, X, y):
        """
        Update model parameters using gradient descent.
        """
        n_samples = X.shape[0]
        z = self.beta0 + np.dot(X, self.beta.T)  # Shape (n_samples, n_classes)
        y_pred = self.softmax(z)  # Shape (n_samples, n_classes)
        
        # One-hot encode the true labels
        y_true = np.zeros_like(y_pred)
        y_true[np.arange(n_samples), y] = 1
        
        # Compute gradients
        grad_beta0 = np.mean(y_pred - y_true, axis=0)  # Shape (n_classes,)
        grad_beta = np.dot((y_pred - y_true).T, X) / n_samples  # Shape (n_classes, n_classes)
        
        # Update parameters
        self.beta0 -= self.learning_rate * grad_beta0
        self.beta -= self.learning_rate * grad_beta
        
    def fit(self, X, y):
        """
        Fit the model to new data.
        """
        self.update_params(X, y)
        
    def sample_next(self, data):
        """
        Select the next data point to annotate based on the smallest margin.
        """
        pi_columns = [f'prob_{k}' for k in range(self.n_classes)]
        pi = data[pi_columns].values  # Shape (n_samples, n_classes)
        probs = self.predict_proba(pi)  # Shape (n_samples, n_classes)
        
        # Compute margins between top two predicted classes
        sorted_probs = np.sort(probs, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        idx = np.argmin(margins)
        return data.iloc[idx]

def ours_graded(data, annotation_budget, learning_rate, batch_size, lambda_, random_state=None, n_groups=1):
    import random
    # Set random seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Check if we're annotating everything
    total_docs = len(data)
    final_dataset = data.copy()
    if annotation_budget >= total_docs - 1:
        print(f"Annotation budget ({annotation_budget}) >= total documents ({total_docs}). Annotating everything.")
        final_dataset['annotation_pred'] = final_dataset['annotation']
        return final_dataset[final_dataset['annotation_pred'].notna()], final_dataset
    
    # Initialize all data as unlabeled
    unlabeled_pool = data.copy()
    unlabeled_pool['is_labeled'] = False
    
    # Determine the number of classes
    n_classes = len([col for col in data.columns if col.startswith('prob_')])
    
    model = GradualMultinomialLogisticRegression(
        learning_rate=learning_rate,
        n_classes=n_classes,
        random_state=random_state,
    )
    
    # Ensure n_groups is within valid range
    n_groups = max(1, min(n_groups, len(unlabeled_pool['topic_id'].unique())))
    
    # Group data by topic_id and get group sizes
    topic_groups = unlabeled_pool.groupby('topic_id')
    group_sizes = topic_groups.size().values
    
    # Randomly assign topics to groups, ensuring a balanced distribution
    topic_ids = unlabeled_pool['topic_id'].unique()
    random.shuffle(topic_ids)
    groups = [[] for _ in range(n_groups)]
    group_doc_counts = [0] * n_groups
    for topic_id in topic_ids:
        group_index = min(range(n_groups), key=lambda i: group_doc_counts[i])
        groups[group_index].append(topic_id)
        group_doc_counts[group_index] += len(unlabeled_pool[unlabeled_pool['topic_id'] == topic_id])
    
    # Calculate initial budget per group
    group_budgets = [min(count, annotation_budget // n_groups) for count in group_doc_counts]
    remaining_budget = annotation_budget - sum(group_budgets)
    
    # Distribute remaining budget
    while remaining_budget > 0:
        for i in range(n_groups):
            if group_budgets[i] < group_doc_counts[i] and remaining_budget > 0:
                group_budgets[i] += 1
                remaining_budget -= 1
            if remaining_budget == 0:
                break
    
    total_annotations = 0
    
    # Use tqdm to track progress of annotations
    with tqdm(total=annotation_budget, desc="Annotating") as pbar:
        while total_annotations < annotation_budget:
            for group_index, group in enumerate(groups):
                if group_budgets[group_index] > 0:
                    group_mask = unlabeled_pool['topic_id'].isin(group)
                    group_data = unlabeled_pool[group_mask & ~unlabeled_pool['is_labeled']]
                    
                    if len(group_data) == 0:
                        # Redistribute budget if group is exhausted
                        remaining_budget = group_budgets[group_index]
                        group_budgets[group_index] = 0
                        for i in range(n_groups):
                            if i != group_index and group_budgets[i] < group_doc_counts[i]:
                                transfer = min(remaining_budget, group_doc_counts[i] - group_budgets[i])
                                group_budgets[i] += transfer
                                remaining_budget -= transfer
                            if remaining_budget == 0:
                                break
                        continue
                    
                    # Sample the next point to annotate from the current group
                    next_sample = model.sample_next(group_data)
                    
                    unlabeled_pool.loc[next_sample.name, 'is_labeled'] = True
                    # Get the pi values and the annotation
                    pi_columns = [f'prob_{k}' for k in range(n_classes)]
                    X_new = next_sample[pi_columns].values.reshape(1, -1)
                    y_new = np.array([next_sample['annotation']])
                    
                    # Update the model with the new labeled point
                    model.fit(X_new, y_new)
                    
                    total_annotations += 1
                    group_budgets[group_index] -= 1
                    pbar.update(1)
                    
                    if total_annotations >= annotation_budget:
                        break
                
                if all(budget == 0 for budget in group_budgets):
                    break
    
    final_dataset = data.copy()
    
    # For annotated points, use the actual annotations
    labeled_data = unlabeled_pool[unlabeled_pool['is_labeled']]
    final_dataset.loc[labeled_data.index, 'annotation_pred'] = labeled_data['annotation']
    
    # For non-annotated points, use model predictions
    non_annotated_mask = ~final_dataset.index.isin(labeled_data.index)
    non_annotated_data = final_dataset.loc[non_annotated_mask]
    pi_columns = [f'prob_{k}' for k in range(n_classes)]
    pi_values = non_annotated_data[pi_columns].values
    predictions_proba = model.predict_proba(pi_values)
    predictions = np.argmax(predictions_proba, axis=1)
    
    final_dataset.loc[non_annotated_mask, 'annotation_pred'] = predictions
    
    return labeled_data, final_dataset


def LLMOnly(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Implements the LLM-Only method for document annotation.
    
    Args:
        data: DataFrame containing annotations
        budget: Maximum number of documents to annotate
    
    Returns:
        Tuple of (annotated_docs_df, complete_dataset)
    """
    modified_data = data.copy()
    
    # Convert pi to binary values based on 0.5 threshold
    modified_data['pi'] = (modified_data['pi'] >= 0.5).astype(int)
    
    # For this implementation, both returned DataFrames are identical
    # since we're just modifying the pi values
    return modified_data, modified_data

class OnlineLogisticRegression:
    def __init__(self, num_features=100_000_000):
        self.weights = np.zeros(num_features)
    
    def predict_proba(self, feature_indices) -> float:
        score = np.sum(self.weights[feature_indices])
        return 1 / (1 + np.exp(-score))
    
    def update(self, feature_indices, y: int):
        prob = self.predict_proba(feature_indices)
        self.weights[feature_indices] += (y - prob)

def sal(data: pd.DataFrame, budget: int, doc_path: str, select_size: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple Active Learning implementation selecting documents with highest uncertainty.
    
    Args:
        data: DataFrame containing annotations (simulated ground truth for evaluation)
        budget: Maximum documents to annotate
        doc_path: Path to preprocessed features
        select_size: Number of documents to select in each iteration
    """
    # Load features
    print("Loading preprocessed document features...")
    with open(doc_path, 'rb') as f:
        doc_features = pickle.load(f)

    # Convert doc_features from dict to a list or array aligned with doc_ids for faster indexing
    data_ = data.copy()
    doc_ids = data_['doc_id'].values
    labels = data_['annotation'].values
    n_docs = len(doc_ids)
    
    # Create an array of features aligned with doc_ids
    # If some doc_id isn't in doc_features, we store None or a placeholder.
    # We'll handle that doc as invalid later.
    feature_list = []
    valid_mask = np.zeros(n_docs, dtype=bool)
    for i, d_id in enumerate(doc_ids):
        if d_id in doc_features:
            feature_list.append(doc_features[d_id])
            valid_mask[i] = True
        else:
            feature_list.append(None)
    # feature_list is now a list of arrays (or None). 
    # If all features have the same dimension, you can stack them into a NumPy array.
    # For example:
    #   feature_dim = feature_list[0].shape[0]
    #   X = np.zeros((n_docs, feature_dim))
    #   for i, f in enumerate(feature_list):
    #       if f is not None:
    #           X[i] = f
    # This allows batch operations if the model supports it.
    # For now, we leave it as a list to not break the original logic.

    total_docs = n_docs
    clf = OnlineLogisticRegression()
    annotated_mask = np.zeros(n_docs, dtype=bool)
    result_indices = []

    if budget >= total_docs - 1:
        print(f"Annotation budget ({budget}) >= total documents ({total_docs}). Annotating everything.")
        data_['pi'] = data_['annotation']
        data_['pi'] = (data_['pi'] > 0.5).astype(int)  # Binarize as requested
        return data_[data_['pi'].notna()], data_

    # Initial seed of min(50, n_docs)
    initial_seed_size = min(50, n_docs)
    seed_indices = np.random.choice(np.arange(n_docs), initial_seed_size, replace=False)
    
    # Update with seed documents
    for idx in seed_indices:
        if valid_mask[idx]:
            clf.update(feature_list[idx], labels[idx])
            annotated_mask[idx] = True
            result_indices.append(idx)
    
    budget_remaining = budget - len(result_indices)

    with tqdm(total=budget, desc='SAL Progress') as pbar:
        pbar.update(len(result_indices))
        
        while budget_remaining > 0:
            remaining_indices = np.where(~annotated_mask)[0]
            if remaining_indices.size == 0:
                break
            
            # Filter only valid remaining
            valid_remaining_mask = valid_mask[remaining_indices]
            if not np.any(valid_remaining_mask):
                break
            valid_remaining_indices = remaining_indices[valid_remaining_mask]

            # Score remaining valid documents
            # Compute probabilities and uncertainties
            # If predict_proba is expensive, consider ways to batch or vectorize it.
            probs = np.array([clf.predict_proba(feature_list[idx]) for idx in valid_remaining_indices])
            # Uncertainty highest at probs close to 0.5 => uncertainty = -abs(prob - 0.5)
            uncertainties = -np.abs(probs - 0.5)
            
            # Select top K uncertain documents
            k = min(select_size, budget_remaining, len(valid_remaining_indices))
            # argpartition to find the top k uncertainties
            # Since we want the highest uncertainties, we use np.argpartition with [-k:]
            selection_indices = np.argpartition(uncertainties, -k)[-k:]
            selected_indices = valid_remaining_indices[selection_indices]
            
            # Update model with selected documents
            for idx in selected_indices:
                clf.update(feature_list[idx], labels[idx])
            
            annotated_mask[selected_indices] = True
            result_indices.extend(selected_indices)
            budget_remaining -= k
            pbar.update(k)
    
    # Create results
    annotated_df = data_.iloc[result_indices].copy()
    annotated_df['pi'] = annotated_df['annotation']
    
    # Handle remaining documents
    remaining_mask = ~annotated_mask
    remaining_df = data_.iloc[remaining_mask].copy()
    
    if len(remaining_df) > 0:
        # Score remaining documents
        # Only compute for valid features
        remaining_valid_mask = valid_mask[remaining_mask]
        remaining_valid_indices = remaining_df.index[remaining_valid_mask]
        
        scores = np.zeros(len(remaining_df))
        
        for i, idx in enumerate(remaining_df.index):
            if valid_mask[idx]:
                scores[i] = clf.predict_proba(feature_list[idx])
            else:
                scores[i] = 0.0

        remaining_df['pi'] = scores
    else:
        remaining_df['pi'] = []
    
    final_df = pd.concat([annotated_df, remaining_df], ignore_index=True)
    # Binarize predictions
    final_df['pi'] = (final_df['pi'] > 0.5).astype(int)
    return annotated_df, final_df


def cal(data: pd.DataFrame, budget: int, doc_path: str, select_size: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    CAL implementation selecting multiple documents per iteration. This version is modified to handle features and
    predictions similarly to the SAL implementation.

    Args:
        data: DataFrame containing annotations (simulating that we know these annotations)
        budget: Maximum documents to annotate
        doc_path: Path to preprocessed features (dict {doc_id: feature_vector})
        select_size: Number of documents to select in each iteration

    Returns:
        annotated_df: DataFrame with documents that were "annotated" (selected)
        final_df: DataFrame with final predictions (pi) for all documents
    """
    # Load features
    print("Loading preprocessed document features...")
    with open(doc_path, 'rb') as f:
        doc_features = pickle.load(f)
    
    data_ = data.copy()
    doc_ids = data_['doc_id'].values
    labels = data_['annotation'].values
    n_docs = len(doc_ids)

    # Convert doc_features from dict to aligned list
    feature_list = []
    valid_mask = np.zeros(n_docs, dtype=bool)
    for i, d_id in enumerate(doc_ids):
        if d_id in doc_features:
            feature_list.append(doc_features[d_id])
            valid_mask[i] = True
        else:
            feature_list.append(None)
    
    total_docs = n_docs
    clf = OnlineLogisticRegression()
    annotated_mask = np.zeros(n_docs, dtype=bool)
    result_indices = []

    if budget >= total_docs - 1:
        print(f"Annotation budget ({budget}) >= total documents ({total_docs}). Annotating everything.")
        data_['pi'] = data_['annotation']
        # Binarize as done in SAL
        data_['pi'] = (data_['pi'] > 0.5).astype(int)
        return data_[data_['pi'].notna()], data_

    # Initial seed
    initial_seed_size = min(50, n_docs)
    seed_indices = np.random.choice(np.arange(n_docs), initial_seed_size, replace=False)
    for idx in seed_indices:
        if valid_mask[idx]:
            clf.update(feature_list[idx], labels[idx])
            annotated_mask[idx] = True
            result_indices.append(idx)

    budget_remaining = budget - len(result_indices)

    # Main CAL loop
    with tqdm(total=budget, desc='CAL Progress') as pbar:
        pbar.update(len(result_indices))
        
        while budget_remaining > 0:
            # Get unannotated and valid documents
            remaining_indices = np.where(~annotated_mask)[0]
            if remaining_indices.size == 0:
                break
            
            valid_remaining_mask = valid_mask[remaining_indices]
            if not np.any(valid_remaining_mask):
                break
            valid_remaining_indices = remaining_indices[valid_remaining_mask]

            # Score remaining valid documents
            probs = np.array([clf.predict_proba(feature_list[idx]) for idx in valid_remaining_indices])

            # Select top K scoring documents
            k = min(select_size, budget_remaining, len(valid_remaining_indices))
            # We want the top scoring documents
            selection_indices = np.argpartition(probs, -k)[-k:]
            selected_indices = valid_remaining_indices[selection_indices]

            # Update model with selected documents
            for idx in selected_indices:
                clf.update(feature_list[idx], labels[idx])

            annotated_mask[selected_indices] = True
            result_indices.extend(selected_indices)
            budget_remaining -= k
            pbar.update(k)

    # Create results
    annotated_df = data_.iloc[result_indices].copy()
    annotated_df['pi'] = annotated_df['annotation']

    # Handle remaining documents
    remaining_mask = ~annotated_mask
    remaining_df = data_.iloc[remaining_mask].copy()
    
    if len(remaining_df) > 0:
        scores = np.zeros(len(remaining_df))
        for i, idx in enumerate(remaining_df.index):
            if valid_mask[idx]:
                scores[i] = clf.predict_proba(feature_list[idx])
            else:
                scores[i] = 0.0
        remaining_df['pi'] = scores
    else:
        remaining_df['pi'] = []

    final_df = pd.concat([annotated_df, remaining_df], ignore_index=True)
    # Binarize predictions
    final_df['pi'] = (final_df['pi'] > 0.5).astype(int)

    return annotated_df, final_df



def _build_features_from_runs(runs_path: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Builds feature matrix from multiple runs for each document.
    """
    run_files = [f for f in os.listdir(runs_path) if f.startswith("input.")]
    
    # Initialize dictionary to store features
    doc_features = {}
    
    # Process each run file
    for run_file in tqdm(run_files, desc='Processing run files'):
        run_df = pd.read_csv(
            os.path.join(runs_path, run_file),
            sep='\s+',
            header=None,
            names=['topic_id', 'Q0', 'doc_id', 'rank', 'score', 'run_name'],
            dtype={'topic_id': str, 'doc_id': str}
        )
        
        # Normalize scores for this run
        run_df['score'] = (run_df['score'] - run_df['score'].min()) / (run_df['score'].max() - run_df['score'].min())
        
        # Update features for each document
        for _, row in run_df.iterrows():
            key = (row['topic_id'], row['doc_id'])
            if key not in doc_features:
                doc_features[key] = {
                    'scores': [],
                    'ranks': [],
                    'topic_id': row['topic_id'],
                    'doc_id': row['doc_id']
                }
            doc_features[key]['scores'].append(row['score'])
            doc_features[key]['ranks'].append(row['rank'])
    
    # Convert to DataFrame with aggregated features
    features_list = []
    for key, features in doc_features.items():
        features_dict = {
            'topic_id': features['topic_id'],
            'doc_id': features['doc_id'],
            'avg_score': np.mean(features['scores']),
            'max_score': np.max(features['scores']),
            'min_score': np.min(features['scores']),
            'std_score': np.std(features['scores']),
            'avg_rank': np.mean(features['ranks']),
            'best_rank': np.min(features['ranks']),
            'worst_rank': np.max(features['ranks']),
            'rank_std': np.std(features['ranks']),
            'n_appearances': len(features['scores'])
        }
        features_list.append(features_dict)
    
    return pd.DataFrame(features_list)

def select_top_by_pi(data: pd.DataFrame, budget: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Ensure 'pi' is numeric
    data = data.copy()
    data['pi'] = pd.to_numeric(data['pi'], errors='coerce')
    
    # Sort data by 'pi' in descending order
    data_sorted = data.sort_values(by='pi', ascending=False)
    
    # Select the top 'budget' rows
    top_data = data_sorted.head(budget)
    
    # Replace the 'pi' value with the 'annotation' value in the selected data
    top_data_replaced = top_data.copy()
    top_data_replaced['pi'] = top_data_replaced['annotation']
    
    # Prepare Version 1: DataFrame with only the selected data points
    version1 = top_data_replaced[['topic_id', 'doc_id', 'annotation', 'pi']]
    
    # Prepare Version 2: Entire data with selected points' 'pi' replaced
    full_data = data.copy()
    # Identify the indices of the selected data points
    selected_indices = top_data_replaced.index
    # Replace 'pi' with 'annotation' for the selected indices
    full_data.loc[selected_indices, 'pi'] = full_data.loc[selected_indices, 'annotation']
    
    # For non-annotated points (those not in `selected_indices`), convert 'pi' to binary
    non_annotated_mask = ~full_data.index.isin(selected_indices)
    full_data.loc[non_annotated_mask, 'pi'] = (full_data.loc[non_annotated_mask, 'pi'] > 0.7).astype(int)
    
    # Prepare Version 2: Ensure columns are in the same order
    version2 = full_data[['topic_id', 'doc_id', 'annotation', 'pi']]
    
    return version1, version2

def maxmean(runs_path: str, data: pd.DataFrame, budget: int) -> pd.DataFrame:
    """
    Implements MM-NS (MaxMean Non-Stationary) document pooling as described in the paper.
    Each run is treated as a bandit arm, using Beta distributions that only consider the last document.
    """
    # Data preparation steps remain similar
    total_docs = len(data)
    print(f"Total documents in data: {total_docs}")
    data_ = data.copy()
    
    if budget >= total_docs-1:
        print(f"Annotation budget ({budget}) >= total documents ({total_docs}). Annotating everything.")
        data_['pi'] = data_['annotation']
        return data_[data_['pi'].notna()], data_
    
    data_['topic_id'] = data_['topic_id'].astype(str)
    data_['doc_id'] = data_['doc_id'].astype(str)
    available_pairs = set(zip(data_['topic_id'], data_['doc_id']))
    print(f"Number of available topic-doc pairs: {len(available_pairs)}")
    
    # Get list of run files
    run_files = [f for f in os.listdir(runs_path) if f.startswith("input.")]
    print(f"Number of run files found: {len(run_files)}")
    
    # Initialize runs with Beta(1,1) - mean of 0.5 for untried runs
    run_states = {
        run_file: {
            'alpha': 1, 
            'beta': 1,
            'documents': [],
            'last_relevant': None
        }
        for run_file in run_files
    }
    
    # Load run files and count total available documents
    total_run_docs = 0
    for run_file in tqdm(run_files, desc='Loading runs'):
        run_df = pd.read_csv(
            os.path.join(runs_path, run_file),
            sep='\s+',
            header=None,
            names=['topic_id', 'Q0', 'doc_id', 'rank', 'score', 'run_name'],
            dtype={'topic_id': str, 'doc_id': str}
        )
        valid_docs = [
            (topic_id, doc_id) 
            for topic_id, doc_id in zip(run_df['topic_id'], run_df['doc_id'])
            if (topic_id, doc_id) in available_pairs
        ]
        run_states[run_file]['documents'] = valid_docs
        total_run_docs += len(valid_docs)
        
    print(f"Total documents across all runs (including duplicates): {total_run_docs}")
    unique_docs = set()
    for run_file in run_states:
        unique_docs.update(run_states[run_file]['documents'])
    print(f"Number of unique documents across all runs: {len(unique_docs)}")
    
    annotated_pairs = set()
    result_list = []
    budget_used = 0
    current_run = None
    
    # Main MM-NS pooling loop
    with tqdm(total=budget, desc='Annotating documents') as pbar:
        while budget_used < budget:
            # If we have a current run and its last document was relevant, stay with it
            if current_run and run_states[current_run]['last_relevant']:
                selected_run = current_run
            else:
                # Calculate means based on last document only
                run_means = {
                    run_file: params['alpha'] / (params['alpha'] + params['beta'])
                    for run_file, params in run_states.items()
                    if len(params['documents']) > 0
                }
                
                if not run_means:
                    print(f"\nNo more documents available in any run after annotating {budget_used}/{budget} documents")
                    remaining_docs = sum(len(params['documents']) for params in run_states.values())
                    print(f"Remaining documents across all runs: {remaining_docs}")
                    print(f"Size of annotated_pairs: {len(annotated_pairs)}")
                    break
                    
                # Select run with highest mean
                selected_run = max(run_means.items(), key=lambda x: x[1])[0]
            
            current_run = selected_run
            
            # Get next document from selected run
            found_valid_doc = False
            while run_states[selected_run]['documents']:
                topic_id, doc_id = run_states[selected_run]['documents'].pop(0)
                
                if (topic_id, doc_id) in annotated_pairs:
                    continue
                    
                if (topic_id, doc_id) not in available_pairs:
                    continue
                    
                # Found a valid document
                found_valid_doc = True
                
                # Add to annotated set
                annotated_pairs.add((topic_id, doc_id))
                result_list.append({'topic_id': topic_id, 'doc_id': doc_id})
                budget_used += 1
                pbar.update(1)
                
                # Get annotation result
                doc_data = data_[
                    (data_['topic_id'] == topic_id) & 
                    (data_['doc_id'] == doc_id)
                ]
                
                if not doc_data.empty and not pd.isna(doc_data['annotation'].iloc[0]):
                    is_relevant = float(doc_data['annotation'].iloc[0] > 0)
                    
                    if is_relevant:
                        run_states[selected_run]['alpha'] = 2
                        run_states[selected_run]['beta'] = 1
                        run_states[selected_run]['last_relevant'] = True
                    else:
                        run_states[selected_run]['alpha'] = 1
                        run_states[selected_run]['beta'] = 2
                        run_states[selected_run]['last_relevant'] = False
                
                break
            
            # If we didn't find a valid document in this run
            if not found_valid_doc:
                current_run = None
                
            if budget_used >= budget:
                break
    
    # Create final dataset
    pooled_df = pd.DataFrame(result_list)
    merged_df = pd.merge(
        pooled_df,
        data_[['topic_id', 'doc_id', 'annotation']],
        on=['topic_id', 'doc_id'],
        how='inner'
    )
    
    merged_df['pi'] = merged_df['annotation']
    qrels_df = merged_df[['topic_id', 'doc_id', 'annotation', 'pi']].drop_duplicates()
    
    # Handle remaining documents
    remaining_mask = ~data_.set_index(['topic_id', 'doc_id']).index.isin(
        qrels_df.set_index(['topic_id', 'doc_id']).index
    )
    remaining_data_ = data_[remaining_mask].copy()
    remaining_data_['pi'] = remaining_data_['annotation'].copy()
    
    final_data_set = pd.concat([qrels_df, remaining_data_], ignore_index=True)
    
    return qrels_df, final_data_set

def move_to_front_pooling(runs_path: str, data: pd.DataFrame, budget: int) -> pd.DataFrame:
    """
    Implements the move-to-front pooling method for document annotation.

    Args:
        runs_path (str): Path to the directory containing run files.
        data (pd.DataFrame): DataFrame containing documents with 'topic_id', 'doc_id', 'pi', and 'annotation' columns.
        budget (int): Maximum number of annotations allowed.

    Returns:
        qrels_df (pd.DataFrame): DataFrame of pooled annotations with 'topic_id', 'doc_id', 'annotation', and 'pi'.
        final_dataset (pd.DataFrame): Combined DataFrame of annotated and non-annotated documents.
    """
    data_ = data.copy()
    total_docs = len(data_)
    if budget >= total_docs-1:
        print(f"Annotation budget ({budget}) >= total documents ({total_docs}). Annotating everything.")
        data_['pi'] = data_['annotation']
        return data_[data_['pi'].notna()], data_
    
    # Ensure 'topic_id' and 'doc_id' are strings
    data_['topic_id'] = data_['topic_id'].astype(str)
    data_['doc_id'] = data_['doc_id'].astype(str)

    # Build a set of available (topic_id, doc_id) pairs from 'data_' for quick look-up
    available_pairs = set(zip(data_['topic_id'], data_['doc_id']))

    # Get the list of run files starting with "input."
    run_files = [f for f in os.listdir(runs_path) if f.startswith("input.")]

    # Initialize a dictionary to store documents per topic with move-to-front ordering
    topic_docs = {topic: [] for topic in data_['topic_id'].unique()}
    max_rank = 0

    # Iterate over run files to collect documents per topic
    for run_file in tqdm(run_files, desc='Processing run files'):
        # Read the run file into a data_Frame
        run_df = pd.read_csv(
            os.path.join(runs_path, run_file),
            sep='\s+',
            header=None,
            names=['topic_id', 'Q0', 'doc_id', 'rank', 'score', 'run_name'],
            dtype={'topic_id': str, 'doc_id': str}
        )

        # Update max_rank
        max_rank = max(max_rank, run_df['rank'].max())

        # Iterate through each row to build move-to-front lists per topic
        for _, row in run_df.iterrows():
            topic = row['topic_id']
            doc = row['doc_id']
            if doc not in topic_docs[topic]:
                topic_docs[topic].append(doc)

    # Initialize variables
    annotated_pairs = set()
    result_list = []
    budget_used = 0

    # Iterate through each topic and pool documents using move-to-front strategy
    for topic in tqdm(sorted(topic_docs.keys()), desc='Processing topics'):
        if budget_used >= budget:
            break
        for doc in tqdm(topic_docs[topic], desc=f'Topic {topic}', leave=False):
            if budget_used >= budget:
                break
            pair = (topic, doc)
            if pair not in annotated_pairs and pair in available_pairs:
                annotated_pairs.add(pair)
                result_list.append({'topic_id': topic, 'doc_id': doc})
                budget_used += 1

    # Create data_Frame from result_list
    pooled_df = pd.DataFrame(result_list)

    # Merge with 'data' to get the original `pi` values and annotations
    merged_df = pd.merge(
        pooled_df,
        data_[['topic_id', 'doc_id', 'pi', 'annotation']],
        on=['topic_id', 'doc_id'],
        how='left'
    )

    # Function to count in ranges (assuming it's defined elsewhere)
    original_pi_counts, bins = count_in_ranges(merged_df['pi'])

    # Replace 'pi' with 'annotation' values
    merged_df['pi'] = merged_df['annotation']

    # Prepare qrels DataFrame
    qrels_df = merged_df[['topic_id', 'doc_id', 'annotation', 'pi']].drop_duplicates()

    # Identify remaining data not annotated
    remaining_data = data_[~data_.set_index(['topic_id', 'doc_id']).index.isin(qrels_df.set_index(['topic_id', 'doc_id']).index)]
    remaining_counts, _ = count_in_ranges(remaining_data['pi'])

    # Print annotation report
    print(f"\nAnnotation report before handling non-annotated data ({budget_used}/{budget} annotations):")
    for i in range(len(bins) - 1):
        print(f"Range [{bins[i]:.1f}, {bins[i+1]:.1f}): "
              f"Annotated: {original_pi_counts[i]}, Remaining: {remaining_counts[i]}")
    print("\n")
    
    # Handle non-annotated points: If pi > 0.5, set pi = 1
    non_annotated_data = remaining_data.copy()
    non_annotated_data.loc[non_annotated_data['pi'] > 0.5, 'pi'] = 1

    # Combine the annotated and non-annotated data back into the final dataset
    final_dataset = pd.concat([qrels_df, non_annotated_data], ignore_index=True)

    return qrels_df, final_dataset

def depth_k(runs_path: str, data: pd.DataFrame, budget: int) -> pd.DataFrame:
    total_docs = len(data)
    data_ = data.copy()
    if budget >= total_docs-1:
        print(f"Annotation budget ({budget}) >= total documents ({total_docs}). Annotating everything.")
        data_['pi'] = data_['annotation']
        return data_[data_['pi'].notna()], data_
    
    # Ensure 'topic_id' and 'doc_id' are strings
    data_['topic_id'] = data_['topic_id'].astype(str)
    data_['doc_id'] = data_['doc_id'].astype(str)

    # Build a set of available (topic_id, doc_id) pairs from 'data_' for quick look-up
    available_pairs = set(zip(data_['topic_id'], data_['doc_id']))

    # Get the list of run files starting with "input."
    run_files = [f for f in os.listdir(runs_path) if f.startswith("input.")]

    # Initialize a dictionary to store documents per depth
    depth_docs = {}
    max_depth = 0

    # Iterate over run files to collect documents at each depth
    for run_file in tqdm(run_files, desc='Processing run files'):
        # Read the run file into a data_Frame
        run_df = pd.read_csv(
            os.path.join(runs_path, run_file),
            sep='\s+',
            header=None,
            names=['topic_id', 'Q0', 'doc_id', 'rank', 'score', 'run_name'],
            dtype={'topic_id': str, 'doc_id': str}
        )

        # Update max_depth
        max_depth = max(max_depth, run_df['rank'].max())

        # Group by depth (rank)
        for depth in run_df['rank'].unique():
            depth = int(depth)
            if depth not in depth_docs:
                depth_docs[depth] = set()
            # Get documents at this depth
            depth_docs[depth].update(zip(
                run_df.loc[run_df['rank'] == depth, 'topic_id'],
                run_df.loc[run_df['rank'] == depth, 'doc_id']
            ))

    # Sort depths in ascending order
    sorted_depths = sorted(depth_docs.keys())

    # Initialize variables
    annotated_pairs = set()
    result_list = []
    budget_used = 0

    # Iterate over depths
    for depth in tqdm(sorted_depths, desc='Processing depths'):
        if budget_used >= budget:
            break
        # Get documents at current depth
        docs_at_depth = depth_docs[depth]
        # Iterate over documents
        for topic_id, doc_id in tqdm(docs_at_depth, desc=f'Depth {depth}', leave=False):
            if budget_used >= budget:
                break
            if (topic_id, doc_id) not in annotated_pairs and (topic_id, doc_id) in available_pairs:
                annotated_pairs.add((topic_id, doc_id))
                result_list.append({'topic_id': topic_id, 'doc_id': doc_id})
                budget_used += 1

    # Create data_Frame from result_list
    pooled_df = pd.DataFrame(result_list)

    # Merge with 'data' to get the original `pi` values and annotations
    merged_df = pd.merge(
        pooled_df,
        data_[['topic_id', 'doc_id', 'pi', 'annotation']],
        on=['topic_id', 'doc_id'],
        how='left'
    )

    original_pi_counts, bins = count_in_ranges(merged_df['pi'])
        
    merged_df['pi'] = merged_df['annotation']

    qrels_df = merged_df[['topic_id', 'doc_id', 'annotation', 'pi']]

    qrels_df = qrels_df.drop_duplicates()
    
    remaining_data_ = data_[~data_.set_index(['topic_id', 'doc_id']).index.isin(qrels_df.set_index(['topic_id', 'doc_id']).index)]
    remaining_counts, _ = count_in_ranges(remaining_data_['pi'])

    print(f"\nAnnotation report before handling non-annotated data_ ({budget_used}/{budget} annotations):")
    for i in range(len(bins) - 1):
        print(f"Range [{bins[i]:.1f}, {bins[i+1]:.1f}): "
              f"Annotated: {original_pi_counts[i]}, Remaining: {remaining_counts[i]}")
    print("\n")
    
    # Handle non-annotated points: If pi > 0.7, set pi = 1
    non_annotated_data_ = remaining_data_.copy()
    non_annotated_data_.loc[non_annotated_data_['pi'] > 0.5, 'pi'] = 1

    # Combine the annotated and non-annotated data_ back into the final data_set
    final_data_set = pd.concat([qrels_df, non_annotated_data_], ignore_index=True)

    return qrels_df, final_data_set


class GradualLogisticRegression:
    def __init__(self, learning_rate, random_state=None, initial_strength=200, lambda_=1000):
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.beta0 = 0
        self.beta1 = 0  # Start with β1 = 0 to represent p(y|π) = 0.5
        self.alpha = 1  # Prior for Beta distribution
        self.beta = 1   # Prior for Beta distribution
        self.total_weight = initial_strength
        self.lambda_ = lambda_
        np.random.seed(random_state)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def predict_proba(self, pi):
        logistic_prob = self.sigmoid(self.beta0 + self.beta1 * pi)
        
        # Calculate the weight based on the variance of the Beta distribution
        weight = 1.2 - beta.var(self.alpha, self.beta)
        
        # Slow down the transition for the first few hundred iterations
        adj_weight = weight * (1 - np.exp(-self.total_weight / self.lambda_))
        
        return (1 - adj_weight) * pi + adj_weight * logistic_prob
    
    def update_params(self, X, y):
        z = self.beta0 + self.beta1 * X
        y_pred = self.sigmoid(z)
        
        grad_b0 = np.mean(y_pred - y)
        grad_b1 = np.mean((y_pred - y) * X)
        
        self.beta0 -= self.learning_rate * grad_b0
        self.beta1 -= self.learning_rate * grad_b1
        
        # Update Beta distribution parameters
        self.alpha += np.sum(y)
        self.beta += np.sum(1 - y)
    
    def fit(self, X, y):
        self.update_params(X, y)
    
    def sample_next(self, data):
        probs = self.predict_proba(data['pi'])
        distances = np.abs(probs - 0.5)
        idx = np.argmin(distances)
        return data.iloc[idx]


def ours(data, annotation_budget=int , learning_rate= float, batch_size=int, lambda_ = int, random_state=None):
    # Initialize all data as unlabeled
    total_docs = len(data)
    data_ = data.copy()
    if annotation_budget >= total_docs-1:
        print(f"Annotation budget ({annotation_budget}) >= total documents ({total_docs}). Annotating everything.")
        data_['pi'] = data_['annotation']
        return data_[data_['pi'].notna()], data_
    
    unlabeled_pool = data.copy()
    unlabeled_pool['is_labeled'] = False
    
    model = GradualLogisticRegression(
        learning_rate=learning_rate,
        random_state=random_state,
    )
    
    total_annotations = 0
    
    # Use tqdm to track progress of annotations
    for _ in tqdm(range(annotation_budget), desc="Annotating"):
        if total_annotations >= annotation_budget:
            break
        
        # Sample the next point to annotate
        next_sample = model.sample_next(unlabeled_pool[~unlabeled_pool['is_labeled']])
        
        
        unlabeled_pool.loc[next_sample.name, 'is_labeled'] = True
        
        # Update the model with the new labeled point
        model.fit(np.array([next_sample['pi']]), np.array([next_sample['annotation']]))
        
        total_annotations += 1
        
        #if total_annotations % (annotation_budget/20) == 0:
            #print(f"Annotations: {total_annotations}/{annotation_budget}")

    final_dataset = data_.copy()

    # For annotated points, use the actual annotations
    labeled_data = unlabeled_pool[unlabeled_pool['is_labeled']]
    final_dataset.loc[labeled_data.index, 'pi'] = labeled_data['annotation']

    # For non-annotated points, use model predictions
    non_annotated_mask = ~final_dataset.index.isin(labeled_data.index)
    non_annotated_pi = final_dataset.loc[non_annotated_mask, 'pi']
    predictions = model.predict_proba(non_annotated_pi)
    
    # For non-annotated points, use model predictions
    non_annotated_mask = ~final_dataset.index.isin(labeled_data.index)
    non_annotated_pi = final_dataset.loc[non_annotated_mask, 'pi']

    annotated_counts, bins = count_in_ranges(labeled_data['pi'])  # Use the original `pi` for the annotated data
    remaining_counts, _ = count_in_ranges(non_annotated_pi)
    
    print(f"\nAnnotation report before handling non-annotated data ({total_annotations}/{annotation_budget} annotations):")
    for i in range(len(bins) - 1):
        print(f"Range [{bins[i]:.1f}, {bins[i+1]:.1f}): "
              f"Annotated: {annotated_counts[i]}, Remaining: {remaining_counts[i]}")
    print("\n")
    
    final_dataset.loc[non_annotated_mask, 'pi'] = (predictions >= 0.5).astype(int)
    
    # print the mean 
    
    return labeled_data, final_dataset

def ours_groups(data, annotation_budget=int, learning_rate=float, batch_size=int, lambda_=int, random_state=None, n_groups=int):
    # Set random seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Check if we're annotating everything
    total_docs = len(data)
    final_dataset = data.copy()
    if annotation_budget >= total_docs-1:
        print(f"Annotation budget ({annotation_budget}) >= total documents ({total_docs}). Annotating everything.")
        final_dataset['pi'] = final_dataset['annotation']
        return final_dataset[final_dataset['pi'].notna()], final_dataset
    
    # Initialize all data as unlabeled
    unlabeled_pool = data.copy()
    unlabeled_pool['is_labeled'] = False
    
    model = GradualLogisticRegression(
        learning_rate=learning_rate,
        random_state=random_state,
    )
    
    # Ensure n_groups is within valid range
    n_groups = max(1, min(n_groups, len(unlabeled_pool['topic_id'].unique())))
    
    # Group data by topic_id and get group sizes
    topic_groups = unlabeled_pool.groupby('topic_id')
    group_sizes = topic_groups.size().values
    
    # Randomly assign topics to groups, ensuring more balanced distribution
    topic_ids = unlabeled_pool['topic_id'].unique()
    random.shuffle(topic_ids)
    groups = [[] for _ in range(n_groups)]
    group_doc_counts = [0] * n_groups
    for topic_id in topic_ids:
        group_index = min(range(n_groups), key=lambda i: group_doc_counts[i])
        groups[group_index].append(topic_id)
        group_doc_counts[group_index] += len(unlabeled_pool[unlabeled_pool['topic_id'] == topic_id])
    
    # Calculate initial budget per group
    group_budgets = [min(count, annotation_budget // n_groups) for count in group_doc_counts]
    remaining_budget = annotation_budget - sum(group_budgets)
    
    # Distribute remaining budget
    while remaining_budget > 0:
        for i in range(n_groups):
            if group_budgets[i] < group_doc_counts[i] and remaining_budget > 0:
                group_budgets[i] += 1
                remaining_budget -= 1
            if remaining_budget == 0:
                break
    
    total_annotations = 0
    
    # Use tqdm to track progress of annotations
    with tqdm(total=annotation_budget, desc="Annotating") as pbar:
        while total_annotations < annotation_budget:
            for group_index, group in enumerate(groups):
                if group_budgets[group_index] > 0:
                    group_mask = unlabeled_pool['topic_id'].isin(group)
                    group_data = unlabeled_pool[group_mask & ~unlabeled_pool['is_labeled']]
                    
                    if len(group_data) == 0:
                        # Redistribute budget if group is exhausted
                        remaining_budget = group_budgets[group_index]
                        group_budgets[group_index] = 0
                        for i in range(n_groups):
                            if i != group_index and group_budgets[i] < group_doc_counts[i]:
                                transfer = min(remaining_budget, group_doc_counts[i] - group_budgets[i])
                                group_budgets[i] += transfer
                                remaining_budget -= transfer
                            if remaining_budget == 0:
                                break
                        continue
                    
                    # Sample the next point to annotate from the current group
                    next_sample = model.sample_next(group_data)
                    
                    unlabeled_pool.loc[next_sample.name, 'is_labeled'] = True
                    
                    # Update the model with the new labeled point
                    model.fit(np.array([next_sample['pi']]), np.array([next_sample['annotation']]))
                    
                    total_annotations += 1
                    group_budgets[group_index] -= 1
                    pbar.update(1)
                    
                    if total_annotations >= annotation_budget:
                        break
            
            if all(budget == 0 for budget in group_budgets):
                break

    final_dataset = data.copy()

    # For annotated points, use the actual annotations
    labeled_data = unlabeled_pool[unlabeled_pool['is_labeled']]
    final_dataset.loc[labeled_data.index, 'pi'] = labeled_data['annotation']

    # For non-annotated points, use model predictions
    non_annotated_mask = ~final_dataset.index.isin(labeled_data.index)
    non_annotated_pi = final_dataset.loc[non_annotated_mask, 'pi']
    predictions = model.predict_proba(non_annotated_pi)
    
    annotated_counts, bins = count_in_ranges(labeled_data['pi'])  # Use the original `pi` for the annotated data
    remaining_counts, _ = count_in_ranges(non_annotated_pi)
    
    print(f"\nAnnotation report before handling non-annotated data ({total_annotations}/{annotation_budget} annotations):")
    for i in range(len(bins) - 1):
        print(f"Range [{bins[i]:.1f}, {bins[i+1]:.1f}): "
              f"Annotated: {annotated_counts[i]}, Remaining: {remaining_counts[i]}")
    print("\n")
    
    final_dataset.loc[non_annotated_mask, 'pi'] = (predictions >= 0.5).astype(int)
    
    return labeled_data, final_dataset



def run_random_experiment(data: pd.DataFrame, annotation_budget: int, random_state: int = 12345) -> tuple:
    
    total_docs = len(data)
    data_ = data.copy()
    if annotation_budget >= total_docs-1:
        print(f"Annotation budget ({annotation_budget}) >= total documents ({total_docs}). Annotating everything.")
        data_['pi'] = data_['annotation']
        return data_[data_['pi'].notna()], data_
    
    rng = np.random.default_rng(random_state)
    annotated_indices = rng.choice(data_.index, size=annotation_budget, replace=False)
    labeled_data = data_.loc[annotated_indices].copy()
    final_dataset = data_.copy()
    final_dataset.loc[annotated_indices, 'pi'] = data_.loc[annotated_indices, 'annotation']
    non_annotated_mask = ~final_dataset.index.isin(annotated_indices)
    final_dataset.loc[non_annotated_mask, 'pi'] = (final_dataset.loc[non_annotated_mask, 'pi'] >= 0.5).astype(int)
    
    return labeled_data, final_dataset

def run_naive_experiment(data: pd.DataFrame, annotation_budget: int, threshold: float = 0.5, random_state: int = 12345) -> tuple:
    total_docs = len(data)
    data_ = data.copy() 
    if annotation_budget >= total_docs-1:
        print(f"Annotation budget ({annotation_budget}) >= total documents ({total_docs}). Annotating everything.")
        data_['pi'] = data_['annotation']
        return data_[data_['pi'].notna()], data_
    
    rng = np.random.default_rng(random_state)
    annotated_indices = set()
    
    for _ in range(annotation_budget):
        remaining_data = data_[~data_.index.isin(annotated_indices)]
        
        if remaining_data.empty:
            break  # Stop if we've annotated all data points
        
        distances = (remaining_data['pi'] - threshold).abs()
        min_distance = distances.min()
        
        epsilon = 1e-10  # Adjust this value if needed
        candidate_indices = remaining_data[distances <= min_distance + epsilon].index
        
        if len(candidate_indices) == 0:
            candidate_indices = remaining_data.index
        
        idx_to_annotate = rng.choice(candidate_indices)
        annotated_indices.add(idx_to_annotate)
    
    labeled_data = data_.loc[list(annotated_indices)].copy()

    final_dataset = data_.copy()
    final_dataset.loc[list(annotated_indices), 'pi'] = data_.loc[list(annotated_indices), 'annotation']
    
    non_annotated_mask = ~final_dataset.index.isin(annotated_indices)
    final_dataset.loc[non_annotated_mask, 'pi'] = (final_dataset.loc[non_annotated_mask, 'pi'] >= threshold).astype(int)
    
    return labeled_data, final_dataset