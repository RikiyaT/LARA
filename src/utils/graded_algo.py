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
from scipy.sparse import csr_matrix, vstack
from typing import Tuple, Dict, List
from contextlib import nullcontext
from torch.optim import Adam

from collections import Counter

import mmh3 
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from collections import defaultdict
import os
from typing import Tuple, Set, List

import pickle
import numpy as np

import numpy as np
from typing import Optional, Tuple
import random
from tqdm import tqdm
import pandas as pd
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.nn.functional as F

from contextlib import nullcontext

# Logistic Regression-Based Calibration Model
class LogisticOrdinalNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        # One logistic regression per cumulative threshold
        self.logits = nn.Linear(n_classes, n_classes - 1)
        
    def forward(self, x):
        return self.logits(x)

class LogisticOrdinalPredictor:
    def __init__(self, n_classes: int, learning_rate: float,
                 confidence_rate: float = 0.001, batch_size: int = 32,
                 random_state: Optional[int] = None, device=None):
        self.n_classes = n_classes
        self.confidence_rate = confidence_rate
        self.batch_size = batch_size
        
        # Set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model and optimizer
        self.model = LogisticOrdinalNet(n_classes).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize storage as lists
        self.X_train = []
        self.y_train = []
        self.n_samples = 0
        
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
            random.seed(random_state)
    
    def _to_ordinal(self, y):
        """Convert class labels to ordinal targets"""
        y_ord = torch.zeros((len(y), self.n_classes - 1), device=self.device)
        for i, yi in enumerate(y):
            y_ord[i, :yi] = 1
        return y_ord
    
    def _from_ordinal(self, cumulative_probs):
        """Convert cumulative probabilities to class probabilities"""
        probs = torch.zeros((len(cumulative_probs), self.n_classes), device=self.device)
        
        # First class probability
        probs[:, 0] = 1 - cumulative_probs[:, 0]
        
        # Middle class probabilities
        for k in range(1, self.n_classes - 1):
            probs[:, k] = cumulative_probs[:, k-1] - cumulative_probs[:, k]
            
        # Last class probability
        probs[:, -1] = cumulative_probs[:, -1]
        
        return F.softmax(probs, dim=1)
    
    def fit_batch(self, X_batch, y_batch):
        """Train on provided batch"""
        X_batch = np.asarray(X_batch, dtype=np.float64)
        y_batch = np.asarray(y_batch, dtype=np.int64)
        
        X_tensor = torch.FloatTensor(X_batch).to(self.device)
        y_tensor = torch.LongTensor(y_batch).to(self.device)
        
        self.model.train()
        for _ in range(2):  # Number of epochs per batch
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda') if self.device.type == 'cuda' else nullcontext():
                logits = self.model(X_tensor)
                y_ord = self._to_ordinal(y_tensor)
                loss = F.binary_cross_entropy_with_logits(logits, y_ord)
                
                cumulative_probs = torch.sigmoid(logits)
                pred_probs = self._from_ordinal(cumulative_probs)
                transition_loss = 0.1 * F.mse_loss(pred_probs, X_tensor)
                
                total_loss = loss + transition_loss
            
            total_loss.backward()
            self.optimizer.step()
    
    def predict_proba(self, X):
        """Predict with smooth transition from input to learned probabilities"""
        X = np.asarray(X, dtype=np.float64)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda') if self.device.type == 'cuda' else nullcontext():
                logits = self.model(X_tensor)
                cumulative_probs = torch.sigmoid(logits)
                learned_probs = self._from_ordinal(cumulative_probs)
                
                # Move to CPU for numpy operations
                learned_probs = learned_probs.cpu()
                
                # Combine with input probabilities based on confidence
                confidence = min(1.0, self.n_samples * self.confidence_rate)
                
                return (1 - confidence) * X + confidence * learned_probs.numpy()
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
    def add_training_sample(self, X, y):
        """Add new sample and train if batch is full"""
        self.X_train.append(X)
        self.y_train.append(int(y))
        self.n_samples += 1
        
        if self.n_samples % self.batch_size == 0:
            X_batch = np.vstack(self.X_train[-self.batch_size:])
            y_batch = np.array(self.y_train[-self.batch_size:], dtype=np.int64)
            self.fit_batch(X_batch, y_batch)

def ours_graded(data: pd.DataFrame,
                    annotation_budget: int,
                    learning_rate: float = 0.01,
                    confidence_rate: float = 0.0001,
                    batch_size: int = 32,
                    n_groups: int = 3,
                    random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Setup
    prob_cols = [col for col in data.columns if col.startswith('prob_')]
    n_classes = len(prob_cols)
    
    for col in prob_cols:
        data[col] = data[col].astype(np.float64)
    data['annotation'] = data['annotation'].astype(np.int64)
    
    if annotation_budget >= len(data)-1:
        print(f"Annotation budget ({annotation_budget}) >= total documents ({len(data)}). Annotating everything.")
        annotated_data = data[['topic_id', 'doc_id', 'annotation']].copy()
        annotated_data['pi'] = annotated_data['annotation']
        final_data = annotated_data.copy()
        return annotated_data, final_data
    
    # Initialize model with GPU support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LogisticOrdinalPredictor(
        n_classes=n_classes,
        learning_rate=learning_rate,
        confidence_rate=confidence_rate,
        batch_size=batch_size,
        random_state=random_state,
        device=device
    )
    
    # Setup groups
    unlabeled_pool = data.copy()
    unlabeled_pool['is_labeled'] = False
    
    n_groups = max(1, min(n_groups, len(unlabeled_pool['topic_id'].unique())))
    topic_ids = unlabeled_pool['topic_id'].unique()
    random.shuffle(topic_ids)
    
    # Pre-compute topic masks
    topic_masks = {topic_id: unlabeled_pool['topic_id'] == topic_id for topic_id in topic_ids}
    
    groups = [[] for _ in range(n_groups)]
    group_doc_counts = [0] * n_groups
    
    for topic_id in topic_ids:
        group_index = min(range(n_groups), key=lambda i: group_doc_counts[i])
        groups[group_index].append(topic_id)
        group_doc_counts[group_index] += np.sum(topic_masks[topic_id])
    
    group_budgets = [min(count, annotation_budget // n_groups) for count in group_doc_counts]
    remaining_budget = annotation_budget - sum(group_budgets)
    
    while remaining_budget > 0:
        for i in range(n_groups):
            if group_budgets[i] < group_doc_counts[i] and remaining_budget > 0:
                group_budgets[i] += 1
                remaining_budget -= 1
            if remaining_budget == 0:
                break
    
    total_annotations = 0
    annotated_indices = []
    
    # Main annotation loop
    with tqdm(total=annotation_budget, desc="Annotating") as pbar:
        while total_annotations < annotation_budget:
            for group_index, group in enumerate(groups):
                if group_budgets[group_index] > 0:
                    # Get group data efficiently using pre-computed masks
                    group_mask = np.zeros(len(unlabeled_pool), dtype=bool)
                    for topic_id in group:
                        group_mask |= topic_masks[topic_id]
                    group_mask &= ~unlabeled_pool['is_labeled']
                    
                    group_data = unlabeled_pool[group_mask]
                    
                    if len(group_data) == 0:
                        # Redistribute budget
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
                    
                    # Get predictions for all samples in batch
                    X_group = group_data[prob_cols].values
                    with torch.amp.autocast('cuda') if device.type == 'cuda' else nullcontext():
                        predictions = model.predict_proba(X_group)
                    
                    # Vectorized uncertainty calculation
                    predictions_sorted = np.sort(predictions, axis=1)[:, ::-1]
                    uncertainties = predictions_sorted[:, 0] - predictions_sorted[:, 1]
                    prob_diff = np.abs(predictions - X_group).sum(axis=1)
                    uncertainties += 0.1 * prob_diff
                    
                    # Select samples using argsort instead of argpartition
                    n_to_select = min(batch_size, group_budgets[group_index], len(group_data))
                    if n_to_select > 0:
                        uncertain_indices = np.argsort(uncertainties)[:n_to_select]
                        
                        selected_samples = group_data.iloc[uncertain_indices]
                        X_batch = selected_samples[prob_cols].values
                        y_batch = selected_samples['annotation'].values
                        
                        # Batch update model
                        model.fit_batch(X_batch, y_batch)
                        
                        # Update tracking
                        unlabeled_pool.loc[selected_samples.index, 'is_labeled'] = True
                        annotated_indices.extend(selected_samples.index)
                        
                        total_annotations += len(selected_samples)
                        group_budgets[group_index] -= len(selected_samples)
                        pbar.update(len(selected_samples))
                    
                    if total_annotations >= annotation_budget:
                        break
            
            if all(budget == 0 for budget in group_budgets):
                break
    
    # Prepare output
    annotated_data = data.loc[annotated_indices, ['topic_id', 'doc_id', 'annotation']].copy()
    annotated_data['pi'] = annotated_data['annotation']
    
    final_data = data[['topic_id', 'doc_id', 'annotation']].copy()
    
    # Process non-annotated samples in batches
    non_annotated_mask = ~final_data.index.isin(annotated_indices)
    non_annotated_indices = final_data.index[non_annotated_mask]
    
    batch_size_predict = 1024  # Larger batch for final predictions
    final_data['pi'] = np.nan
    
    for i in range(0, len(non_annotated_indices), batch_size_predict):
        batch_indices = non_annotated_indices[i:i + batch_size_predict]
        batch_probs = data.loc[batch_indices, prob_cols].values
        with torch.amp.autocast('cuda') if device.type == 'cuda' else nullcontext():
            final_data.loc[batch_indices, 'pi'] = model.predict(batch_probs)
    
    final_data.loc[annotated_indices, 'pi'] = final_data.loc[annotated_indices, 'annotation']
    
    # Ensure correct column order
    annotated_data = annotated_data[['topic_id', 'doc_id', 'annotation', 'pi']]
    final_data = final_data[['topic_id', 'doc_id', 'annotation', 'pi']]
    
    return annotated_data, final_data



def LLMOnly(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Implements the LLM-Only method for document annotation in a multi-class (graded relevance) setting.

    Args:
        data (pd.DataFrame): DataFrame containing the following columns:
                             - 'topic_id'
                             - 'doc_id'
                             - 'annotation' (true labels: 0, 1, 2, ..., K)
                             - 'prob_0', 'prob_1', ..., 'prob_K' (LLM predicted probabilities for each class)

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - annotated_docs_df: DataFrame with predicted annotations based on LLM probabilities.
            - complete_dataset: Same as annotated_docs_df since no manual annotations are performed.
    """
    modified_data = data.copy()

    # Identify all probability columns (assumes they start with 'prob_')
    prob_columns = [col for col in data.columns if col.startswith('prob_')]

    if not prob_columns:
        raise ValueError("No probability columns found. Ensure columns start with 'prob_'.")

    # Assign the predicted label as the class with the highest probability
    modified_data['pi'] = modified_data[prob_columns].idxmax(axis=1).apply(lambda x: int(x.split('_')[1]))

    # Since LLMOnly does not perform any manual annotations, both DataFrames are identical
    annotated_docs_df = modified_data.copy()
    complete_dataset = modified_data.copy()

    return annotated_docs_df, complete_dataset

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from scipy.sparse import csr_matrix
from typing import Tuple
from numba import njit

@njit
def predict_proba_batch_numba(weights, feature_matrix_data, feature_matrix_indices, feature_matrix_indptr, num_features):
    num_docs = feature_matrix_indptr.size - 1
    probs = np.empty(num_docs, dtype=np.float32)
    for i in range(num_docs):
        start = feature_matrix_indptr[i]
        end = feature_matrix_indptr[i+1]
        dot = 0.0
        for j in range(start, end):
            idx = feature_matrix_indices[j]
            dot += weights[idx]
        probs[i] = 1 / (1 + np.exp(-dot))
    return probs

@njit
def update_batch_numba(weights, feature_matrix_data, feature_matrix_indices, feature_matrix_indptr, y, num_features):
    num_docs = feature_matrix_indptr.size - 1
    for i in range(num_docs):
        start = feature_matrix_indptr[i]
        end = feature_matrix_indptr[i+1]
        dot = 0.0
        for j in range(start, end):
            idx = feature_matrix_indices[j]
            dot += weights[idx]
        prob = 1 / (1 + np.exp(-dot))
        gradient = y[i] - prob
        for j in range(start, end):
            idx = feature_matrix_indices[j]
            weights[idx] += gradient
    return weights

class OnlineLogisticRegression:
    def __init__(self, num_features=100_000_000):
        self.weights = np.zeros(num_features, dtype=np.float32)
    
    def predict_proba_batch(self, feature_matrix):
        return predict_proba_batch_numba(
            self.weights, feature_matrix.data, 
            feature_matrix.indices, feature_matrix.indptr, 
            self.weights.size
        )
    
    def update_batch(self, feature_matrix, y):
        self.weights = update_batch_numba(
            self.weights, feature_matrix.data,
            feature_matrix.indices, feature_matrix.indptr,
            y, self.weights.size
        )

def sal(
    data: pd.DataFrame, 
    budget: int, 
    doc_path: str, 
    select_size: int = 40
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load features
    print("Loading preprocessed document features...")
    with open(doc_path, 'rb') as f:
        doc_features = pickle.load(f)
    
    doc_ids = data['doc_id'].values
    num_docs = len(doc_ids)
    num_features = 100_000_000  # assuming this is fixed
    row_indices = []
    col_indices = []
    data_values = []
    
    for doc_idx, doc_id in enumerate(doc_ids):
        features = doc_features.get(doc_id, [])
        row_indices.extend([doc_idx] * len(features))
        col_indices.extend(features)
        data_values.extend([1] * len(features))
    
    feature_matrix = csr_matrix((data_values, (row_indices, col_indices)), 
                                shape=(num_docs, num_features), dtype=np.float32)
    
    clf = OnlineLogisticRegression(num_features=num_features)
    
    # Determine threshold using k // 2 + 1
    max_annotation = data['annotation'].max()  # e.g., 2
    k = max_annotation + 1                     # e.g., 3 for {0,1,2}
    threshold = k // 2 + 1                     # for {0,1,2}, threshold = 2
    
    # Create binary labels based on the new threshold
    binary_labels = (data['annotation'].values >= threshold).astype(np.float32)
    
    annotated_mask = np.zeros(num_docs, dtype=bool)
    result_indices = []
    
    # Early exit if budget covers all documents
    if budget >= num_docs:
        data['pi'] = data['annotation']
        return data[['topic_id', 'doc_id', 'annotation', 'pi']], \
               data[['topic_id', 'doc_id', 'annotation', 'pi']]
    
    # Randomly select 40 initial seed documents (regardless of relevance)
    initial_seed_count = 40
    if initial_seed_count > num_docs:
        initial_seed_count = num_docs  # In case there are fewer than 40 docs
    seed_indices = np.random.choice(num_docs, initial_seed_count, replace=False)
    
    # Update model with these initial randomly selected seeds
    seed_features = feature_matrix[seed_indices]
    clf.update_batch(seed_features, binary_labels[seed_indices])
    annotated_mask[seed_indices] = True
    result_indices.extend(seed_indices)
    
    budget_remaining = budget - len(result_indices)
    
    # Main loop
    with tqdm(total=budget, desc='SAL Progress') as pbar:
        pbar.update(len(result_indices))
        
        while budget_remaining > 0:
            remaining_indices = np.where(~annotated_mask)[0]
            if len(remaining_indices) == 0:
                break
            
            # Process in batches
            batch_size = 100_000
            all_uncertainties = []
            
            for start in range(0, len(remaining_indices), batch_size):
                end = min(start + batch_size, len(remaining_indices))
                batch_indices = remaining_indices[start:end]
                batch_features = feature_matrix[batch_indices]
                
                probs = clf.predict_proba_batch(batch_features)
                # Uncertainty is higher when probs are close to 0.5
                uncertainties = -np.abs(probs - 0.5)
                all_uncertainties.extend(uncertainties)
            
            uncertainties = np.array(all_uncertainties)
            
            k = min(select_size, budget_remaining, len(uncertainties))
            if k <= 0:
                break
                
            selected_positions = np.argpartition(uncertainties, -k)[-k:]
            selected_indices = remaining_indices[selected_positions]
            
            # Update model with newly annotated documents
            selected_features = feature_matrix[selected_indices]
            clf.update_batch(selected_features, binary_labels[selected_indices])
            
            annotated_mask[selected_indices] = True
            result_indices.extend(selected_indices)
            budget_remaining -= k
            pbar.update(k)
    
    # Prepare output
    annotated_indices = np.array(result_indices)
    annotated_df = data.iloc[annotated_indices].copy()
    annotated_df['pi'] = annotated_df['annotation']
    
    remaining_mask = ~annotated_mask
    remaining_df = data.iloc[remaining_mask].copy()
    if len(remaining_df) > 0:
        remaining_features = feature_matrix[remaining_df.index]
        remaining_probs = clf.predict_proba_batch(remaining_features)
        # Assign label based on threshold
        remaining_df['pi'] = np.where(remaining_probs >= 0.5, threshold, 0)
    else:
        remaining_df['pi'] = []
    
    final_df = pd.concat([annotated_df, remaining_df], ignore_index=True)
    cols = ['topic_id', 'doc_id', 'annotation', 'pi']
    
    return annotated_df[cols], final_df[cols]


def cal(
    data: pd.DataFrame, 
    budget: int, 
    doc_path: str, 
    select_size: int = 40
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lightning fast CAL implementation with Numba acceleration
    """
    # Load features
    print("Loading preprocessed document features...")
    with open(doc_path, 'rb') as f:
        doc_features = pickle.load(f)
    
    # Convert doc_features to sparse matrix
    doc_ids = data['doc_id'].values
    num_docs = len(doc_ids)
    num_features = 100_000_000
    row_indices = []
    col_indices = []
    data_values = []
    
    for doc_idx, doc_id in enumerate(doc_ids):
        features = doc_features.get(doc_id, [])
        row_indices.extend([doc_idx] * len(features))
        col_indices.extend(features)
        data_values.extend([1] * len(features))
    
    feature_matrix = csr_matrix((data_values, (row_indices, col_indices)), 
                                shape=(num_docs, num_features), dtype=np.float32)
    
    # Initialize classifier
    clf = OnlineLogisticRegression(num_features=num_features)
    
    # Compute threshold using k // 2 + 1
    max_annotation = data['annotation'].max()
    k = max_annotation + 1
    threshold = k // 2 + 1  # For {0,1,2}, threshold = 2

    # Create binary labels based on the new threshold
    binary_labels = (data['annotation'].values >= threshold).astype(np.float32)
    
    annotated_mask = np.zeros(num_docs, dtype=bool)
    result_indices = []
    
    # Early exit if budget covers all docs
    if budget >= num_docs:
        data['pi'] = data['annotation']
        return data[['topic_id', 'doc_id', 'annotation', 'pi']], \
               data[['topic_id', 'doc_id', 'annotation', 'pi']]
    
    # Select 40 initial seed documents completely at random (if fewer than 40 docs, take them all)
    initial_seed_count = min(40, num_docs)
    seed_indices = np.random.choice(num_docs, initial_seed_count, replace=False)
    
    # Initial model update
    seed_features = feature_matrix[seed_indices]
    clf.update_batch(seed_features, binary_labels[seed_indices])
    annotated_mask[seed_indices] = True
    result_indices.extend(seed_indices)
    
    budget_remaining = budget - len(result_indices)
    
    # Main loop
    with tqdm(total=budget, desc='CAL Progress') as pbar:
        pbar.update(len(result_indices))
        
        while budget_remaining > 0:
            remaining_indices = np.where(~annotated_mask)[0]
            if len(remaining_indices) == 0:
                break
            
            # Process in batches to manage memory
            batch_size = 100_000
            all_scores = []
            
            for start in range(0, len(remaining_indices), batch_size):
                end = min(start + batch_size, len(remaining_indices))
                batch_indices = remaining_indices[start:end]
                batch_features = feature_matrix[batch_indices]
                
                # In CAL, we use the probability directly as the score
                scores = clf.predict_proba_batch(batch_features)
                all_scores.extend(scores)
            
            scores = np.array(all_scores)
            
            k_docs = min(select_size, budget_remaining, len(scores))
            if k_docs <= 0:
                break
                
            selected_positions = np.argpartition(scores, -k_docs)[-k_docs:]
            selected_indices = remaining_indices[selected_positions]
            
            # Update model with newly annotated documents
            selected_features = feature_matrix[selected_indices]
            clf.update_batch(selected_features, binary_labels[selected_indices])
            
            annotated_mask[selected_indices] = True
            result_indices.extend(selected_indices)
            budget_remaining -= k_docs
            pbar.update(k_docs)
            
            # CAL-specific: attempt to select additional relevant documents
            rel_remaining = np.where((binary_labels == 1) & ~annotated_mask)[0]
            if len(rel_remaining) > 0:
                k_rel = min(k_docs, len(rel_remaining), budget_remaining)
                if k_rel > 0:
                    rand_rel_indices = np.random.choice(rel_remaining, k_rel, replace=False)
                    rel_features = feature_matrix[rand_rel_indices]
                    clf.update_batch(rel_features, np.ones(k_rel, dtype=np.float32))
                    
                    annotated_mask[rand_rel_indices] = True
                    result_indices.extend(rand_rel_indices)
                    budget_remaining -= k_rel
                    pbar.update(k_rel)
    
    # Prepare output
    annotated_indices = np.array(result_indices)
    annotated_df = data.iloc[annotated_indices].copy()
    annotated_df['pi'] = annotated_df['annotation']
    
    remaining_mask = ~annotated_mask
    remaining_df = data.iloc[remaining_mask].copy()
    
    # Get predictions for remaining docs
    if len(remaining_df) > 0:
        remaining_features = feature_matrix[remaining_df.index]
        remaining_probs = clf.predict_proba_batch(remaining_features)
        # Assign label based on threshold
        remaining_df['pi'] = np.where(remaining_probs >= 0.5, threshold, 0)
    else:
        remaining_df['pi'] = []
    
    final_df = pd.concat([annotated_df, remaining_df], ignore_index=True)
    cols = ['topic_id', 'doc_id', 'annotation', 'pi']
    
    return annotated_df[cols], final_df[cols]


def maxmean(runs_path: str, data: pd.DataFrame, budget: int) -> pd.DataFrame:
    """
    Implements MM-NS (MaxMean Non-Stationary) document pooling adapted for graded relevance.
    Each run is treated as a bandit arm, using Beta distributions that only consider the last document.
    For learning purposes, documents are considered "relevant" if their grade is >= threshold,
    where threshold is (k+1)//2 for k being the maximum grade.

    Args:
        runs_path (str): Path to the run files
        data (pd.DataFrame): DataFrame containing graded annotations (0 to k)
        budget (int): Maximum number of documents to annotate

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - annotated_df: DataFrame with annotated documents and their true graded labels
            - final_df: Complete DataFrame with 'pi' predictions
    """
    total_docs = len(data)
    print(f"Total documents in data: {total_docs}")
    data_ = data.copy()
    
    # Determine number of classes and threshold for binary conversion
    k = data_['annotation'].max()
    threshold = (k + 1) // 2  # e.g., for k=3, threshold=2
    
    # Print initial grade distribution
    print("\nInitial grade distribution:")
    for grade in range(k + 1):
        count = (data_['annotation'] == grade).sum()
        print(f"Grade {grade}: {count} documents")
    
    if budget >= total_docs-1:
        print(f"Annotation budget ({budget}) >= total documents ({total_docs}). Annotating everything.")
        data_['pi'] = data_['annotation']
        return data_[['topic_id', 'doc_id', 'annotation', 'pi']], data_[['topic_id', 'doc_id', 'annotation', 'pi']]
    
    data_['topic_id'] = data_['topic_id'].astype(str)
    data_['doc_id'] = data_['doc_id'].astype(str)
    available_pairs = set(zip(data_['topic_id'], data_['doc_id']))
    print(f"Number of available topic-doc pairs: {len(available_pairs)}")
    
    run_files = [f for f in os.listdir(runs_path) if f.startswith("input.")]
    print(f"Number of run files found: {len(run_files)}")
    
    run_states = {
        run_file: {
            'alpha': 1, 
            'beta': 1,
            'documents': [],
            'last_relevant': None,
            'grades': []  # Track grades for debugging
        }
        for run_file in run_files
    }
    
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
            if current_run and run_states[current_run]['last_relevant']:
                selected_run = current_run
            else:
                run_means = {
                    run_file: params['alpha'] / (params['alpha'] + params['beta'])
                    for run_file, params in run_states.items()
                    if len(params['documents']) > 0
                }
                
                if not run_means:
                    print(f"\nNo more documents available in any run after annotating {budget_used}/{budget} documents")
                    print(f"Size of annotated_pairs: {len(annotated_pairs)}")
                    break
                    
                selected_run = max(run_means.items(), key=lambda x: x[1])[0]
            
            current_run = selected_run
            
            found_valid_doc = False
            while run_states[selected_run]['documents']:
                topic_id, doc_id = run_states[selected_run]['documents'].pop(0)
                
                if (topic_id, doc_id) in annotated_pairs:
                    continue
                if (topic_id, doc_id) not in available_pairs:
                    continue
                
                found_valid_doc = True
                annotated_pairs.add((topic_id, doc_id))
                result_list.append({'topic_id': topic_id, 'doc_id': doc_id})
                budget_used += 1
                pbar.update(1)
                
                doc_data = data_[
                    (data_['topic_id'] == topic_id) & 
                    (data_['doc_id'] == doc_id)
                ]
                
                if not doc_data.empty and not pd.isna(doc_data['annotation'].iloc[0]):
                    grade = doc_data['annotation'].iloc[0]
                    run_states[selected_run]['grades'].append(grade)  # Track grade for debugging
                    
                    # Consider document relevant if grade >= threshold
                    is_relevant = float(grade >= threshold)
                    
                    if is_relevant:
                        run_states[selected_run]['alpha'] = 2
                        run_states[selected_run]['beta'] = 1
                        run_states[selected_run]['last_relevant'] = True
                    else:
                        run_states[selected_run]['alpha'] = 1
                        run_states[selected_run]['beta'] = 2
                        run_states[selected_run]['last_relevant'] = False
                
                break
            
            if not found_valid_doc:
                current_run = None
                
            if budget_used >= budget:
                break
    
    # Print annotation statistics
    print("\nAnnotation statistics:")
    for run_file, state in run_states.items():
        if state['grades']:
            grade_counts = Counter(state['grades'])
            print(f"\nRun {run_file}:")
            for grade in range(k + 1):
                count = grade_counts[grade]
                print(f"Grade {grade}: {count} documents")
    
    # Create final datasets
    pooled_df = pd.DataFrame(result_list)
    merged_df = pd.merge(
        pooled_df,
        data_[['topic_id', 'doc_id', 'annotation']],
        on=['topic_id', 'doc_id'],
        how='inner'
    )
    
    merged_df['pi'] = merged_df['annotation']
    annotated_df = merged_df[['topic_id', 'doc_id', 'annotation', 'pi']].drop_duplicates()
    
    # Handle remaining documents
    remaining_mask = ~data_.set_index(['topic_id', 'doc_id']).index.isin(
        annotated_df.set_index(['topic_id', 'doc_id']).index
    )
    remaining_data_ = data_[remaining_mask].copy()
    remaining_data_['pi'] = remaining_data_['annotation'].copy()
    
    final_df = pd.concat([annotated_df, remaining_data_], ignore_index=True)
    final_df = final_df[['topic_id', 'doc_id', 'annotation', 'pi']]
    
    return annotated_df, final_df

def move_to_front_pooling(runs_path: str, data: pd.DataFrame, budget: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_ = data.copy()
    total_docs = len(data_)
    k = data_['annotation'].max()
    
    if budget >= total_docs-1:
        data_['pi'] = data_['annotation']
        return (data_[['topic_id', 'doc_id', 'annotation', 'pi']], 
                data_[['topic_id', 'doc_id', 'annotation', 'pi']])
    
    data_['topic_id'] = data_['topic_id'].astype(str)
    data_['doc_id'] = data_['doc_id'].astype(str)
    available_pairs = set(zip(data_['topic_id'], data_['doc_id']))
    data_lookup = data_.set_index(['topic_id', 'doc_id'])['annotation'].to_dict()
    
    run_files = [f for f in os.listdir(runs_path) if f.startswith("input.")]
    
    all_pairs = []
    for run_file in run_files:
        pairs = pd.read_csv(
            os.path.join(runs_path, run_file),
            sep='\s+',
            header=None,
            names=['topic_id', 'doc_id'],
            usecols=[0, 2],  # Only read topic_id and doc_id columns
            dtype={'topic_id': str, 'doc_id': str}
        )[['topic_id', 'doc_id']].values.tolist()
        all_pairs.extend(pairs)
    
    # Use dict for O(1) lookup to remove duplicates while preserving order
    seen = {}
    unique_pairs = [(t, d) for t, d in all_pairs if not (t, d) in seen and not seen.update({(t, d): 1})]
    
    annotated_pairs = []
    annotations_per_topic = defaultdict(list)
    
    for topic_id, doc_id in unique_pairs:
        if len(annotated_pairs) >= budget:
            break
            
        if (topic_id, doc_id) in available_pairs:
            annotation = data_lookup[(topic_id, doc_id)]
            annotated_pairs.append({
                'topic_id': topic_id, 
                'doc_id': doc_id,
                'annotation': annotation,
                'pi': annotation
            })
            annotations_per_topic[topic_id].append(annotation)
    
    annotated_df = pd.DataFrame(annotated_pairs)
    
    # Ensure we have at least two classes per topic for metrics
    for topic_id in annotations_per_topic:
        unique_annotations = len(set(annotations_per_topic[topic_id]))
        if unique_annotations < 2:
            # Add dummy annotation of different class if needed
            if len(annotated_pairs) < budget:
                for pair in available_pairs:
                    if pair[0] == topic_id and data_lookup[pair] != annotations_per_topic[topic_id][0]:
                        annotated_pairs.append({
                            'topic_id': pair[0],
                            'doc_id': pair[1],
                            'annotation': data_lookup[pair],
                            'pi': data_lookup[pair]
                        })
                        break
    
    annotated_df = pd.DataFrame(annotated_pairs)
    annotated_pairs_set = {(row['topic_id'], row['doc_id']) for row in annotated_pairs}
    remaining_data = data_[~data_.set_index(['topic_id', 'doc_id']).index.isin(annotated_pairs_set)].copy()
    remaining_data['pi'] = remaining_data['annotation']
    
    final_df = pd.concat([
        annotated_df[['topic_id', 'doc_id', 'annotation', 'pi']], 
        remaining_data[['topic_id', 'doc_id', 'annotation', 'pi']]
    ], ignore_index=True)
    
    return annotated_df, final_df

def pool_trec8(runs_path: str, data: pd.DataFrame, budget: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Implements depth-k pooling (TREC-style) adapted for graded relevance annotations.
    Core pooling logic remains the same as it only depends on document ranks.
    
    Args:
        runs_path (str): Path to the directory containing run files
        data (pd.DataFrame): DataFrame containing graded annotations (0 to k)
        budget (int): Maximum number of annotations allowed
    """
    total_docs = len(data)
    data_ = data.copy()
    
    # Determine number of classes and print initial distribution
    k = data_['annotation'].max()
    print("\nInitial grade distribution:")
    for grade in range(k + 1):
        count = (data_['annotation'] == grade).sum()
        print(f"Grade {grade}: {count} documents")
    print("")
    
    if budget >= total_docs-1:
        print(f"Annotation budget ({budget}) >= total documents ({total_docs}). Annotating everything.")
        data_['pi'] = data_['annotation']
        return (data_[['topic_id', 'doc_id', 'annotation', 'pi']], 
                data_[['topic_id', 'doc_id', 'annotation', 'pi']])
    
    data_['topic_id'] = data_['topic_id'].astype(str)
    data_['doc_id'] = data_['doc_id'].astype(str)
    
    # Build available pairs set
    available_pairs = set(zip(data_['topic_id'], data_['doc_id']))
    
    run_files = [f for f in os.listdir(runs_path) if f.startswith("input.")]
    print(f"Number of run files found: {len(run_files)}")
    
    # Initialize depth tracking
    depth_docs = {}
    max_depth = 0
    depth_stats = {}  # Track grades at each depth
    
    # Process run files
    for run_file in tqdm(run_files, desc='Processing run files'):
        run_df = pd.read_csv(
            os.path.join(runs_path, run_file),
            sep='\s+',
            header=None,
            names=['topic_id', 'Q0', 'doc_id', 'rank', 'score', 'run_name'],
            dtype={'topic_id': str, 'doc_id': str}
        )
        
        max_depth = max(max_depth, run_df['rank'].max())
        
        for depth in run_df['rank'].unique():
            depth = int(depth)
            if depth not in depth_docs:
                depth_docs[depth] = set()
                depth_stats[depth] = []
            
            depth_docs[depth].update(zip(
                run_df.loc[run_df['rank'] == depth, 'topic_id'],
                run_df.loc[run_df['rank'] == depth, 'doc_id']
            ))
    
    # Sort depths
    sorted_depths = sorted(depth_docs.keys())
    
    # Initialize tracking variables
    annotated_pairs = set()
    result_list = []
    budget_used = 0
    
    # Main pooling loop
    with tqdm(total=budget, desc='Processing depths') as pbar:
        for depth in sorted_depths:
            if budget_used >= budget:
                break
                
            docs_at_depth = depth_docs[depth]
            for topic_id, doc_id in docs_at_depth:
                if budget_used >= budget:
                    break
                    
                if (topic_id, doc_id) not in annotated_pairs and (topic_id, doc_id) in available_pairs:
                    # Add to annotated set
                    annotated_pairs.add((topic_id, doc_id))
                    result_list.append({'topic_id': topic_id, 'doc_id': doc_id})
                    
                    # Track grade for statistics
                    doc_data = data_[(data_['topic_id'] == topic_id) & (data_['doc_id'] == doc_id)]
                    if not doc_data.empty:
                        grade = doc_data['annotation'].iloc[0]
                        depth_stats[depth].append(grade)
                    
                    budget_used += 1
                    pbar.update(1)
    
    # Print depth statistics
    print("\nGrade distribution by depth:")
    for depth in sorted(depth_stats.keys()):
        if depth_stats[depth]:
            print(f"\nDepth {depth}:")
            grade_counts = Counter(depth_stats[depth])
            for grade in range(k + 1):
                count = grade_counts[grade]
                print(f"Grade {grade}: {count} documents")
    
    # Create final datasets
    pooled_df = pd.DataFrame(result_list)
    merged_df = pd.merge(
        pooled_df,
        data_[['topic_id', 'doc_id', 'annotation']],
        on=['topic_id', 'doc_id'],
        how='inner'
    )
    
    merged_df['pi'] = merged_df['annotation']
    annotated_df = merged_df[['topic_id', 'doc_id', 'annotation', 'pi']].drop_duplicates()
    
    # Handle remaining documents
    remaining_mask = ~data_.set_index(['topic_id', 'doc_id']).index.isin(
        annotated_df.set_index(['topic_id', 'doc_id']).index
    )
    remaining_data = data_[remaining_mask].copy()
    remaining_data['pi'] = remaining_data['annotation']
    
    # Combine annotated and non-annotated data
    final_df = pd.concat([annotated_df, remaining_data[['topic_id', 'doc_id', 'annotation', 'pi']]], 
                        ignore_index=True)
    
    # Print final statistics
    print("\nFinal annotation statistics:")
    print(f"Total documents annotated: {len(annotated_df)}")
    print(f"Remaining documents: {len(remaining_data)}")
    print("\nGrade distribution in annotated documents:")
    for grade in range(k + 1):
        count = (annotated_df['annotation'] == grade).sum()
        print(f"Grade {grade}: {count} documents")
    
    return annotated_df, final_df

def run_random_experiment(data: pd.DataFrame, annotation_budget: int, random_state: int = 12345) -> tuple:
    """
    Random baseline for document selection adapted for graded relevance annotations.
    Non-annotated documents are assigned the class with highest probability from prob_k values.
    
    Args:
        data (pd.DataFrame): DataFrame containing graded annotations (0 to k) and prob_k columns
        annotation_budget (int): Number of documents to annotate
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (annotated_df, final_df) with graded relevance annotations
    """
    total_docs = len(data)
    data_ = data.copy()
    
    # Determine number of classes and probability columns
    prob_cols = [col for col in data_.columns if col.startswith('prob_')]
    k = len(prob_cols) - 1  # Number of classes minus 1
    
    print("\nInitial grade distribution:")
    for grade in range(k + 1):
        count = (data_['annotation'] == grade).sum()
        print(f"Grade {grade}: {count} documents")
    print("")
    
    if annotation_budget >= total_docs-1:
        print(f"Annotation budget ({annotation_budget}) >= total documents ({total_docs}). Annotating everything.")
        data_['pi'] = data_['annotation']
        return data_[['topic_id', 'doc_id', 'annotation', 'pi']], data_[['topic_id', 'doc_id', 'annotation', 'pi']]
    
    # Random selection
    rng = np.random.default_rng(random_state)
    annotated_indices = rng.choice(data_.index, size=annotation_budget, replace=False)
    
    # Create annotated dataset
    labeled_data = data_.loc[annotated_indices, ['topic_id', 'doc_id', 'annotation']].copy()
    labeled_data['pi'] = labeled_data['annotation']
    
    # Create final dataset
    final_dataset = data_[['topic_id', 'doc_id', 'annotation']].copy()
    
    # For annotated documents, use true annotations
    final_dataset.loc[annotated_indices, 'pi'] = data_.loc[annotated_indices, 'annotation']
    
    # For non-annotated documents, predict class with highest probability
    non_annotated_mask = ~final_dataset.index.isin(annotated_indices)
    prob_matrix = data_.loc[non_annotated_mask, prob_cols].values
    final_dataset.loc[non_annotated_mask, 'pi'] = np.argmax(prob_matrix, axis=1)
    
    # Print annotation statistics
    print("\nFinal annotation statistics:")
    print(f"Total documents annotated: {len(labeled_data)}")
    print(f"Remaining documents: {len(final_dataset) - len(labeled_data)}")
    print("\nGrade distribution in annotated documents:")
    for grade in range(k + 1):
        count = (labeled_data['annotation'] == grade).sum()
        print(f"Grade {grade}: {count} documents")
    
    return labeled_data[['topic_id', 'doc_id', 'annotation', 'pi']], final_dataset[['topic_id', 'doc_id', 'annotation', 'pi']]


def run_naive_experiment(
    data: pd.DataFrame,
    annotation_budget: int,
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run LLM trust experiment for graded relevance using probability distributions.
    
    Args:
        data: DataFrame with columns topic_id, doc_id, annotation, prob_0, prob_1, ..., prob_k
        annotation_budget: Number of samples to annotate
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (labeled_data, final_dataset)
    """
    # Setup
    total_docs = len(data)
    data_ = data.copy()
    prob_cols = [col for col in data.columns if col.startswith('prob_')]
    
    # Handle case where budget exceeds data size
    if annotation_budget >= total_docs - 1:
        print(f"Annotation budget ({annotation_budget}) >= total documents ({total_docs}). Annotating everything.")
        annotated_data = data_[['topic_id', 'doc_id', 'annotation']].copy()
        annotated_data['pi'] = annotated_data['annotation']
        final_data = annotated_data.copy()
        return annotated_data, final_data
    
    rng = np.random.default_rng(random_state)
    annotated_indices = set()
    
    # Calculate initial predictions
    probs = data_[prob_cols].values
    # Instead of expected value, use argmax for discrete class prediction
    data_['pi'] = np.argmax(probs, axis=1)
    
    # Main annotation loop
    for _ in range(annotation_budget):
        remaining_data = data_[~data_.index.isin(annotated_indices)]
        
        if remaining_data.empty:
            break
        
        # Calculate uncertainty scores
        probs_remaining = remaining_data[prob_cols].values
        
        # Uncertainty based on entropy and prediction confidence
        entropy = -np.sum(probs_remaining * np.log(probs_remaining + 1e-10), axis=1)
        max_probs = np.max(probs_remaining, axis=1)
        uncertainty_scores = entropy + (1 - max_probs)
        
        # Select most uncertain sample
        max_uncertainty = uncertainty_scores.max()
        epsilon = 1e-10
        candidate_indices = remaining_data[uncertainty_scores >= max_uncertainty - epsilon].index
        
        if len(candidate_indices) == 0:
            candidate_indices = remaining_data.index
            
        idx_to_annotate = rng.choice(candidate_indices)
        annotated_indices.add(idx_to_annotate)
    
    # Prepare output datasets
    labeled_data = data_.loc[list(annotated_indices), 
                           ['topic_id', 'doc_id', 'annotation']].copy()
    labeled_data['pi'] = data_.loc[list(annotated_indices), 'annotation']
    
    final_dataset = data_[['topic_id', 'doc_id', 'annotation']].copy()
    
    # For non-annotated samples, use argmax of probabilities for discrete prediction
    final_dataset['pi'] = np.argmax(data_[prob_cols].values, axis=1)
    final_dataset.loc[list(annotated_indices), 'pi'] = data_.loc[list(annotated_indices), 'annotation']
    
    return labeled_data, final_dataset