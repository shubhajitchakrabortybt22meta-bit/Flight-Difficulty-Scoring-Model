"""Weight optimization utility.

Approach:
 1. Load existing scored (or engineer fresh) dataset.
 2. Select candidate feature columns from config weights keys.
 3. Use a target proxy (departure delay minutes if available) or difficulty_class indicator (Difficult=1 else 0).
 4. Perform K-fold cross-validation optimizing non-negative weights summing to 1 to maximize Pearson correlation (or AUC if binary) between weighted feature blend and target.
 5. Output recommended weights JSON.

Usage:
  python optimize_weights.py --scores final_result_data/flight_difficulty_scores.csv --config config.yaml --k 5

Note: This is a heuristic; in production you'd likely persist baseline weights and review changes.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import difficulty_scoring as ds

try:
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_auc_score
except ImportError:
    KFold = None


def simplex_project(v: np.ndarray) -> np.ndarray:
    # Euclidean projection of v onto probability simplex
    if np.sum(v) == 1 and np.all(v >= 0):
        return v
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u + (1 - cssv) / (np.arange(n) + 1) > 0)[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w


def objective(weights, X, y, is_binary):
    w = simplex_project(weights)
    score = X @ w
    if is_binary:
        # AUC objective; guard against degenerate y
        if len(np.unique(y)) < 2:
            return -np.corrcoef(score, y)[0,1]
        try:
            return -roc_auc_score(y, score)
        except ValueError:
            return -np.corrcoef(score, y)[0,1]
    # Continuous correlation (neg for minimization)
    return -np.corrcoef(score, y)[0,1]


def optimize_weights(X, y, iterations=400, lr=0.05, is_binary=False, seed=42):
    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    w = np.ones(n_features) / n_features
    best_w = w.copy()
    best_obj = objective(best_w, X, y, is_binary)
    for i in range(iterations):
        grad = np.zeros_like(w)
        base = objective(w, X, y, is_binary)
        eps = 1e-5
        for j in range(n_features):
            w_perturb = w.copy()
            w_perturb[j] += eps
            grad[j] = (objective(w_perturb, X, y, is_binary) - base) / eps
        w = w - lr * grad
        w = simplex_project(w)
        cur = objective(w, X, y, is_binary)
        if cur < best_obj:
            best_obj = cur
            best_w = w.copy()
    return best_w, -best_obj


def cross_validate(X, y, k=5, is_binary=False):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    weights_list = []
    scores = []
    for train_idx, test_idx in kf.split(X):
        w, s = optimize_weights(X[train_idx], y[train_idx], is_binary=is_binary)
        weights_list.append(w)
        scores.append(s)
    return np.stack(weights_list).mean(axis=0), float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(description='Optimize feature weights via constrained regression proxy.')
    parser.add_argument('--scores', help='Existing scored CSV (optional). If omitted, will exit.', required=True)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--k', type=int, default=5, help='K-folds for cross validation')
    parser.add_argument('--target', choices=['delay','difficulty'], default='delay', help='Proxy target')
    parser.add_argument('--output', default='final_result_data/optimized_weights.json')
    args = parser.parse_args()

    if not Path(args.scores).exists():
        raise SystemExit('Scores file not found; run pipeline first.')

    config = ds.load_config(args.config)
    df = pd.read_csv(args.scores)

    # Determine target
    if args.target == 'delay' and 'dep_delay_minutes' in df.columns:
        y = df['dep_delay_minutes'].fillna(0).to_numpy()
        is_binary = False
    else:
        # fallback: difficult vs not
        y = (df.get('difficulty_class','').astype(str) == 'Difficult').astype(int).to_numpy()
        is_binary = True

    weight_keys = list(config.get('weights', ds.DEFAULT_WEIGHTS).keys())
    feature_cols = [k for k in weight_keys if k in df.columns]
    if not feature_cols:
        raise SystemExit('No overlapping feature columns found in scores file.')

    # Use scaled versions if available for stable ranges
    X = []
    for col in feature_cols:
        scl = f'{col}_scaled'
        if scl in df.columns:
            X.append(df[scl].to_numpy())
        else:
            # normalize raw column
            series = df[col].astype(float)
            rng = series.max() - series.min()
            if rng == 0:
                X.append(np.zeros(len(series)))
            else:
                X.append((series - series.min()) / rng)
    X = np.vstack(X).T

    best_w, cv_score = cross_validate(X, y, k=args.k, is_binary=is_binary)
    weight_map = {feature_cols[i]: float(best_w[i]) for i in range(len(feature_cols))}

    with open(args.output,'w') as f:
        json.dump({'weights': weight_map, 'cv_score': cv_score, 'target': args.target}, f, indent=2)

    print('Optimized weights written to', args.output)
    print('Cross-validated score (higher better):', cv_score)
    print('Top weights:')
    for k,v in sorted(weight_map.items(), key=lambda kv: kv[1], reverse=True)[:15]:
        print(f'  {k}: {v:.4f}')

if __name__ == '__main__':
    main()
