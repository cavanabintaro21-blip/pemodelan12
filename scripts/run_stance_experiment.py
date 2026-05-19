"""
Run reproducible stance analysis experiments for thesis.

Usage examples:
    python scripts/run_stance_experiment.py --posts posts.csv --comments comments.csv

This script runs both the improved lexicon-based pipeline and the legacy transformer pipeline,
exports results to `results/` and computes basic distribution + optional ground-truth metrics
(if `stance_validation_results.csv` is present).
"""

import os
import argparse
import pandas as pd
from pathlib import Path
import sys, os

# Ensure project root is on sys.path so local modules can be imported when running from /scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stance_analysis import run_stance_analysis


def compute_distribution(df, out_path):
    dist = df['stance'].value_counts(dropna=False).to_dict()
    pd.DataFrame(list(dist.items()), columns=['stance', 'count']).to_csv(out_path, index=False)


def compute_basic_metrics(pred_df, gt_path, out_path):
    if not os.path.exists(gt_path):
        print(f"Ground truth not found at {gt_path}; skipping metrics.")
        return None
    gt = pd.read_csv(gt_path)
    # Try to match by text columns
    left = pred_df.copy()
    left['text_norm'] = left['full_text_comments'].astype(str).str.lower().str.strip()
    gt['text_norm'] = gt['text'].astype(str).str.lower().str.strip()
    merged = pd.merge(gt, left, on='text_norm', suffixes=('_gt', '_pred'))
    if merged.empty:
        print("No matching examples found between predictions and ground truth. Skipping metrics.")
        return None

    label_map = {
        'positive': 'support',
        'pos': 'support',
        'pro': 'support',
        'support': 'support',
        'negative': 'oppose',
        'neg': 'oppose',
        'contra': 'oppose',
        'against': 'oppose',
        'oppose': 'oppose',
        'neutral': 'neutral',
        'netral': 'neutral',
        'none': 'neutral'
    }
    merged['expected'] = merged['expected'].astype(str).str.lower().map(lambda x: label_map.get(x, x))
    labels = ['oppose', 'neutral', 'support']
    results = []
    for label in labels:
        tp = ((merged['expected'] == label) & (merged['stance'] == label)).sum()
        fp = ((merged['expected'] != label) & (merged['stance'] == label)).sum()
        fn = ((merged['expected'] == label) & (merged['stance'] != label)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results.append({'label': label, 'precision': precision, 'recall': recall, 'f1': f1, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)})

    pd.DataFrame(results).to_csv(out_path, index=False)
    return merged


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--posts', default='posts.csv', help='CSV with posts (post_id, clean_text)')
    parser.add_argument('--comments', default='comments.csv', help='CSV with comments (post_id, full_text_comments)')
    parser.add_argument('--ground_truth', default='stance_validation_results.csv', help='Optional ground truth CSV')
    parser.add_argument('--outdir', default='results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--skip_transformer', action='store_true', help='Skip running the legacy transformer analyzer (avoids model download)')
    args = parser.parse_args()

    ensure_dir(args.outdir)

    posts_df = pd.read_csv(args.posts)
    comments_df = pd.read_csv(args.comments)

    # Run improved analyzer
    print('Running improved analyzer...')
    improved_results = run_stance_analysis(posts_df, comments_df, use_improved=True, confidence_threshold=0.45)
    improved_path = os.path.join(args.outdir, 'stance_results_improved.csv')
    improved_results.to_csv(improved_path, index=False)
    print(f'Improved results saved to {improved_path}')

    # Run transformer legacy analyzer (optional)
    transformer_results = None
    if not args.skip_transformer:
        print('Running transformer analyzer (legacy)...')
        try:
            transformer_results = run_stance_analysis(posts_df, comments_df, use_improved=False, confidence_threshold=0.45, batch_size=args.batch_size)
            transformer_path = os.path.join(args.outdir, 'stance_results_transformer.csv')
            transformer_results.to_csv(transformer_path, index=False)
            print(f'Transformer results saved to {transformer_path}')
        except Exception as e:
            print(f'Failed to run transformer analyzer: {e}')
            print('You can retry without --skip_transformer or run on machine with internet to download model.')
    else:
        print('Skipping transformer analyzer as requested (--skip_transformer)')

    # Distributions
    compute_distribution(improved_results, os.path.join(args.outdir, 'dist_improved.csv'))
    if transformer_results is not None:
        compute_distribution(transformer_results, os.path.join(args.outdir, 'dist_transformer.csv'))

    # Compute basic metrics if ground truth provided
    if args.ground_truth and os.path.exists(args.ground_truth):
        print('Computing basic metrics using ground truth...')
        compute_basic_metrics(improved_results, args.ground_truth, os.path.join(args.outdir, 'metrics_improved.csv'))
        if transformer_results is not None:
            compute_basic_metrics(transformer_results, args.ground_truth, os.path.join(args.outdir, 'metrics_transformer.csv'))
        print('Metrics saved to results folder')
    else:
        print('Ground truth not found; only distributions exported.')


if __name__ == '__main__':
    main()
