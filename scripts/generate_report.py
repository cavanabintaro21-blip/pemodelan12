"""
Generate evaluation report (metrics, confusion matrix, error examples)

Usage:
    python scripts/generate_report.py --pred results/gt_experiment/stance_results_improved.csv \
        --gt stance_validation_results.csv --out results/gt_experiment
"""

import argparse
import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def normalize_text(s):
    return str(s).lower().strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True)
    parser.add_argument('--gt', required=True)
    parser.add_argument('--out', default='results/gt_experiment')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    pred = pd.read_csv(args.pred)
    gt = pd.read_csv(args.gt)

    pred['text_norm'] = pred['full_text_comments'].astype(str).apply(normalize_text)
    gt['text_norm'] = gt['text'].astype(str).apply(normalize_text)

    merged = pd.merge(gt, pred, on='text_norm', how='inner', suffixes=('_gt', '_pred'))
    if merged.empty:
        print('No matches between prediction and ground truth; aborting report generation')
        return

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
    y_true = merged['expected']
    y_pred = merged['stance'].astype(str).str.lower()
    labels = ['oppose', 'neutral', 'support']

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    metrics = pd.DataFrame({
        'label': labels,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    })
    metrics_path = os.path.join(args.out, 'detailed_metrics.csv')
    metrics.to_csv(metrics_path, index=False)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f'actual_{l}' for l in labels], columns=[f'pred_{l}' for l in labels])
    cm_path = os.path.join(args.out, 'confusion_matrix.csv')
    cm_df.to_csv(cm_path)

    # Error examples
    errors = merged[merged['expected'] != merged['stance']].copy()
    errors_path = os.path.join(args.out, 'error_examples.csv')
    errors.to_csv(errors_path, index=False)

    # Summary markdown
    summary = []
    summary.append('# Evaluation Report')
    summary.append('')
    summary.append('## Metrics')
    summary.append(metrics.to_string(index=False))
    summary.append('')
    summary.append('## Confusion Matrix')
    summary.append(cm_df.to_string())
    summary.append('')
    summary.append(f'- Total matched examples: {len(merged)}')
    summary.append(f'- Total errors: {len(errors)}')
    summary.append('')
    summary.append('## Top 10 Error Examples')
    for _, r in errors.head(10).iterrows():
        summary.append(f"- Text: {r['text'][:240]}")
        summary.append(f"  - Expected: {r['expected']} | Predicted: {r['stance']} ({r.get('stance_confidence', '')})")
        summary.append('')

    report_path = os.path.join(args.out, 'REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))

    print('Report generated:')
    print(' -', metrics_path)
    print(' -', cm_path)
    print(' -', errors_path)
    print(' -', report_path)


if __name__ == '__main__':
    main()
