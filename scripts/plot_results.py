"""
Generate visualizations for stance analysis results:
 - distribution bar chart
 - confusion matrix heatmap

Saves images to `results/gt_experiment/figs/`.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def plot_distribution(pred_path, out_dir):
    df = pd.read_csv(pred_path)
    counts = df['stance'].value_counts()
    fig, ax = plt.subplots(figsize=(6,4))
    counts.reindex(['oppose', 'neutral', 'support']).plot(kind='bar', color=['#d62728', '#7f7f7f', '#2ca02c'], ax=ax)
    ax.set_title('Stance Distribution (Improved Analyzer)')
    ax.set_ylabel('Count')
    ax.set_xlabel('Stance')
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'distribution_improved.png')
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(cm_path, out_dir):
    cm = pd.read_csv(cm_path, index_col=0)
    labels = [c.replace('pred_','') for c in cm.columns]
    data = cm.values
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(data, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Show all ticks and label them
    ax.set(xticks=np.arange(data.shape[1]), yticks=np.arange(data.shape[0]), xticklabels=labels, yticklabels=[r.replace('actual_','') for r in cm.index], ylabel='Actual', xlabel='Predicted', title='Confusion Matrix')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = data.max() / 2.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, format(int(data[i, j]), fmt), ha="center", va="center", color="white" if data[i, j] > thresh else "black")

    fig.tight_layout()
    out_path = os.path.join(out_dir, 'confusion_matrix_heatmap.png')
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    out_root = 'results/gt_experiment'
    figs_dir = os.path.join(out_root, 'figs')
    ensure_dir(figs_dir)

    pred_path = os.path.join(out_root, 'stance_results_improved.csv')
    cm_path = os.path.join(out_root, 'confusion_matrix.csv')

    if not os.path.exists(pred_path) or not os.path.exists(cm_path):
        print('Required files not found in', out_root)
        return

    dpath = plot_distribution(pred_path, figs_dir)
    cpath = plot_confusion_matrix(cm_path, figs_dir)

    print('Saved distribution plot to', dpath)
    print('Saved confusion matrix heatmap to', cpath)


if __name__ == '__main__':
    main()
