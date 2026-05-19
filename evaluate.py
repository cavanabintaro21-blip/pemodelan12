import argparse
import json
from pathlib import Path
from typing import Dict, List

from model_evaluation import (
    evaluate_stance_ground_truth,
    evaluate_topic_model,
    format_confusion_matrix,
    save_evaluation_results,
    print_topic_evaluation,
    TopicEvaluationResults
)
import pandas as pd


def load_topics_from_csv(path: str) -> tuple[Dict[int, List[str]], List[str]]:
    """Load topics and documents from CSV file."""
    df = pd.read_csv(path)

    # Assuming CSV has columns: topic_id, word1, word2, ..., document
    topics = {}
    documents = []

    if 'document' in df.columns:
        documents = df['document'].dropna().tolist()

    # Group by topic_id and collect words
    if 'topic_id' in df.columns:
        for topic_id in df['topic_id'].unique():
            topic_words = []
            topic_data = df[df['topic_id'] == topic_id]

            # Collect all word columns
            word_cols = [col for col in df.columns if col.startswith('word')]
            for _, row in topic_data.iterrows():
                for col in word_cols:
                    if pd.notna(row[col]):
                        topic_words.append(str(row[col]))

            if topic_words:
                topics[int(topic_id)] = list(set(topic_words))  # Remove duplicates

    return topics, documents


def evaluate_topics(ground_truth_path: str, output_path: str) -> None:
    """Evaluate topic model using ground truth data."""
    topics, documents = load_topics_from_csv(ground_truth_path)

    if not topics:
        print("No topics found in the CSV file")
        return

    # Load topic sizes if available
    df = pd.read_csv(ground_truth_path)
    topic_sizes = None
    if 'size' in df.columns:
        topic_sizes = df.groupby('topic_id')['size'].first().tolist()

    # Evaluate topics
    results = evaluate_topic_model(
        topics=topics,
        documents=documents,
        topic_sizes=topic_sizes
    )

    print("=== TOPIC MODEL EVALUATION ===")
    print(print_topic_evaluation(results))

    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"Saved topic evaluation results to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate stance classification or topic modeling using ground truth data."
    )
    parser.add_argument(
        "evaluation_type",
        choices=["stance", "topics"],
        help="Type of evaluation to perform"
    )
    parser.add_argument(
        "ground_truth",
        type=Path,
        help="Path to the ground truth CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the evaluation results JSON",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=["support", "oppose", "neutral"],
        help="Optional label order for evaluation",
    )
    args = parser.parse_args()

    if not args.ground_truth.exists():
        raise FileNotFoundError(f"Ground truth file not found: {args.ground_truth}")

    # Set default output path
    if args.output is None:
        if args.evaluation_type == "stance":
            args.output = Path("stance_evaluation_results.json")
        else:
            args.output = Path("topic_evaluation_results.json")

    if args.evaluation_type == "stance":
        results = evaluate_stance_ground_truth(
            str(args.ground_truth),
            prediction_col="model_prediction",
            label_col="expert_stance",
            label_order=args.labels,
        )

        print("=== STANCE EVALUATION SUMMARY ===")
        print(results.summary_text())
        print("=== CONFUSION MATRIX ===")
        print(format_confusion_matrix(results))
        print("=== CLASSIFICATION REPORT ===")
        print(results.classification_report)

        save_evaluation_results(results, str(args.output))

    elif args.evaluation_type == "topics":
        evaluate_topics(str(args.ground_truth), str(args.output))
    print(f"Saved evaluation results to: {args.output}")


if __name__ == "__main__":
    main()
