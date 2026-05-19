"""
Stance Analysis Module
======================

Provides two approaches for stance classification:
1. Original: Transformer-based sentiment analysis (backward compatible)
2. Improved: Lexicon + signal-based (better for Indonesian, RECOMMENDED)

Updated: 2026-05-17
"""

import pandas as pd
from transformers import pipeline
from typing import Optional
import logging

# Try to import improved analyzer (optional dependency)
try:
    from improved_stance_analyzer import ImprovedStanceAnalyzer
    IMPROVED_ANALYZER_AVAILABLE = True
except ImportError:
    IMPROVED_ANALYZER_AVAILABLE = False

logger = logging.getLogger(__name__)


# Final standardized mapping: numeric -> stance label
LABEL_MAPPING = {
    0: "oppose",
    1: "neutral",
    2: "support",
}


def _normalize_label(label: str, id2label: Optional[dict] = None) -> str:
    """Normalize transformer or analyzer output to standardized stance labels.

    Returns one of: 'support', 'oppose', 'neutral' (all lowercase).
    """
    label_orig = str(label)
    label = label_orig.upper()
    if label.startswith("LABEL_") and id2label is not None:
        try:
            numeric = int(label.replace("LABEL_", ""))
            return LABEL_MAPPING.get(numeric, str(label_orig).lower())
        except Exception:
            pass

    if any(token in label for token in ["NEG", "AGAINST", "CONTRA", "TIDAK", "NO"]):
        return "oppose"
    if any(token in label for token in ["NEU", "NET", "NEUTRAL"]):
        return "neutral"
    if any(token in label for token in ["POS", "FAVOR", "FOR", "SUPPORT", "SETUJU"]):
        return "support"
    return str(label_orig).lower()


def run_stance_analysis_improved(
    comments_df: pd.DataFrame,
    confidence_threshold: float = 0.45,
    posts_df: Optional[pd.DataFrame] = None,
    use_signals: bool = True,
    use_sarcasm_detection: bool = True,
) -> pd.DataFrame:
    """
    Run improved lexicon-based stance analysis.
    
    RECOMMENDED: Better for Indonesian political discourse with slang/sarcasm detection.
    
    Args:
        comments_df: DataFrame with 'full_text_comments' column
        confidence_threshold: Minimum confidence for non-neutral classification
        use_signals: Apply intensity signal boosts (CAPS, punctuation)
        use_sarcasm_detection: Detect sarcasm patterns
        
    Returns:
        DataFrame with 'stance' and 'stance_confidence' columns added
    """
    
    if not IMPROVED_ANALYZER_AVAILABLE:
        logger.warning("⚠️ Improved analyzer not available. Install required files.")
        return comments_df.copy()
    
    if comments_df.empty:
        return comments_df.copy()
    
    comments_df = comments_df.copy()
    
    # Initialize analyzer
    analyzer = ImprovedStanceAnalyzer(
        confidence_threshold=confidence_threshold,
        use_signals=use_signals,
        use_sarcasm_detection=use_sarcasm_detection,
        debug=False,
    )
    
    # Determine comment column name
    comment_col = 'full_text_comments' if 'full_text_comments' in comments_df.columns else 'clean_comments'

    # Build post context mapping if posts_df provided
    post_texts = {}
    if posts_df is not None and 'post_id' in posts_df.columns and 'clean_text' in posts_df.columns:
        post_texts = posts_df.set_index('post_id')['clean_text'].to_dict()
    
    if comment_col not in comments_df.columns:
        logger.warning(f"Comment column not found. Available: {comments_df.columns.tolist()}")
        return comments_df
    
    # Add columns
    comments_df['stance'] = 'Neutral'
    comments_df['stance_confidence'] = 0.0
    
    # Analyze each comment
    for idx, row in comments_df.iterrows():
        comment_text = str(row[comment_col] or "")
        post_id = str(row.get('post_id', '')) if 'post_id' in comments_df.columns else ''
        post_context = post_texts.get(post_id, "")
        if len(comment_text.strip()) > 0:
            stance, confidence, reasoning = analyzer.analyze(comment_text, post_context)
            # Normalize analyzer outputs to standardized labels
            stance = _normalize_label(stance)
            comments_df.at[idx, 'stance'] = stance
            comments_df.at[idx, 'stance_confidence'] = confidence
    
    logger.info(f"✓ Improved stance analysis complete: {len(comments_df)} comments analyzed")
    return comments_df


def run_stance_analysis(
    posts_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    batch_size: int = 32,
    confidence_threshold: float = 0.45,
    use_improved: bool = False,
) -> pd.DataFrame:
    """
    Perform stance analysis on comments using the parent post context.
    
    Args:
        posts_df: DataFrame with posts data
        comments_df: DataFrame with comments data
        model_name: Transformer model name (ignored if use_improved=True)
        batch_size: Batch size for transformer
        confidence_threshold: Confidence threshold for classification
        use_improved: Use improved lexicon-based analyzer (RECOMMENDED!)
        
    Returns:
        DataFrame with stance analysis results
    """
    
    if comments_df.empty:
        return comments_df.copy()
    
    # ========================================================================
    # IMPROVED ANALYZER (RECOMMENDED)
    # ========================================================================
    if use_improved:
        logger.info("Using IMPROVED lexicon-based stance analyzer...")
        return run_stance_analysis_improved(
            comments_df,
            confidence_threshold=confidence_threshold,
            posts_df=posts_df,
            use_signals=True,
            use_sarcasm_detection=True,
        )
    
    # ========================================================================
    # ORIGINAL TRANSFORMER-BASED (LEGACY)
    # ========================================================================
    logger.info("Using ORIGINAL transformer-based stance analyzer...")
    
    comments_df = comments_df.copy()
    comments_df["stance"] = "Neutral"
    comments_df["stance_confidence"] = 0.0

    model = pipeline("sentiment-analysis", model=model_name)
    id2label = getattr(model.model.config, "id2label", None)

    post_texts = posts_df.set_index("post_id")["clean_text"].to_dict()
    inputs = []
    
    comment_col = 'full_text_comments' if 'full_text_comments' in comments_df.columns else 'clean_comments'
    
    for record in comments_df.itertuples(index=False):
        post_id = getattr(record, 'post_id', '')
        post_text = post_texts.get(str(post_id), "")
        comment_text = str(getattr(record, comment_col, "") or "")
        inputs.append(f"Post: {post_text} \nComment: {comment_text}")

    results = model(inputs, batch_size=batch_size)
    for idx, prediction in enumerate(results):
        label = prediction.get("label", "Neutral")
        score = float(prediction.get("score", 0.0))
        stance = _normalize_label(label, id2label)
        if score < confidence_threshold and stance != "Neutral":
            stance = "Neutral"
        comments_df.at[idx, "stance"] = stance
        comments_df.at[idx, "stance_confidence"] = score

    return comments_df
