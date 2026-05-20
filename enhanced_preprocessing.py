"""
Enhanced Text Preprocessing for Stance Analysis
================================================

Preprocesses text while preserving sentiment signals (ALL CAPS, punctuation, etc.)
that are normally lost during standard preprocessing.

Created: 2026-05-17
Last Updated: 2026-05-17
Version: 1.0

Usage:
    from enhanced_preprocessing import preprocess_with_signals, apply_signal_boost
    
    signals = preprocess_with_signals("GOBLOGnya presiden!!!")
    print(signals.clean_text)  # "goblognya presiden"
    print(signals.has_multiple_exclamation)  # 3
"""

import re
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PreprocessingSignals:
    """
    Stores preprocessing signals that indicate emotional intensity.
    
    Attributes:
        has_multiple_caps: Number of ALL CAPS words (≥2 letters)
        has_multiple_exclamation: Number of consecutive ! marks
        has_multiple_question: Number of consecutive ? marks  
        has_repetition: Number of repeated character sequences (gakkkk)
        has_mention: Whether @mention exists
        has_hashtag: Whether #hashtag exists
        has_url: Whether URL exists
        clean_text: The cleaned text for analysis
    """
    has_multiple_caps: int = 0
    has_multiple_exclamation: int = 0
    has_multiple_question: int = 0
    has_repetition: int = 0
    has_mention: bool = False
    has_hashtag: bool = False
    has_url: bool = False
    clean_text: str = ""
    
    def has_intensity_signals(self) -> bool:
        """Check if text has strong intensity signals."""
        return (
            self.has_multiple_caps >= 2 or
            self.has_multiple_exclamation >= 2 or
            self.has_multiple_question >= 2 or
            self.has_repetition >= 1
        )
    
    def intensity_score(self) -> float:
        """
        Calculate intensity score (0.0 to 3.0+).
        
        Returns:
            Score indicating text intensity/emotion
        """
        score = 0.0
        score += self.has_multiple_caps * 0.3
        score += self.has_multiple_exclamation * 0.5
        score += self.has_multiple_question * 0.5
        score += self.has_repetition * 0.3
        return score


def preprocess_with_signals(text: str, preserve_abbrev: bool = True) -> PreprocessingSignals:
    """
    Preprocess text while preserving important emotional/intensity signals.
    
    Args:
        text: Raw text input
        preserve_abbrev: Whether to expand Indonesian abbreviations (gk→gak, dgn→dengan)
        
    Returns:
        PreprocessingSignals object with clean text + signal counts
        
    Example:
        >>> signals = preprocess_with_signals("GOBLOGnya presiden!!!")
        >>> signals.clean_text
        'goblognya presiden'
        >>> signals.has_multiple_exclamation
        3
    """
    
    if not isinstance(text, str):
        text = str(text)
    
    signals = PreprocessingSignals()
    
    # ========================================================================
    # STEP 1: EXTRACT SIGNALS BEFORE CLEANING
    # ========================================================================
    
    # Detect ALL CAPS words (2+ letters)
    caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
    signals.has_multiple_caps = len(caps_words)
    
    # Detect multiple exclamation marks
    exclamation_matches = re.findall(r'!{2,}', text)
    signals.has_multiple_exclamation = (
        sum(len(m) for m in exclamation_matches) if exclamation_matches else 0
    )
    
    # Detect multiple question marks
    question_matches = re.findall(r'\?{2,}', text)
    signals.has_multiple_question = (
        sum(len(m) for m in question_matches) if question_matches else 0
    )
    
    # Detect character repetition (gakkkk, buuuuk, etc)
    repetition_matches = re.findall(r'(.)\1{2,}', text)
    signals.has_repetition = len(repetition_matches)
    
    # Detect mentions, hashtags, URLs
    signals.has_mention = bool(re.search(r'@\w+', text))
    signals.has_hashtag = bool(re.search(r'#\w+', text))
    signals.has_url = bool(re.search(r'http\S+|www\.\S+', text))
    
    # ========================================================================
    # STEP 2: CLEAN TEXT
    # ========================================================================
    
    clean = text
    
    # Remove URLs only; keep @mentions and #hashtags because they carry political targets and stance signals
    clean = re.sub(r'http\S+|www\.\S+', '', clean)
    
    # Preserve abbreviations without aggressively lowercasing the entire comment.
    # Capitalization is kept because it can signal emphasis in political stance.
    if preserve_abbrev:
        clean = re.sub(r'\bgk\b', 'gak', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\bdgn\b', 'dengan', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\bdr\b', 'dari', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\bkrn\b', 'karena', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\bk\b', 'ke', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\bsmg\b', 'semoga', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\bbgt\b', 'banget', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\btdk\b', 'tidak', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\byd\b', 'yang dimaksud', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\butk\b', 'untuk', clean, flags=re.IGNORECASE)
    
    # Reduce multiple punctuation to single (but signal already captured)
    clean = re.sub(r'!{2,}', '!', clean)
    clean = re.sub(r'\?{2,}', '?', clean)
    clean = re.sub(r'\.{2,}', '.', clean)
    
    # Reduce repeated characters (gakkkk → gak)
    clean = re.sub(r'(.)\1{2,}', r'\1\1', clean)  # Keep 2 chars max for distinctiveness
    
    # Remove extra whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    
    # Remove leading/trailing punctuation, but preserve mentions/hashtags
    clean = re.sub(r'^[^@\w#]+|[^@\w#]+$', '', clean)
    
    signals.clean_text = clean
    return signals


def apply_signal_boost(
    sentiment_score: float,
    signals: PreprocessingSignals,
    base_sentiment: str = 'both'
) -> Tuple[float, str]:
    """
    Apply signal-based boosts to sentiment score.
    
    This amplifies sentiment when intensity signals are present.
    E.g., "keren!!!" should be stronger than "keren"
    
    Args:
        sentiment_score: Base sentiment score (usually 0.0 to 1.0)
        signals: PreprocessingSignals object
        base_sentiment: 'positive', 'negative', or 'both'
        
    Returns:
        (adjusted_score, reasoning) tuple
        
    Example:
        >>> score, reason = apply_signal_boost(0.8, signals, 'positive')
        >>> score
        1.0
        >>> reason
        'Multiple ! marks (x3); ALL CAPS words (x2)'
    """
    
    if sentiment_score == 0.0:
        return sentiment_score, "No base sentiment"
    
    adjusted_score = sentiment_score
    reasons = []
    
    # Only apply boosts for strong sentiments (not neutral)
    if base_sentiment in ['positive', 'negative', 'both'] and sentiment_score != 0.0:
        
        # Multiple exclamation marks = high intensity
        if signals.has_multiple_exclamation >= 2:
            multiplier = min(1 + (signals.has_multiple_exclamation * 0.15), 1.5)
            adjusted_score *= multiplier
            reasons.append(f"Multiple ! marks (x{signals.has_multiple_exclamation})")
        
        # Multiple question marks = frustration/sarcasm
        if signals.has_multiple_question >= 2:
            multiplier = min(1 + (signals.has_multiple_question * 0.15), 1.5)
            adjusted_score *= multiplier
            reasons.append(f"Multiple ? marks (x{signals.has_multiple_question})")
        
        # ALL CAPS = emphasis/yelling
        if signals.has_multiple_caps >= 2:
            multiplier = min(1 + (signals.has_multiple_caps * 0.12), 1.4)
            adjusted_score *= multiplier
            reasons.append(f"ALL CAPS words (x{signals.has_multiple_caps})")
        
        # Character repetition = emotion emphasis
        if signals.has_repetition >= 1:
            adjusted_score *= 1.2
            reasons.append(f"Character repetition (x{signals.has_repetition})")
    
    # Clamp to valid range [-1.0, 1.0]
    adjusted_score = max(-1.0, min(1.0, adjusted_score))
    
    reasoning = "; ".join(reasons) if reasons else "Base score"
    
    return adjusted_score, reasoning


def detect_strong_emotion(text: str) -> bool:
    """
    Quick check if text shows strong emotion signals.
    
    Args:
        text: Input text
        
    Returns:
        True if strong emotion signals detected
    """
    signals = preprocess_with_signals(text)
    return signals.has_intensity_signals()


def normalize_text(text: str) -> str:
    """
    Simple text normalization for compatibility.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    signals = preprocess_with_signals(text)
    return signals.clean_text


# ============================================================================
# TESTING & DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ENHANCED TEXT PREPROCESSING - DEMONSTRATION")
    print("=" * 70)
    
    test_cases = [
        "GOBLOGnya presiden @prabowo gk ada obatnya di dunia international!!!...",
        "Program renovasi rumah bikin hati lega",
        "@kompascom Melayani & mengayomi rakyat aja gak becus??",
        "Keren banget ini langkah presiden!!!!",
        "Bangga kami atas kinerja TNI",
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{i}. Original:")
        print(f"   {text}")
        
        signals = preprocess_with_signals(text)
        
        print(f"\n   Cleaned:")
        print(f"   {signals.clean_text}")
        
        print(f"\n   Signals:")
        print(f"   - ALL CAPS words: {signals.has_multiple_caps}")
        print(f"   - Exclamation marks: {signals.has_multiple_exclamation}")
        print(f"   - Question marks: {signals.has_multiple_question}")
        print(f"   - Repetitions: {signals.has_repetition}")
        print(f"   - Has intensity: {signals.has_intensity_signals()}")
        print(f"   - Intensity score: {signals.intensity_score():.2f}")
        
        # Example signal boost
        boost_score, reason = apply_signal_boost(0.85, signals, 'positive')
        print(f"\n   If base sentiment 0.85 → boosted: {boost_score:.2f}")
        print(f"   Reason: {reason}")
        print("-" * 70)
