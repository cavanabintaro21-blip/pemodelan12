"""
Improved Stance Analyzer for Indonesian Political Discourse
===========================================================

Combines lexicon-based analysis with signal preservation and context awareness.

Features:
- Lexicon-based sentiment detection (400+ words)
- Intensity signal preservation (CAPS, punctuation, repetition)
- Sarcasm pattern detection
- Context-aware post-level analysis
- Fallback to conservative neutral classification

Created: 2026-05-17
Last Updated: 2026-05-17
Version: 1.0

Usage:
    from improved_stance_analyzer import ImprovedStanceAnalyzer
    
    analyzer = ImprovedStanceAnalyzer()
    stance, confidence, reasoning = analyzer.analyze("Menteri paling gak becus")
    print(f"Stance: {stance} ({confidence:.2f})")
"""

import re
from typing import Tuple, List, Dict
import logging

from indonesian_stance_lexicon import (
    INDONESIAN_NEGATIVE_LEXICON,
    INDONESIAN_POSITIVE_LEXICON,
    INTENSITY_BOOSTERS,
    INTENSITY_REDUCERS,
    NEGATION_WORDS,
    SARCASM_PATTERNS,
    RHETORICAL_MARKERS,
    POLITICAL_CRITICISM_KEYWORDS,
    get_sentiment_score,
    find_sentiment_words,
    detect_sarcasm_pattern,
    detect_rhetorical_question,
)
from enhanced_preprocessing import (
    preprocess_with_signals,
    apply_signal_boost,
    PreprocessingSignals,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AGREEMENT_PATTERNS = [
    'betul sekali',
    'benar sekali',
    'nah ini benar',
    'akhirnya ada yang berani bicara',
    'setuju',
    'sangat setuju',
    'setuju banget',
    'iya',
    'amin',
    'mantap',
    'ini benar',
    'pas banget',
    'tepat sekali',
]

DISAGREEMENT_PATTERNS = [
    'tidak setuju',
    'gak setuju',
    'nggak setuju',
    'malah',
    'justru',
    'bukan',
    'jangan',
    'tidak benar',
    'salah',
]


class ImprovedStanceAnalyzer:
    """
    Improved stance analyzer using lexicon + signal preservation.
    
    Attributes:
        confidence_threshold: Minimum confidence for non-neutral classification
        use_signals: Whether to apply intensity signal boosts
        use_sarcasm_detection: Whether to detect sarcasm patterns
        debug: Whether to log debug information
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.45,
        use_signals: bool = True,
        use_sarcasm_detection: bool = True,
        debug: bool = False,
    ):
        """
        Initialize the analyzer.
        
        Args:
            confidence_threshold: Minimum confidence for non-neutral results
            use_signals: Apply intensity signal boosts
            use_sarcasm_detection: Detect sarcasm patterns
            debug: Print debug information
        """
        self.confidence_threshold = confidence_threshold
        self.use_signals = use_signals
        self.use_sarcasm_detection = use_sarcasm_detection
        self.debug = debug
        self.negative_lexicon = INDONESIAN_NEGATIVE_LEXICON
        self.positive_lexicon = INDONESIAN_POSITIVE_LEXICON
    
    def analyze(
        self,
        text: str,
        post_context: str = "",
    ) -> Tuple[str, float, str]:
        """
        Analyze stance of text.
        
        Args:
            text: Comment/response text to analyze
            post_context: Optional parent post text for context
            
        Returns:
            (stance, confidence, reasoning) tuple
                stance: 'support', 'oppose', or 'neutral'
                confidence: 0.0 to 1.0
                reasoning: Brief explanation
        """
        
        if not text or not isinstance(text, str):
            return 'neutral', 0.0, "Empty or invalid text"
        signals = preprocess_with_signals(text)
        clean_text = signals.clean_text
        
        if self.debug:
            logger.info(f"Original: {text}")
            logger.info(f"Cleaned: {clean_text}")
            logger.info(f"Signals: caps={signals.has_multiple_caps}, "
                       f"!={signals.has_multiple_exclamation}, "
                       f"?={signals.has_multiple_question}")
        
        clean_text_lower = clean_text.lower()

        # Step 2: Check for special patterns first (high-confidence indicators)
        special_result = self._check_special_patterns(text, clean_text_lower)
        if special_result is not None:
            return special_result

        # Step 3: Score lexicon matches
        neg_score, neg_words = self._score_lexicon_words(clean_text_lower, 'negative')
        pos_score, pos_words = self._score_lexicon_words(clean_text_lower, 'positive')
        neg_score = self._apply_intensity_modifiers(clean_text_lower, neg_score)
        pos_score = self._apply_intensity_modifiers(clean_text_lower, pos_score)
        
        # Step 5: Apply signal boosts
        if self.use_signals:
            neg_score_boosted, neg_reason = apply_signal_boost(neg_score, signals, 'negative')
            pos_score_boosted, pos_reason = apply_signal_boost(pos_score, signals, 'positive')
            neg_score = neg_score_boosted
            pos_score = pos_score_boosted
        
        # Step 6: Apply negation handling
        neg_score, pos_score = self._handle_negation(clean_text_lower, neg_score, pos_score)
        
        # Step 7: Determine stance
        stance, confidence = self._determine_stance(neg_score, pos_score)
        
        # Step 8: Apply context override if available
        if post_context:
            stance, confidence = self._apply_post_context(
                stance, confidence, text, post_context
            )
        
        # Build reasoning
        reasoning = self._build_reasoning(
            stance, confidence, neg_score, pos_score, neg_words, pos_words
        )
        
        return stance, confidence, reasoning
    
    def _check_special_patterns(
        self,
        original_text: str,
        clean_text: str,
    ) -> Tuple[str, float, str] | None:
        """
        Check for high-confidence pattern matches.
        
        Args:
            original_text: Original text (for patterns that need caps/punctuation)
            clean_text: Cleaned text
            
        Returns:
            (stance, confidence, reasoning) if pattern matches, else None
        """
        
        # 1. Check for sarcasm (usually negative)
        if self.use_sarcasm_detection and detect_sarcasm_pattern(original_text):
            words = find_sentiment_words(clean_text, 'negative')
            if words['negative']:  # Has negative words + sarcasm = very negative
                return 'oppose', 0.90, "Sarcasm pattern detected + negative words"
            else:
                # Pure sarcasm without negative words = likely criticism
                return 'oppose', 0.80, "Passive-aggressive sarcasm detected"
        
        # 2. Check for rhetorical questions (usually negative/critical)
        if detect_rhetorical_question(original_text):
            words = find_sentiment_words(clean_text, 'negative')
            if words['negative']:
                return 'oppose', 0.88, "Rhetorical question + negative sentiment"
        
        # 3. Check for political corruption accusations (very strong negative)
        corruption_count = sum(
            1 for keyword in POLITICAL_CRITICISM_KEYWORDS
            if keyword in clean_text
        )
        if corruption_count >= 2:  # Multiple corruption keywords
            return 'oppose', 0.92, f"Multiple corruption indicators ({corruption_count})"
        
        return None
    
    def _score_lexicon_words(
        self,
        text: str,
        polarity: str,
    ) -> Tuple[float, List[str]]:
        """
        Score text based on lexicon word matches.
        
        Args:
            text: Cleaned text
            polarity: 'negative' or 'positive'
            
        Returns:
            (score, words_found) tuple
        """
        
        lexicon = (
            self.negative_lexicon if polarity == 'negative'
            else self.positive_lexicon
        )
        
        total_score = 0.0
        found_words = []
        
        for word, sentiment_value in lexicon.items():
            if word in text:
                total_score += abs(sentiment_value)
                found_words.append(word)
        
        # Average score across found words
        if found_words:
            avg_score = total_score / len(found_words)
        else:
            avg_score = 0.0
        
        return avg_score, found_words
    
    def _apply_intensity_modifiers(self, text: str, base_score: float) -> float:
        """
        Apply intensity boosters and reducers.
        
        E.g., "sangat baik" > "baik"
        
        Args:
            text: Cleaned text
            base_score: Base sentiment score
            
        Returns:
            Modified score
        """
        
        score = base_score
        
        # Apply boosters
        for word, multiplier in INTENSITY_BOOSTERS.items():
            if word in text:
                score *= multiplier
        
        # Apply reducers
        for word, multiplier in INTENSITY_REDUCERS.items():
            if word in text:
                score *= multiplier
        
        # Clamp to [0, 1]
        return min(1.0, score)
    
    def _handle_negation(
        self,
        text: str,
        neg_score: float,
        pos_score: float,
    ) -> Tuple[float, float]:
        """
        Handle negation (tidak, gak, etc).
        
        E.g., "tidak setuju" → flip positive to negative
        E.g., "tidak bagus" → negate positive sentiment
        
        Args:
            text: Cleaned text
            neg_score: Negative sentiment score
            pos_score: Positive sentiment score
            
        Returns:
            (modified_neg_score, modified_pos_score)
        """
        
        # Check if text contains negation
        has_negation = any(neg_word in text for neg_word in NEGATION_WORDS)
        
        if not has_negation:
            return neg_score, pos_score
        
        # If text has negation + positive words → FLIP to negative
        if pos_score > 0.0 and neg_score == 0.0:
            neg_score = pos_score * 0.8  # Transfer positive score to negative
            pos_score = 0.0
        elif pos_score > 0.0:
            pos_score *= 0.5  # Reduce positive by half
        
        # If text has negation + negative words → amplify
        if neg_score > 0.0:
            neg_score *= 1.3  # Amplify negative by 30%
        
        return neg_score, pos_score
    
    def _determine_stance(self, neg_score: float, pos_score: float) -> Tuple[str, float]:
        """
        Determine final stance from scores.
        
        Args:
            neg_score: Negative sentiment score (0-1)
            pos_score: Positive sentiment score (0-1)
            
        Returns:
            (stance, confidence) tuple
        """
        
        # Both scores below threshold → neutral
        if neg_score < self.confidence_threshold and pos_score < self.confidence_threshold:
            return 'neutral', max(max(neg_score, pos_score), 0.3)
        
        # Negative wins
        if neg_score > pos_score:
            return 'oppose', min(neg_score, 1.0)
        
        # Positive wins
        if pos_score > neg_score:
            return 'support', min(pos_score, 1.0)
        
        # Tie → neutral
        return 'neutral', max(neg_score, pos_score)
    
    def _apply_post_context(
        self,
        stance: str,
        confidence: float,
        comment_text: str,
        post_text: str,
    ) -> Tuple[str, float]:
        """
        Apply post context to adjust stance if needed.
        
        E.g., if post is positive and comment says "tapi ini hanya untuk...",
        comment is actually criticism = negative
        
        Args:
            stance: Current predicted stance
            confidence: Current confidence
            comment_text: Comment text
            post_text: Parent post text
            
        Returns:
            (adjusted_stance, adjusted_confidence)
        """
        
        # Check if post is positive
        post_pos_words = find_sentiment_words(post_text.lower(), 'positive')
        post_is_positive = len(post_pos_words['positive']) >= 2
        
        # Check if comment contradicts post
        contradiction_words = ['tapi', 'namun', 'hanya', 'cuma', 'tetapi', 'padahal']
        has_contradiction = any(word in comment_text.lower() for word in contradiction_words)
        
        # If post is positive but comment has contradiction → likely negative
        if post_is_positive and has_contradiction:
            return 'oppose', 0.75

        post_stance = self._infer_post_stance(post_text)
        if post_stance != 'neutral':
            comment_lower = comment_text.lower()
            if any(pattern in comment_lower for pattern in DISAGREEMENT_PATTERNS):
                opposite = 'oppose' if post_stance == 'support' else 'support'
                return opposite, max(confidence, 0.72)
            if any(pattern in comment_lower for pattern in AGREEMENT_PATTERNS):
                return post_stance, max(confidence, 0.72)
            if stance == 'neutral' and len(comment_lower.strip()) <= 30:
                if any(pattern in comment_lower for pattern in AGREEMENT_PATTERNS):
                    return post_stance, max(confidence, 0.72)

        return stance, confidence
    
    def _infer_post_stance(self, post_text: str) -> str:
        """
        Infer parent post stance from post text using lexicon scores.
        """
        if not post_text or not isinstance(post_text, str):
            return 'neutral'

        post_lower = post_text.lower()
        post_sentiments = find_sentiment_words(post_lower, 'both')
        post_pos_score = sum(score for _, score in post_sentiments['positive'])
        post_neg_score = sum(abs(score) for _, score in post_sentiments['negative'])

        if post_pos_score >= post_neg_score and post_pos_score >= 0.5:
            return 'support'
        if post_neg_score > post_pos_score and post_neg_score >= 0.5:
            return 'oppose'
        return 'neutral'

    def _build_reasoning(
        self,
        stance: str,
        confidence: float,
        neg_score: float,
        pos_score: float,
        neg_words: List[str],
        pos_words: List[str],
    ) -> str:
        """Build human-readable reasoning."""
        
        parts = []
        
        if neg_score > 0.0:
            parts.append(f"Neg: {neg_score:.2f} ({len(neg_words)}w)")
        if pos_score > 0.0:
            parts.append(f"Pos: {pos_score:.2f} ({len(pos_words)}w)")
        
        if len(parts) == 0:
            return "No sentiment words found"
        
        return " | ".join(parts)


# ============================================================================
# TESTING & DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("IMPROVED STANCE ANALYZER - DEMONSTRATION")
    print("=" * 70)
    
    analyzer = ImprovedStanceAnalyzer(debug=False)
    
    test_cases = [
        # Should be oppose
        ("@Menlu_RI Menteri paling gak becus.", "oppose"),
        ("@Menlu_RI Mentri tolol", "oppose"),
        ("GOBLOGnya presiden @prabowo gk ada obatnya!!!", "oppose"),
        ("@P3gEl Emang mulut pejabat kita kayak kurang makan sekolahan", "oppose"),
        ("Gmana MALING bs tangkap maling??", "oppose"),
        
        # Should be support
        ("@KotaNusantara Program renovasi rumah bikin hati lega", "support"),
        ("Langkah Presiden Prabowo ini keren banget!", "support"),
        ("@kusuma4a Bangga kami atas kinerja TNI", "support"),
        ("Terima kasih bakti TNI", "support"),
        
        # Should be neutral
        ("Presiden membuat keputusan setelah konsultasi", "neutral"),
        ("Implementasi dimulai bulan depan", "neutral"),
    ]
    
    correct = 0
    for text, expected in test_cases:
        stance, confidence, reasoning = analyzer.analyze(text)
        
        match = "✓" if stance == expected else "✗"
        correct += 1 if stance == expected else 0
        
        print(f"\n{match} Text: {text[:60]}...")
        print(f"   Expected: {expected:10s} | Got: {stance:10s} ({confidence:.2f})")
        print(f"   Reasoning: {reasoning}")
    
    print("\n" + "=" * 70)
    print(f"Accuracy: {correct}/{len(test_cases)} ({100*correct/len(test_cases):.1f}%)")
    print("=" * 70)
