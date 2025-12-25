"""
Input preprocessing module for text classification tasks.
Handles text normalization, tokenization, and feature extraction.
Includes cheap difficulty signals for pre-inference routing.
"""

import re
from typing import List, Dict, Any


def normalize_text(text: str) -> str:
    """
    Normalize input text by lowercasing and removing extra whitespace.
    
    Args:
        text: Raw input text string
        
    Returns:
        Normalized text string
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def compute_lexical_diversity(text: str) -> float:
    """
    Compute lexical diversity (type-token ratio) as a cheap difficulty signal.
    Higher diversity indicates more varied vocabulary, potentially more complex.
    
    Computed in O(n) time where n is the number of tokens.
    Requires no model inference or external services.
    
    Args:
        text: Normalized text string
        
    Returns:
        Lexical diversity ratio (unique words / total words)
    """
    words = text.split()
    if len(words) == 0:
        return 0.0
    
    unique_words = len(set(words))
    total_words = len(words)
    
    return unique_words / total_words if total_words > 0 else 0.0


def extract_features(text: str) -> Dict[str, Any]:
    """
    Extract simple features from text for difficulty estimation.
    All features are computed in O(n) time and require no model inference.
    
    Args:
        text: Input text string
        
    Returns:
        Dictionary containing extracted features
    """
    normalized = normalize_text(text)
    words = normalized.split()
    
    features = {
        'length': len(normalized),
        'word_count': len(words),
        'char_count': len(normalized),
        'has_numbers': bool(re.search(r'\d', normalized)),
        'has_special_chars': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', normalized)),
        'avg_word_length': len(normalized) / max(len(words), 1),
        'lexical_diversity': compute_lexical_diversity(normalized),
    }
    
    return features


def preprocess_input(text: str) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for input text.
    
    Args:
        text: Raw input text string
        
    Returns:
        Dictionary containing normalized text and extracted features
    """
    normalized = normalize_text(text)
    features = extract_features(normalized)
    
    return {
        'text': normalized,
        'raw_text': text,
        'features': features
    }
