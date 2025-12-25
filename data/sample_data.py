"""
Sample data generation for text classification task.
Generates synthetic sentiment classification dataset with clear easy/hard distinction.
"""

import numpy as np
from typing import List, Tuple


def generate_sample_texts(num_samples: int = 1000, seed: int = 42) -> Tuple[List[str], List[int]]:
    """
    Generate synthetic text classification dataset.
    Creates texts with clear distinction between easy (fast model sufficient) 
    and hard (slow model needed) inputs.
    
    Args:
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (texts, labels) where labels are 0 (negative) or 1 (positive)
    """
    np.random.seed(seed)
    
    # Simple, unambiguous positive patterns (easy for both models)
    easy_positive_patterns = [
        "great amazing wonderful excellent fantastic",
        "love this product highly recommend",
        "best purchase ever very satisfied",
        "outstanding quality exceeded expectations",
        "perfect exactly what I needed",
        "delighted with the service",
        "top notch quality and value",
        "exceeded all my expectations",
        "absolutely fantastic experience",
        "could not be happier"
    ]
    
    # Simple, unambiguous negative patterns (easy for both models)
    easy_negative_patterns = [
        "terrible awful horrible disappointed",
        "waste of money do not buy",
        "poor quality broken immediately",
        "worst purchase ever regret",
        "completely useless product",
        "very disappointed with quality",
        "not worth the price",
        "defective item arrived broken",
        "terrible customer service",
        "would not recommend to anyone"
    ]
    
    # Complex, ambiguous patterns that require nuanced understanding (hard)
    # These contain mixed signals, subtle negations, or context-dependent meaning
    hard_ambiguous_patterns = [
        "the product is okay but could be better than expected",
        "not bad but not great either somewhat disappointed",
        "mixed feelings about this purchase some good some bad",
        "some good aspects some bad aspects overall mediocre",
        "mediocre quality for the price point expected more",
        "it works but has some issues that concern me",
        "average product nothing special but functional",
        "decent but expected more value for the money",
        "functional but lacks polish and attention to detail",
        "acceptable quality with reservations about durability",
        "initially satisfied but later discovered problems",
        "good value but quality could be significantly improved",
        "satisfactory performance with notable limitations",
        "meets basic requirements but falls short of expectations",
        "reasonable purchase with some important caveats"
    ]
    
    # Very long, complex texts with multiple clauses (hard)
    hard_complex_patterns = [
        "while the initial impression was positive and the packaging seemed professional the actual product quality did not meet my expectations and I found several issues that concern me about long term reliability",
        "this product has both positive and negative aspects that make it difficult to recommend without reservations the good parts work well but the bad parts are significant enough to impact overall satisfaction",
        "after using this for several weeks I have mixed feelings the core functionality works as advertised but there are enough minor annoyances and one major flaw that prevent me from giving it a full recommendation",
        "the product description was accurate in some ways but misleading in others which created expectations that were not met in practice leading to disappointment despite some redeeming qualities",
        "I wanted to love this product and there are things I genuinely like about it but the combination of quality issues and design flaws make it hard to justify the price point even with the positive aspects"
    ]
    
    texts = []
    labels = []
    difficulty_flags = []  # Track which inputs are hard
    
    # Generate easy samples (60% of dataset)
    # Both models should perform similarly on these
    num_easy = int(num_samples * 0.6)
    for _ in range(num_easy):
        if np.random.random() < 0.5:
            pattern = np.random.choice(easy_positive_patterns)
            label = 1
        else:
            pattern = np.random.choice(easy_negative_patterns)
            label = 0
        
        # Add minimal variation
        words = pattern.split()
        np.random.shuffle(words)
        text = " ".join(words)
        
        # Occasionally add simple modifiers
        if np.random.random() < 0.2:
            text = text + " " + np.random.choice(["really", "very", "quite"])
        
        texts.append(text)
        labels.append(label)
        difficulty_flags.append(False)
    
    # Generate hard ambiguous samples (30% of dataset)
    # Slow model should clearly outperform fast model on these
    num_hard_ambiguous = int(num_samples * 0.3)
    for _ in range(num_hard_ambiguous):
        pattern = np.random.choice(hard_ambiguous_patterns)
        # Hard samples have correct labels but are ambiguous
        # Label based on overall sentiment (slightly negative bias for realism)
        if "okay" in pattern or "decent" in pattern or "acceptable" in pattern:
            label = 1 if np.random.random() < 0.6 else 0
        elif "disappointed" in pattern or "concern" in pattern or "issues" in pattern:
            label = 0 if np.random.random() < 0.7 else 1
        else:
            label = np.random.randint(0, 2)
        
        # Add variation
        words = pattern.split()
        np.random.shuffle(words)
        text = " ".join(words)
        
        # Add filler phrases to increase complexity
        if np.random.random() < 0.4:
            filler = np.random.choice([
                "I think that", "in my opinion", "from my experience",
                "to be honest", "frankly speaking", "honestly"
            ])
            text = filler + " " + text
        
        texts.append(text)
        labels.append(label)
        difficulty_flags.append(True)
    
    # Generate very hard complex samples (10% of dataset)
    # These are long and require deep understanding
    num_hard_complex = num_samples - num_easy - num_hard_ambiguous
    for _ in range(num_hard_complex):
        pattern = np.random.choice(hard_complex_patterns)
        # Complex patterns are labeled based on overall sentiment
        if "positive" in pattern or "like" in pattern or "good" in pattern:
            label = 1 if np.random.random() < 0.6 else 0
        elif "disappointment" in pattern or "issues" in pattern or "flaws" in pattern:
            label = 0 if np.random.random() < 0.7 else 1
        else:
            label = np.random.randint(0, 2)
        
        texts.append(pattern)
        labels.append(label)
        difficulty_flags.append(True)
    
    # Shuffle
    indices = np.arange(len(texts))
    np.random.shuffle(indices)
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    difficulty_flags = [difficulty_flags[i] for i in indices]
    
    return texts, labels
