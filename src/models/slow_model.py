"""
Slow accurate inference model.
Higher latency, higher accuracy for complex inputs.
"""

import time
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class SlowModel:
    """
    Slow accurate model for high-accuracy inference.
    Uses more complex features and potentially deeper model.
    """
    
    def __init__(self, latency_ms: float = 100.0, accuracy: float = 0.95):
        """
        Initialize the slow model.
        
        Args:
            latency_ms: Simulated inference latency in milliseconds
            accuracy: Model accuracy (used for simulation)
        """
        self.latency_ms = latency_ms
        self.accuracy = accuracy
        self.model = None
        self.vectorizer = None
        self.is_trained = False
    
    def train(self, texts: list, labels: list):
        """
        Train the slow model on provided data.
        
        Args:
            texts: List of training text strings
            labels: List of training labels
        """
        # Use more features and higher-order n-grams for better accuracy
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        # Use more iterations for better convergence
        self.model = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', LogisticRegression(
                max_iter=500,
                C=0.1,
                random_state=42,
                solver='liblinear'
            ))
        ])
        
        self.model.fit(texts, labels)
        self.is_trained = True
    
    def predict(self, text: str) -> Tuple[int, float, np.ndarray]:
        """
        Perform slow inference on input text.
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (predicted_label, confidence, probability_distribution)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before inference")
        
        # Simulate inference latency (slower than fast model)
        time.sleep(self.latency_ms / 1000.0)
        
        # Get prediction probabilities
        probs = self.model.predict_proba([text])[0]
        predicted_label = int(np.argmax(probs))
        confidence = float(probs[predicted_label])
        
        return predicted_label, confidence, probs
    
    def predict_batch(self, texts: list) -> list:
        """
        Perform batch inference on multiple texts.
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of (predicted_label, confidence, probability_distribution) tuples
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before inference")
        
        # Simulate batch latency
        time.sleep(self.latency_ms / 1000.0 * len(texts) * 0.7)
        
        probs = self.model.predict_proba(texts)
        results = []
        
        for prob in probs:
            predicted_label = int(np.argmax(prob))
            confidence = float(prob[predicted_label])
            results.append((predicted_label, confidence, prob))
        
        return results

