"""
Fast approximate inference model.
Low latency, moderate accuracy for simple inputs.
"""

import time
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class FastModel:
    """
    Fast approximate model for low-latency inference.
    Uses simple linear classifier with TF-IDF features.
    """
    
    def __init__(self, latency_ms: float = 10.0, accuracy: float = 0.85):
        """
        Initialize the fast model.
        
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
        Train the fast model on provided data.
        
        Args:
            texts: List of training text strings
            labels: List of training labels
        """
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        self.model = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', LogisticRegression(max_iter=100, random_state=42))
        ])
        
        self.model.fit(texts, labels)
        self.is_trained = True
    
    def predict(self, text: str) -> Tuple[int, float, np.ndarray]:
        """
        Perform fast inference on input text.
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (predicted_label, confidence, probability_distribution)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before inference")
        
        # Simulate inference latency
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
        
        # Simulate batch latency (slightly more efficient per item)
        time.sleep(self.latency_ms / 1000.0 * len(texts) * 0.8)
        
        probs = self.model.predict_proba(texts)
        results = []
        
        for prob in probs:
            predicted_label = int(np.argmax(prob))
            confidence = float(prob[predicted_label])
            results.append((predicted_label, confidence, prob))
        
        return results

