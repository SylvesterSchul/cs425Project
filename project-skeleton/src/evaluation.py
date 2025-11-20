"""
Evaluation
Metrics for evaluating recommendation quality and speed
"""
import numpy as np
import time


def precision_at_k(recommendations, test_set, k=10, threshold=4.0):
    """
    Calculate Precision@K
    
    Args:
        recommendations: Dict {user_id: [(item_id, rating), ...]}
        test_set: Dict {user_id: {item_id: rating, ...}}
        k: Top k recommendations to consider
        threshold: Rating threshold for "relevant"
        
    Returns:
        Average Precision@K across users
    """
    # TODO: For each user, check how many of top k recs are relevant in test set
    pass


def hit_rate(recommendations, test_set, k=10, threshold=4.0):
    """
    Calculate hit rate (did we recommend at least one good item?)
    
    Args:
        recommendations: Dict {user_id: [(item_id, rating), ...]}
        test_set: Dict {user_id: {item_id: rating, ...}}
        k: Top k recommendations
        threshold: Rating threshold
        
    Returns:
        Proportion of users with at least one hit
    """
    # TODO: Check if any top k recommendations are relevant
    pass


def measure_time(model, user_ids, n_recommendations=10):
    """
    Measure average recommendation time
    
    Args:
        model: Recommendation model with recommend() method
        user_ids: List of user IDs to test
        n_recommendations: Number of recommendations per user
        
    Returns:
        Average time in seconds per user
    """
    # TODO: Time how long it takes to generate recommendations
    pass


def compare_models(baseline, optimized, test_users, test_set):
    """
    Compare baseline vs optimized models
    
    Args:
        baseline: Baseline CF model
        optimized: LSH-optimized model
        test_users: Users to test on
        test_set: Test ratings
        
    Returns:
        Dict with comparison metrics
    """
    # TODO: Compare both models on accuracy and speed
    pass


if __name__ == "__main__":
    # Test your code here
    pass
