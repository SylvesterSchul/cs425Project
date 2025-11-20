"""
Collaborative Filtering
User-based collaborative filtering with cosine similarity
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class UserBasedCF:
    """User-based Collaborative Filtering"""
    
    def __init__(self, k=20):
        """
        Args:
            k: Number of nearest neighbors
        """
        self.k = k
        self.utility_matrix = None
        self.user_similarity = None
        
    def fit(self, utility_matrix):
        """
        Fit model by computing user similarities
        
        Args:
            utility_matrix: Sparse matrix (users x items)
        """
        self.utility_matrix = utility_matrix
        # TODO: Compute user-user cosine similarity
        pass
    
    def find_neighbors(self, user_id):
        """
        Find k most similar users
        
        Args:
            user_id: Target user
            
        Returns:
            neighbor_ids: Array of k nearest neighbor user IDs
            similarities: Array of similarity scores
        """
        # TODO: Get top k similar users (excluding user_id itself)
        pass
    
    def predict_rating(self, user_id, item_id):
        """
        Predict rating for user-item pair
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        # TODO: 
        # 1. Find k neighbors
        # 2. Get their ratings for item_id
        # 3. Compute weighted average (weighted by similarity)
        pass
    
    def recommend(self, user_id, n=10):
        """
        Generate top-N recommendations
        
        Args:
            user_id: User ID
            n: Number of recommendations
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        # TODO:
        # 1. Get all unrated items for this user
        # 2. Predict rating for each
        # 3. Return top N by predicted rating
        pass


if __name__ == "__main__":
    # Test your code here
    pass
