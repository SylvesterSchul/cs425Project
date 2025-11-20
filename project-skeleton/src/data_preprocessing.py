"""
Data Preprocessing
Load MovieLens data and create utility matrix
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def load_movielens_data(data_path='data/raw/ml-100k'):
    """
    Load MovieLens ratings data
    
    Args:
        data_path: Path to ml-100k folder
        
    Returns:
        DataFrame with columns: user_id, item_id, rating, timestamp
    """
    # TODO: Load u.data file (tab-separated)
    # Column names: user_id, item_id, rating, timestamp
    pass


def create_utility_matrix(ratings_df):
    """
    Create utility matrix from ratings DataFrame
    
    Args:
        ratings_df: DataFrame with user_id, item_id, rating columns
        
    Returns:
        utility_matrix: Sparse matrix (users x items)
        user_mapping: Dict mapping user_id to row index
        item_mapping: Dict mapping item_id to column index
    """
    # TODO: Create sparse matrix
    # Users = rows, Items = columns, Values = ratings
    pass


def train_test_split(ratings_df, test_size=0.2):
    """
    Split ratings into train and test sets
    
    Args:
        ratings_df: DataFrame with user_id, item_id, rating
        test_size: Proportion for test set
        
    Returns:
        train_df, test_df
    """
    # TODO: For each user, randomly select test_size% of their ratings for test set
    pass


if __name__ == "__main__":
    # Test your code here
    pass
