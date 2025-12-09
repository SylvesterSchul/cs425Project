import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


def load_movielens(data_dir='data/raw/ml-1m'):
    """Load MovieLens 1M dataset"""
    # Load ratings (user_id::movie_id::rating::timestamp)
    ratings_df = pd.read_csv(
        f'{data_dir}/ratings.dat',
        sep='::',
        engine='python',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        encoding='latin-1'
    )
    
    # Load movies (movie_id::title::genres)
    movies_df = pd.read_csv(
        f'{data_dir}/movies.dat',
        sep='::',
        engine='python',
        names=['movie_id', 'title', 'genres'],
        encoding='latin-1'
    )
    
    print(f"Loaded {len(ratings_df):,} ratings from {ratings_df['user_id'].nunique():,} users "
          f"on {ratings_df['movie_id'].nunique():,} movies")
    
    return ratings_df, movies_df


def create_utility_matrix(ratings_df):
    """
    Create sparse utility matrix from ratings
    
    Returns:
        utility_matrix: Sparse matrix (users × movies)
        user_id_to_idx: Dict mapping user_id -> matrix row index
        movie_id_to_idx: Dict mapping movie_id -> matrix column index
        idx_to_user_id: Dict mapping matrix row index -> user_id
        idx_to_movie_id: Dict mapping matrix column index -> movie_id
    """
    unique_users = sorted(ratings_df['user_id'].unique())
    unique_movies = sorted(ratings_df['movie_id'].unique())
    
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
    
    idx_to_user_id = {idx: user_id for user_id, idx in user_id_to_idx.items()}
    idx_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_to_idx.items()}
    
    # Get dimensions
    n_users = len(unique_users)
    n_movies = len(unique_movies)
    
    # FAST: Vectorized construction using sparse matrix constructor
    user_indices = ratings_df['user_id'].map(user_id_to_idx).values
    movie_indices = ratings_df['movie_id'].map(movie_id_to_idx).values
    ratings = ratings_df['rating'].values
    
    # Create sparse matrix directly (MUCH faster than iterrows!)
    utility_matrix = csr_matrix(
        (ratings, (user_indices, movie_indices)),
        shape=(n_users, n_movies)
    )
    
    # Calculate sparsity
    total_entries = n_users * n_movies
    filled_entries = utility_matrix.nnz
    sparsity = 100 * (1 - filled_entries / total_entries)
    
    print(f"Utility Matrix: ({n_users}, {n_movies}) | "
          f"Ratings: {filled_entries:,} | Sparsity: {sparsity:.2f}%")
    
    return utility_matrix, user_id_to_idx, movie_id_to_idx, idx_to_user_id, idx_to_movie_id


if __name__ == "__main__":
    print("Testing data loading...")
    ratings_df, movies_df = load_movielens()
    
    print("\nCreating utility matrix...")
    utility_matrix, user_map, movie_map, user_reverse, movie_reverse = create_utility_matrix(ratings_df)
    
    print("\n✓ Data preprocessing working!")
    print(f"  Users: {utility_matrix.shape[0]}")
    print(f"  Movies: {utility_matrix.shape[1]}")
    print(f"  Ratings: {utility_matrix.nnz:,}")
