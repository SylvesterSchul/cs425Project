import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import os


def load_movielens_data(data_path='data/raw/ml-latest-small'):
    ratings_path = os.path.join(data_path, 'ratings.csv')
    movies_path = os.path.join(data_path, 'movies.csv')
    
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)
    
    print(f"Total of {len(ratings_df)} ratings")
    print(f"{ratings_df['userId'].nunique()} users")
    print(f"{ratings_df['movieId'].nunique()} movies")
    
    return ratings_df, movies_df


def create_utility_matrix(ratings_df):

    user_set = sorted(ratings_df['userId'].unique())
    movie_set = sorted(ratings_df['movieId'].unique())

    user_id_index = {user_id: idx for idx, user_id in enumerate(user_set)}
    movie_id_index = {movie_id: idx for idx, movie_id in enumerate(movie_set)}

    index_to_user_id = {idx: user_id for user_id, idx in user_id_index.items()}
    index_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_index.items()}

    row = ratings_df['userId'].map(user_id_index).values
    col = ratings_df['movieId'].map(movie_id_index).values
    ratings = ratings_df['rating'].values
    
    n_users = len(user_set)
    n_movies = len(movie_set)
    
    utility_matrix = csr_matrix(
        (ratings, (row, col)),
        shape=(n_users, n_movies)
    )

    sparsity = 1 - (utility_matrix.nnz / (n_users * n_movies))

    print(f"Total ratings: {utility_matrix.nnz:,}")
    print(f"Sparsity: {sparsity:.2%}")
    
    return utility_matrix, user_id_index, movie_id_index, index_to_user_id, index_to_movie_id


def train_test_split(ratings_df, test_size=0.2, random_state=42):

    np.random.seed(random_state)
    
    train_list = []
    test_list = []
    
    for user_id in ratings_df['userId'].unique():
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        
        if len(user_ratings) <= 2:
            train_list.append(user_ratings)
            continue
        
        user_ratings = user_ratings.sample(frac=1, random_state=random_state)
        n_test = max(1, int(len(user_ratings) * test_size))
        
        test_list.append(user_ratings.iloc[:n_test])
        train_list.append(user_ratings.iloc[n_test:])
    
    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    print(f"\nTrain/Test Split:")
    print(f"  Training: {len(train_df):,} ratings")
    print(f"  Test: {len(test_df):,} ratings")
    
    return train_df, test_df

def visualize_utility_matrix(utility_matrix, index_to_user_id, index_to_movie_id, movies_df, n_users=50, n_movies=100):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    print(f"Showing first {n_users} users and first {n_movies} movies")
    
    sample = utility_matrix[:n_users, :n_movies].toarray()

    masked_sample = np.ma.masked_where(sample == 0, sample)

    fig, ax = plt.subplots(figsize=(20, 12))

    cmap = plt.cm.YlOrRd
    cmap.set_bad(color='white') 

    im = ax.imshow(masked_sample, 
                   cmap=cmap,
                   aspect='auto',
                   vmin=0.5,  
                   vmax=5.0)  

    cbar = plt.colorbar(im, ax=ax, label='Rating (0.5-5.0 stars)')
    cbar.ax.set_ylabel('Rating (0.5-5.0 stars)', fontsize=12)

    ax.set_xlabel('Movies', fontsize=12)
    ax.set_ylabel('Users', fontsize=12)
    ax.set_title(f'Utility Matrix Heatmap: First 50 Users × First 100 Movies', 
                 fontsize=14, fontweight='bold')

    ax.set_xticks(range(0, n_movies, 10))
    ax.set_yticks(range(0, n_users, 5))
    ax.set_xticklabels(range(1, n_movies+1, 10))
    ax.set_yticklabels([index_to_user_id[i] for i in range(0, n_users, 5)])
    
    plt.tight_layout()

    output_path = 'utility_matrix_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f" Saved heatmap to: {output_path}")

if __name__ == "__main__":
    print("="*50)
    print("MOVIELENS DATA PREPROCESSING")
    print("="*50)
    
    print("\n1. Loading data...")
    ratings_df, movies_df = load_movielens_data()
    
    print("\n2. Creating utility matrix...")
    utility_matrix, user_map, movie_map, user_reverse, movie_reverse = create_utility_matrix(ratings_df)
 
    visualize_utility_matrix(utility_matrix, user_reverse, movie_reverse, movies_df)
    
    print("\n3. Creating train/test split...")
    train_df, test_df = train_test_split(ratings_df)
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE!")
    print("="*50)