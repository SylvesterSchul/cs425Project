import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import lil_matrix



class CollaborativeFiltering:
    
    def __init__(self, k=120):
        self.k = k
        self.utility_matrix = None
        self.user_similarity = None
        self.user_means = None
        self.n_users = None
        self.n_movies = None
    
    def fit(self, utility_matrix):
        print(f"\nTraining CF (k={self.k})...")
        self.utility_matrix = utility_matrix
        self.n_users, self.n_movies = utility_matrix.shape
        
        # Calculate user means
        self.user_means = np.zeros(self.n_users)
        for user_idx in range(self.n_users):
            user_ratings = utility_matrix[user_idx].data
            if len(user_ratings) > 0:
                self.user_means[user_idx] = user_ratings.mean()
        
        # Convert cosine to Pearson
        centered_matrix = lil_matrix(utility_matrix.shape)
        
        for user_idx in range(self.n_users):
            user_row = utility_matrix[user_idx]
            if user_row.nnz > 0:
                indices = user_row.indices
                values = user_row.data
                centered_values = values - self.user_means[user_idx]
                
                for idx, val in zip(indices, centered_values):
                    centered_matrix[user_idx, idx] = val
        
        centered_matrix = centered_matrix.tocsr()
        
        # Compute Pearson correlation (cosine on centered data)
        self.user_similarity = cosine_similarity(centered_matrix, dense_output=True)
        np.fill_diagonal(self.user_similarity, -1)
    
    def find_neighbors(self, user_idx):

        similarities = self.user_similarity[user_idx]
        
        if self.k < len(similarities):
            neighbor_indices = np.argpartition(similarities, -self.k)[-self.k:]
            neighbor_indices = neighbor_indices[np.argsort(similarities[neighbor_indices])][::-1]
        else:
            neighbor_indices = np.argsort(similarities)[::-1][:self.k]
        
        return neighbor_indices, similarities[neighbor_indices]
    
    def predict_rating(self, user_idx, movie_idx):

        neighbor_indices, neighbor_similarities = self.find_neighbors(user_idx)
        
        neighbor_ratings = []
        neighbor_weights = []
        
        for neighbor_idx, similarity in zip(neighbor_indices, neighbor_similarities):
            rating = self.utility_matrix[neighbor_idx, movie_idx]
            
            if rating > 0:
                centered_rating = rating - self.user_means[neighbor_idx]
                neighbor_ratings.append(centered_rating)
                neighbor_weights.append(similarity)
        
        # Weighted prediction
        if len(neighbor_ratings) > 0 and np.sum(np.abs(neighbor_weights)) > 0:
            weighted_sum = np.dot(neighbor_ratings, neighbor_weights)
            weight_sum = np.sum(np.abs(neighbor_weights))
            predicted = self.user_means[user_idx] + (weighted_sum / weight_sum)
            return np.clip(predicted, 1.0, 5.0)
        
        return np.clip(self.user_means[user_idx], 1.0, 5.0) if self.user_means[user_idx] > 0 else 3.0
    
    def recommend_movies(self, user_idx, n_recommendations=10):
        user_ratings = self.utility_matrix[user_idx].toarray().flatten()
        unrated_movies = np.where(user_ratings == 0)[0]
        
        predictions = []
        for movie_idx in unrated_movies:
            predicted_rating = self.predict_rating(user_idx, movie_idx)
            predictions.append((movie_idx, predicted_rating))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]


if __name__ == "__main__":
    from data_preprocessing import load_movielens, create_utility_matrix
    
    print("Testing User-Based Collaborative Filtering...")
    ratings_df, movies_df = load_movielens()
    utility_matrix, user_map, movie_map, user_reverse, movie_reverse = create_utility_matrix(ratings_df)
    
    cf = CollaborativeFiltering(k=120)
    cf.fit(utility_matrix)
    
    test_user_idx = 0
    test_user_id = user_reverse[test_user_idx]

    neighbor_indices, neighbor_similarities = cf.find_neighbors(test_user_idx)
    
    print(f"\nTop 5 Most Similar Users:")
    for i, (neighbor_idx, similarity) in enumerate(zip(neighbor_indices[:5], neighbor_similarities[:5]), 1):
        neighbor_id = user_reverse[neighbor_idx]
        neighbor_ratings = utility_matrix[neighbor_idx].data
        neighbor_count = len(neighbor_ratings)
        neighbor_avg = neighbor_ratings.mean() if len(neighbor_ratings) > 0 else 0
        
        print(f"  {i}. User {neighbor_id}: Pearson={similarity:.3f} "
              f"(rated {neighbor_count} movies, avg={neighbor_avg:.2f})")

    recommendations = cf.recommend_movies(test_user_idx, n_recommendations=5)
    
    print(f"\nTop 5 recommendations for User {test_user_id}:")
    for rank, (movie_idx, pred_rating) in enumerate(recommendations, 1):
        movie_id = movie_reverse[movie_idx]
        movie_info = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
        print(f"  {rank}. {movie_info['title']}")
    
    print("\nCF complete")