import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def build_maps(ratings_df):

    user_movie = {}
    movie_user = {}

    for row in ratings_df.itertuples(index=False):
        uid = int(row.userId)
        mid = int(row.movieId)
        r = float(row.rating)

        user_movie.setdefault(uid, {})[mid] = r
        movie_user.setdefault(mid, {})[uid] = r

    return user_movie, movie_user


def pearson_movie_similarity(movieA, movieB, movie_user, min_common_users=2):

    usersA = movie_user.get(movieA, {})
    usersB = movie_user.get(movieB, {})

    common_users = set(usersA.keys()) & set(usersB.keys())
    if len(common_users) < min_common_users:
        return 0.0

    a_vals = [usersA[u] for u in common_users]
    b_vals = [usersB[u] for u in common_users]

    meanA = sum(a_vals) / len(a_vals)
    meanB = sum(b_vals) / len(b_vals)

    num = 0.0
    denomA = 0.0
    denomB = 0.0

    for u in common_users:
        da = usersA[u] - meanA
        db = usersB[u] - meanB
        num += da * db
        denomA += da * da
        denomB += db * db

    denom = math.sqrt(denomA) * math.sqrt(denomB)
    return (num / denom) if denom != 0 else 0.0


def predict_rating(user_id, movie_id, user_movie, movie_user, user_means, per_movie_neighbors=30):

    user_baseline = user_means.get(user_id, 3.0)
    
    if user_id not in user_movie:
        return user_baseline
    
    user_rated = user_movie[user_id]
    
    sims = []
    for seen_movie, seen_rating in user_rated.items():
        if seen_movie == movie_id:
            continue
        
        s = pearson_movie_similarity(movie_id, seen_movie, movie_user)
        if s != 0:
            deviation = seen_rating - user_baseline
            sims.append((s, deviation))
    
    if not sims:
        return user_baseline
    
    sims.sort(key=lambda x: abs(x[0]), reverse=True)
    sims = sims[:per_movie_neighbors]
    
    numerator = sum(s * dev for s, dev in sims)
    denominator = sum(abs(s) for s, _ in sims)
    
    if denominator == 0:
        return user_baseline
    
    prediction = user_baseline + (numerator / denominator)
    
    return np.clip(prediction, 1.0, 5.0)


def calculate_user_means(user_movie):

    user_means = {}
    for user_id, ratings in user_movie.items():
        user_means[user_id] = sum(ratings.values()) / len(ratings)
    return user_means


def evaluate_k(k_value):
    ratings_df = pd.read_csv('ml-latest-small/ml-latest-small/ratings.csv')
    train_df, test_df = train_test_split(ratings_df, test_size=0.20, random_state=42)
    user_movie, movie_user = build_maps(train_df)
    user_means = calculate_user_means(user_movie)
    print(f"k={k_value}...", end=" ", flush=True)
    
    predictions = []
    actuals = []
    
    for row in test_df.itertuples(index=False):
        pred = predict_rating(
            user_id=int(row.userId),
            movie_id=int(row.movieId),
            user_movie=user_movie,
            movie_user=movie_user,
            user_means=user_means,
            per_movie_neighbors=k_value
        )
        
        predictions.append(pred)
        actuals.append(float(row.rating))
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    print(f"RMSE: {rmse:.3f}")
    
    return rmse


def main():
    """Test different k values"""
    print("="*60)
    print("ITEM-BASED CF - K-VALUE OPTIMIZATION")
    print("="*60)
    print()
    
    # Test these k values
    k_values = [10, 20, 30, 40, 50, 60, 70, 80]
    
    results = {}
    
    for k in k_values:
        rmse = evaluate_k(k)
        results[k] = rmse
    
    # Print results table
    print()
    print(f"\n{'k':>5} | {'RMSE':>8}")
    print("-" * 17)
    
    for k in sorted(results.keys()):
        print(f"{k:>5} | {results[k]:>8.3f}")
    
    best_k = min(results, key=results.get)
    print()
    print("="*60)
    print(f"Best: k={best_k} (RMSE={results[best_k]:.3f})")
    print("="*60)


if __name__ == "__main__":
    main()