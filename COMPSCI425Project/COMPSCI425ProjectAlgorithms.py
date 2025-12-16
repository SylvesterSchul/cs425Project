# Data Mining algorithms (Item-based Collaborative Filtering)
# Pearson correlation on MOVIES (items) instead of USERS

import pandas as pd
import math

# ----------------------------
# Helper: Build maps for fast lookup
# user -> {movieId: rating}
# movie -> {userId: rating}
# ----------------------------
def build_maps(ratings_df: pd.DataFrame):
    user_movie = {}
    movie_user = {}

    for row in ratings_df.itertuples(index=False):
        uid = int(row.userId)
        mid = int(row.movieId)
        r = float(row.rating)

        user_movie.setdefault(uid, {})[mid] = r
        movie_user.setdefault(mid, {})[uid] = r

    return user_movie, movie_user


# ----------------------------
# Pearson similarity between two MOVIES
# Uses ratings from users who rated BOTH movies
# ----------------------------
def pearson_movie_similarity(movieA: int, movieB: int, movie_user: dict, min_common_users: int = 2) -> float:
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


# ----------------------------
# Item-based recommender
# ----------------------------
def recommend_item_based(user_id: int, top_n: int = 5, per_movie_neighbors: int = 30, min_common_users: int = 2):
    ratings_df = pd.read_csv('ml-latest-small/ml-latest-small/ratings.csv')
    movies_df = pd.read_csv('ml-latest-small/ml-latest-small/movies.csv')

    user_movie, movie_user = build_maps(ratings_df)

    if user_id not in user_movie:
        print("❌ That user ID does not exist in the dataset.\n")
        return

    user_rated = user_movie[user_id]
    rated_movies = set(user_rated.keys())

    movie_info = movies_df.set_index("movieId")[["title", "genres"]]

    sim_cache = {}

    def sim(m1: int, m2: int) -> float:
        key = (m1, m2) if m1 < m2 else (m2, m1)
        if key in sim_cache:
            return sim_cache[key]
        s = pearson_movie_similarity(m1, m2, movie_user, min_common_users=min_common_users)
        sim_cache[key] = s
        return s

    all_movies = set(movie_user.keys())
    candidates = list(all_movies - rated_movies)

    scored = []

    for cand in candidates:
        sims = []
        for seen_movie, seen_rating in user_rated.items():
            s = sim(cand, seen_movie)
            if s != 0:
                sims.append((s, seen_rating))

        if not sims:
            continue

        sims.sort(key=lambda x: abs(x[0]), reverse=True)
        sims = sims[:per_movie_neighbors]

        numerator = sum(s * r for s, r in sims)
        denominator = sum(abs(s) for s, _ in sims)

        if denominator == 0:
            continue

        pred_score = numerator / denominator
        scored.append((cand, pred_score, len(sims)))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_n]

    print(f"\nTop {top_n} movie recommendations for user {user_id} (Item-based Pearson):\n")

    rec_num = 1
    for movie_id, pred_score, support in top:
        if movie_id in movie_info.index:
            title = movie_info.loc[movie_id, "title"]
            genres = movie_info.loc[movie_id, "genres"]
        else:
            title = f"(movieId={movie_id})"
            genres = "Unknown"

        print(f"{rec_num}. {title}")
        print(f"   Genres: {genres}")
        print(f"   Predicted score: {pred_score:.3f} | Similar-movie links used: {support}\n")
        rec_num += 1


if __name__ == "__main__":
    df = pd.read_csv('ml-latest-small/ml-latest-small/ratings.csv')
    valid_users = set(df['userId'].unique())

    while True:
        try:
            user_input = input("Enter a user ID (or 'q' to quit): ").strip()

            if user_input.lower() == "q":
                print("Exiting program.")
                break

            user_input = int(user_input)

            if user_input not in valid_users:
                print("❌ That user ID does not exist in the dataset. Try again.\n")
                continue

            print(f"\nRunning item-based recommendations for user {user_input}...\n")
            recommend_item_based(
                user_id=user_input,
                top_n=5,
                per_movie_neighbors=30,
                min_common_users=2
            )
            break

        except ValueError:
            print("❌ Please enter a valid number.\n")
