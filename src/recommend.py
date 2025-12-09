from data_preprocessing import load_movielens, create_utility_matrix
from user_user_cf import CollaborativeFiltering


def main():
    """Interactive recommendation system"""
    print("MOVIE RECOMMENDATION SYSTEM")

    
    print("\nLoading data...")
    ratings_df, movies_df = load_movielens()
    
    print("\nCreating utility matrix...")
    utility_matrix, user_map, movie_map, user_reverse, movie_reverse = create_utility_matrix(ratings_df)

    cf = CollaborativeFiltering(k=120)
    cf.fit(utility_matrix)

    valid_users = set(user_map.keys())

    while True:
        print(f"Available users: 1 to {max(valid_users)}")
        
        try:
            user_input = input("\nEnter User ID (or 'q' to quit): ")
            if user_input.lower() == 'q':
                print("\nGoodbye!")
                break
            user_id = int(user_input)

            if user_id not in valid_users:
                print(f"❌ User {user_id} not found in dataset. Try again.")
                continue
            
            user_idx = user_map[user_id]

            user_ratings = utility_matrix[user_idx].data
            print(f"\n{'='*80}")
            print(f"USER {user_id} PROFILE")
            print(f"{'='*80}")
            print(f"Movies rated: {len(user_ratings)}")
            print(f"Average rating: {user_ratings.mean():.2f} stars")

            recommendations = cf.recommend_movies(user_idx, n_recommendations=10)
            
            print(f"\n{'='*80}")
            print(f"TOP 10 RECOMMENDATIONS FOR USER {user_id}")
            print(f"{'='*80}\n")
            
            for rank, (movie_idx, pred_rating) in enumerate(recommendations, 1):
                movie_id = movie_reverse[movie_idx]
                movie_info = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
                
                print(f"{rank:2d}. {movie_info['title']}")
                print(f"    Predicted: {pred_rating:.2f} stars | {movie_info['genres']}\n")
            
        except ValueError:
            print("❌ Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()