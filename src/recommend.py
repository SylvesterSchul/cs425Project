from data_preprocessing import load_movielens, create_utility_matrix
from user_user_cf import CollaborativeFiltering


def main():
    """Interactive recommendation system"""
    print("="*80)
    print("MOVIE RECOMMENDATION SYSTEM")
    print("="*80)
    
    # Load and prepare data
    print("\nLoading data...")
    ratings_df, movies_df = load_movielens()
    
    print("\nCreating utility matrix...")
    utility_matrix, user_map, movie_map, user_reverse, movie_reverse = create_utility_matrix(ratings_df)
    
    # Train model
    cf = CollaborativeFiltering(k=100)
    cf.fit(utility_matrix)
    
    # Interactive loop
    while True:
        print("\n" + "="*80)
        print(f"Available users: 1 to {len(user_map)} (Enter 0 to exit)")
        print("="*80)
        
        try:
            user_id = int(input("\nEnter User ID: "))
            
            if user_id == 0:
                print("\nGoodbye!")
                break
            
            if user_id not in user_map:
                print(f"❌ User {user_id} not found.")
                continue
            
            user_idx = user_map[user_id]
            
            # Show user info
            user_ratings = utility_matrix[user_idx].data
            print(f"\n{'='*80}")
            print(f"USER {user_id} PROFILE")
            print(f"{'='*80}")
            print(f"Movies rated: {len(user_ratings)}")
            print(f"Average rating: {user_ratings.mean():.2f} stars")
            
            # Get recommendations
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
            print("❌ Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()
