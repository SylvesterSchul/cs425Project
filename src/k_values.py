import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from data_preprocessing import load_movielens, create_utility_matrix
from user_user_cf import CollaborativeFiltering


def create_test_set(utility_matrix, test_size=0.20):
    """Create train/test split"""
    print(f"\nCreating train/test split ({int(test_size*100)}% test)...")
    
    rows, cols = utility_matrix.nonzero()
    all_pairs = [(row, col, utility_matrix[row, col]) for row, col in zip(rows, cols)]
    train_pairs, test_pairs = train_test_split(all_pairs, test_size=test_size, random_state=42)
    
    train_matrix = utility_matrix.copy().tolil()
    for user_idx, movie_idx, _ in test_pairs:
        train_matrix[user_idx, movie_idx] = 0
    train_matrix = train_matrix.tocsr()
    
    print(f"  Training: {len(train_pairs):,} | Test: {len(test_pairs):,}")
    return train_matrix, test_pairs


def evaluate(cf, test_pairs):
    """Evaluate model on test set"""
    print(f"  Evaluating on {len(test_pairs):,} ratings...", end=" ", flush=True)
    
    predictions = [cf.predict_rating(u, m) for u, m, _ in test_pairs]
    actuals = [r for _, _, r in test_pairs]
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    print(f"RMSE: {rmse:.3f}")
    
    return rmse


def test_k_values(train_matrix, k_values, test_pairs):
    """Test different k values"""
    results = {}
    
    print("\n" + "="*80)
    print("TESTING K VALUES")
    print("="*80)
    
    for k in k_values:
        print(f"\nk={k}...")
        cf = CollaborativeFiltering(k=k)
        cf.fit(train_matrix)
        rmse = evaluate(cf, test_pairs)
        results[k] = rmse
    
    return results


def print_results(results):
    """Print results table"""
    print("\n" + "="*80)
    print("K-VALUE RESULTS")
    print("="*80)
    print(f"\n{'k':>5} | {'RMSE':>8}")
    print("-" * 17)
    
    for k in sorted(results.keys()):
        print(f"{k:>5} | {results[k]:>8.3f}")
    
    best_k = min(results, key=results.get)
    print("\n" + "="*80)
    print(f"Best: k={best_k} (RMSE={results[best_k]:.3f})")
    print("="*80)


def main():
    """Main evaluation"""
    print("="*80)
    print("K-VALUE ANALYSIS")
    print("="*80)
    
    print("\nLoading data...")
    ratings_df, movies_df = load_movielens()
    
    print("\nCreating utility matrix...")
    utility_matrix, _, _, _, _ = create_utility_matrix(ratings_df)
    
    train_matrix, test_pairs = create_test_set(utility_matrix, test_size=0.20)
    
    k_values = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
    
    print(f"\nTesting k values: {k_values}")
    
    results = test_k_values(train_matrix, k_values, test_pairs)
    print_results(results)
    
    print("\n✓ Evaluation complete!")
    print("Run plot_k120_optimal.py to generate visualization.")


if __name__ == "__main__":
    main()