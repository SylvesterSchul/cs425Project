import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import random
import os

from data_preprocessing import load_movielens, create_utility_matrix
from user_user_cf import CollaborativeFiltering


def create_test_set(utility_matrix, test_size=0.10):
    """Create train/test split"""
    print(f"\nCreating train/test split ({int(test_size*100)}% test)...")
    rows, cols = utility_matrix.nonzero()
    all_pairs = [(row, col, utility_matrix[row, col]) for row, col in zip(rows, cols)]
    train_pairs, test_pairs = train_test_split(all_pairs, test_size=test_size, random_state=42)
    
    train_matrix = utility_matrix.copy().tolil()
    for user_idx, movie_idx, _ in test_pairs:
        train_matrix[user_idx, movie_idx] = 0
    train_matrix = train_matrix.tocsr()
    
    print(f"  Training: {len(train_pairs):,} | Testing: {len(test_pairs):,}")
    return train_matrix, test_pairs


def evaluate_on_test_set(cf, test_pairs, sample_size=15000):
    """Evaluate model on sampled test set"""
    if len(test_pairs) > sample_size:
        test_pairs = random.sample(test_pairs, sample_size)
    
    print(f"  Evaluating on {len(test_pairs):,} ratings...", end=" ", flush=True)
    
    predictions = [cf.predict_rating(u, m) for u, m, _ in test_pairs]
    actuals = [rating for _, _, rating in test_pairs]
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    print(f"RMSE: {rmse:.3f}, MAE: {mae:.3f}")
    return rmse, mae


def test_k_values(train_matrix, k_values, test_pairs):
    """Test different k values"""
    results = {}
    print("\n" + "="*80)
    print("TESTING K VALUES")
    print("="*80)
    
    for k in k_values:
        print(f"\nk={k}...")
        start = time.time()
        
        cf = CollaborativeFiltering(k=k)
        cf.fit(train_matrix)
        rmse, mae = evaluate_on_test_set(cf, test_pairs)
        
        results[k] = (rmse, mae, time.time() - start)
        print(f"  Time: {results[k][2]:.2f}s")
    
    return results


def visualize_and_print_results(results):
    """Create visualization and print results"""
    k_values = sorted(results.keys())
    rmse_values = [results[k][0] for k in k_values]
    mae_values = [results[k][1] for k in k_values]
    
    # Print table
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\n{'k':>5} | {'RMSE':>8} | {'MAE':>8} | {'Time (s)':>10}")
    print("-" * 40)
    
    for k in k_values:
        rmse, mae, elapsed = results[k]
        print(f"{k:>5} | {rmse:>8.3f} | {mae:>8.3f} | {elapsed:>10.2f}")
    
    best_k = min(k_values, key=lambda k: results[k][0])
    print("\n" + "="*80)
    print(f"Best: k={best_k} (RMSE={results[best_k][0]:.3f}, MAE={results[best_k][1]:.3f})")
    print("="*80)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # RMSE
    ax1.plot(k_values, rmse_values, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('k (Number of Neighbors)', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('RMSE vs k', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.scatter([best_k], [results[best_k][0]], color='red', s=200, zorder=5,
                label=f'Best: k={best_k}')
    ax1.legend()
    
    # MAE
    ax2.plot(k_values, mae_values, 'g-o', linewidth=2, markersize=8)
    ax2.set_xlabel('k (Number of Neighbors)', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title('MAE vs k', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.scatter([best_k], [results[best_k][1]], color='red', s=200, zorder=5,
                label=f'Best: k={best_k}')
    ax2.legend()
    
    # Save
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/k_value_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: results/k_value_analysis.png")
    plt.close()


def main():
    """Main evaluation"""
    print("="*80)
    print("K-VALUE ANALYSIS")
    print("="*80)
    
    print("\nLoading data...")
    ratings_df, movies_df = load_movielens()
    
    print("\nCreating utility matrix...")
    utility_matrix, _, _, _, _ = create_utility_matrix(ratings_df)
    
    train_matrix, test_pairs = create_test_set(utility_matrix, test_size=0.10)
    
    k_values = [5, , 70, 90, 110]
    print(f"\nTesting k values: {k_values}")
    print("Using 15,000 sampled predictions per k for speed")
    
    results = test_k_values(train_matrix, k_values, test_pairs)
    visualize_and_print_results(results)
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()