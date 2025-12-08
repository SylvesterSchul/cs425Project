import numpy as np
import matplotlib.pyplot as plt


def plot_similarity_distribution(user_similarity, output_path='similarity_distribution_pearson.png'):
    """
    Plot distribution of user-user Pearson correlation scores
    Shows how similar users are to each other using Pearson correlation
    """
    print("\nCreating Pearson correlation distribution plot...")
    
    # Get upper triangle (avoid counting each pair twice)
    triu_indices = np.triu_indices_from(user_similarity, k=1)
    similarities = user_similarity[triu_indices]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(similarities, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(similarities.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {similarities.mean():.3f}')
    ax1.set_xlabel('Pearson Correlation', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of User-User Pearson Correlation Scores', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Box plot
    ax2.boxplot(similarities, vert=True)
    ax2.set_ylabel('Pearson Correlation', fontsize=12)
    ax2.set_title('User-User Pearson Correlation Statistics', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Statistics text
    stats_text = (f"Min: {similarities.min():.3f}\n"
                  f"Max: {similarities.max():.3f}\n"
                  f"Mean: {similarities.mean():.3f}\n"
                  f"Median: {np.median(similarities):.3f}\n"
                  f"Std: {similarities.std():.3f}")
    ax2.text(1.15, similarities.mean(), stats_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {output_path}")
    
    # Print statistics to console
    print(f"\nPearson Correlation Statistics:")
    print(f"  Mean:   {similarities.mean():.3f}")
    print(f"  Median: {np.median(similarities):.3f}")
    print(f"  Min:    {similarities.min():.3f}")
    print(f"  Max:    {similarities.max():.3f}")
    print(f"  Std:    {similarities.std():.3f}")


if __name__ == "__main__":
    from data_preprocessing import load_movielens, utility_matrix as create_matrix
    from user_user_cf import UserBasedCF
    
    print("="*80)
    print("PEARSON CORRELATION DISTRIBUTION ANALYSIS")
    print("="*80)
    
    print("\nLoading data...")
    ratings_df, movies_df = load_movielens()
    
    print("Creating utility matrix...")
    utility_mat, user_map, movie_map, user_reverse, movie_reverse = create_matrix(ratings_df)
    
    print("Computing Pearson correlation (k=60)...")
    cf = UserBasedCF(k=60)
    cf.fit(utility_mat)
    
    # Generate visualization
    plot_similarity_distribution(cf.user_similarity)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)