"""
Main Script
Run the movie recommender system
"""
from data_preprocessing import load_movielens_data, create_utility_matrix, train_test_split
from collaborative_filtering import UserBasedCF
from minhash_lsh import OptimizedCF
from evaluation import precision_at_k, measure_time, compare_models


def main():
    print("Movie Recommender System")
    print("=" * 50)
    
    # TODO: Load data
    print("Loading data...")
    
    # TODO: Create utility matrix
    print("Creating utility matrix...")
    
    # TODO: Train/test split
    print("Splitting into train/test...")
    
    # TODO: Train baseline model
    print("\nTraining baseline model...")
    
    # TODO: Train optimized model
    print("Training optimized model...")
    
    # TODO: Evaluate both models
    print("\nEvaluating models...")
    
    # TODO: Compare results
    print("\nResults:")
    

if __name__ == "__main__":
    main()
