# Project Proposal

**Team Members:**
- [Name 1] - Data preprocessing & baseline CF
- [Name 2] - MinHash/LSH optimization  
- [Name 3] - Evaluation & presentation

**Project:** Movie Recommendation System with MinHash/LSH Optimization

## Approach

**1. Baseline: User-Based Collaborative Filtering**
- Build utility matrix from MovieLens dataset
- Compute user-user cosine similarity
- Use k-nearest neighbors to predict ratings
- Recommend top-N items with highest predicted ratings

**2. Optimization: MinHash + LSH**
- Create MinHash signatures for users (compact representation)
- Use LSH to hash similar users into same buckets
- Only compute exact similarity for candidates in same bucket
- Achieve 10-20x speedup while maintaining accuracy

**3. Evaluation**
- Dataset: MovieLens 100K (100,000 ratings)
- Metrics: Precision@10, Hit Rate, Response Time
- Compare: Baseline vs Optimized approach

## Innovation
- Apply approximate algorithms (MinHash/LSH) to recommendation systems
- Demonstrate accuracy-speed trade-off
- Show scalability benefits for large datasets

## Timeline
- Week 1: Data loading, baseline CF
- Week 2: MinHash/LSH implementation
- Week 3: Evaluation & comparison
- Week 4: Presentation preparation
