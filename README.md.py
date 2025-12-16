""""
cs425Project
Movie Finder System

Team Members:
Rachel Lo
Sylvester Schulteis
Jagger Vicente

Project Description:
This project is a movie recommender system built using item-based collaborative filtering with Pearson correlation. The goal of the project is to recommend movies to a user by looking at how similar different movies are based on user rating patterns.

Instead of directly comparing users to one another, the system compares movies using ratings from users who have watched both. When a user ID is entered, the program looks at the movies that user has already rated and predicts ratings for movies they have not seen yet. The top 5 movies with the highest predicted scores are then recommended.

Dataset:
The system uses the MovieLens small dataset, which contains user ratings and movie information such as titles and genres. The data is loaded and processed using the pandas library in Python.

Requirements:
- Python 3
- pandas

How to Run:
1. Make sure Python 3 and pandas are installed.
2. Place the MovieLens dataset in the following directory:
   ml-latest-small/ml-latest-small/
3. Run the program using:
   python COMPSCI425ProjectAlgorithms.py
4. Enter a valid user ID from the dataset when prompted.

Output:
The program prints the top 5 recommended movies for the selected user, along with each movie’s title, genres, and predicted rating score.

Something to keep in mind:
This project uses item-based collaborative filtering rather than user-based filtering. One limitation of this approach is the cold-start problem, meaning the system cannot generate recommendations for users who have no previous ratings in the dataset.
"""