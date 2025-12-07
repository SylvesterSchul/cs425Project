#Data Mining algorithms
import pandas as pd
import math

#user to user function
#Given a user, compare to all other user ratings
#Find the top 5 most similar users
#Print the highest rating among them that the given user hasn't rated
#Excel file has a userID, movieID, and rating
#Not sure if there's any point in hashing. Data is already very simple. Just following slides
def userToUser(userID):
    similarUserIDs = [-1, -1, -1, -1, -1]
    similarityMeasures = [-1, -1, -1, -1, -1]
    df = pd.read_csv('ml-latest-small/ml-latest-small/ratings.csv')
    userIDIndex = 1
    givenUserDF = df.loc[df['userId'] == userID]
    #The same index corresponds to these two lists
    givenUserMovies = givenUserDF['movieId'].tolist()
    givenUserRatings = givenUserDF['rating'].tolist()
    givenUserAvgRatings = givenUserDF['rating'].mean()
    maxID = df['userId'].max()
    userIndex = 1
    #Finding most similar users
    while userIndex < maxID:
        if(userIndex != userID):
            numerator = 0
            denom1 = 0
            denom2 = 0
            xIndex = 0
            userDF = df.loc[df['userId'] == userIndex]
            userMovies = userDF['movieId'].tolist()
            userRatings = userDF['rating'].tolist()
            userAvgRatings = userDF['rating'].mean()
            #similarity calculation
            for x in givenUserMovies:
                yIndex = 0
                for y in userMovies:
                    if x == y:
                        numerator += (givenUserRatings[xIndex] - givenUserAvgRatings) * (userRatings[yIndex] - userAvgRatings)
                        denom1 += (givenUserRatings[xIndex] - givenUserAvgRatings) * (givenUserRatings[xIndex] - givenUserAvgRatings)
                        denom2 += (userRatings[yIndex] - userAvgRatings) * (userRatings[yIndex] - userAvgRatings)
                    yIndex += 1
                xIndex += 1
            totalDenom = (math.sqrt(denom1) * math.sqrt(denom2))
            if totalDenom > 0: #There is no overlap
                similarityCalc = (numerator / (math.sqrt(denom1) * math.sqrt(denom2)))
            else:
                similarityCalc = 0
            simIndex = 0
            tempUserIndex = userIndex
            #Swapping values if better similarity is found
            while simIndex < 5:
                if (similarityCalc > similarityMeasures[simIndex]):
                    temp = similarityMeasures[simIndex]
                    similarityMeasures[simIndex] = similarityCalc
                    similarityCalc = temp
                    temp = similarUserIDs[simIndex]
                    similarUserIDs[simIndex] = tempUserIndex
                    tempUserIndex = temp
                simIndex += 1
        userIndex += 1
    #Finding best recommendations from similar users
    #For each user, recommend highest rated movie that given user hasn't rated
    movieDF = pd.read_csv('ml-latest-small/ml-latest-small/movies.csv')
    for x in similarUserIDs:
        userDF = df.loc[df['userId'] == x]
        userDF = userDF.sort_values(by='rating', ascending=False)
        userMovies = userDF['movieId'].tolist()
        notPrinted = True
        movieIndex = 0
        while movieIndex < userDF['movieId'].count() and notPrinted:
            #Check if the movie is rated by given user
            if not userMovies[movieIndex] in givenUserDF.values:
                print(movieDF.loc[movieDF['movieId'] == userMovies[movieIndex]])
                notPrinted = False
            movieIndex += 1
                
    
userToUser(1)