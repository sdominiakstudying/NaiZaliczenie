"""
Authors: Stanisław Dominiak s18864, Mateusz Pioch s21331

based on https://towardsdatascience.com/unsupervised-classification-project-building-a-movie-recommender-with-clustering-analysis-and-4bab0738efe6
as well as https://programming.rhysshea.com/K-means_movie_ratings/
and https://pandas.pydata.org/pandas-docs/stable/user_guide/sparse.html
and https://stackoverflow.com/questions/20459536/convert-pandas-dataframe-to-sparse-numpy-matrix-directly ...
and the data taken from fellow students at 15.11.2022 21:00 (est.)
"""

"""
Importing all the necessary libs
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import itertools


### uploading the data from the same folder
ratings = pd.read_csv('data.csv', sep=';', header = None)
#test = ratings.iloc[4].at[3]
###### making the ta0ble workable

### listing all the users
users = range(0,15)

### listing all the movies - in a preliminary way
movies = ratings.iloc[0,1:30:2]

### combining the two into a Grand Table
df = pd. DataFrame(index=users, columns=movies)

###Adding all the movies and their ratings - yeah, a bit suboptimal, but it always works on the data provided
for y in range (1,73,2):
    for x in range (0,15):
        if not ratings.iloc[x,y] in df.columns:
            df[ratings.iloc[x,y]] = np.nan
        df.loc[x,ratings.iloc[x,y]] = ratings.iloc[x,y+1]

### Now that we're done, let's update movies
allMovies = df.columns

### now gotta transform into sparse ratings with a data array
sparse_df = df.astype(pd.SparseDtype("float64",np.nan))



### time to select a scientifically appropriate number of clusters.
### It is our conclusion that there are three types of opinions - mine, less correct and wrong -
### - therefore 3 are enough.
predictions = KMeans(n_clusters=3, algorithm='full').fit_predict(sparse_df)

### and, to now get clustered properly
clustered = pd.concat([df.reset_index(), pd.DataFrame({'group':predictions})], axis=1)

### Now, to pick one of 3 clusters:
cluster = clustered[clustered.group == 0].drop(['index', 'group'], axis=1)


#now, to select a user - from 0 (i e the lecturer, who's beyond number one) to the others
user_id = 0
user_2_ratings  = cluster.loc[user_id, :]
user_2_unrated_movies =  user_2_ratings[user_2_ratings.isnan()]
avg_ratings = pd.concat([user_2_unrated_movies, cluster.mean()], axis=1, join='inner').loc[:,0]
avg_ratings.drop([df.index[1]])
### Best movies!
avg_ratings.sort_values(by='ratings', ascending=False)[:5]
### ...greatly unappealing movies
avg_ratings.sort_values(by='ratings', ascending=True)[:5]














