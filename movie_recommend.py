import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
movie_dataset=pd.read_csv(r"C:/Users/user/Desktop/college/pro coder alka/ML/movie_recommendation/movies_data.csv")
print(movie_dataset.head())
print(movie_dataset.shape)
imp_features=['genres','keywords','tagline','cast','director']
for features in imp_features:
    movie_dataset[features]=movie_dataset[features].fillna(" ")
final_features= movie_dataset['genres']+' '+movie_dataset['keywords']+' '+movie_dataset['tagline']+' '+movie_dataset['cast']+' '+movie_dataset['director']
print(final_features)
vectorizer=TfidfVectorizer()
features_vectorized=vectorizer.fit_transform(final_features)
print(features_vectorized)
comparison=cosine_similarity(features_vectorized)
print(comparison)
user_input=input("Enter your favourite movie name: ")
all_titles=movie_dataset["title"].tolist()
print(all_titles)
titles_np=np.asarray(all_titles)
close_match=difflib.get_close_matches(user_input,titles_np)
print(close_match)
closest_match=close_match[0]
print(closest_match)
movie_index=movie_dataset[movie_dataset.title == closest_match]['index'].values[0]
print(movie_index)
similarity_score=list(enumerate(comparison[movie_index]))
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print('Movies suggested for you : /n')
i = 1
for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movie_dataset[movie_dataset.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1
