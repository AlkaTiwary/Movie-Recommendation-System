import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
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
comparison=linear_kernel(features_vectorized, features_vectorized)
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
indices = pd.Series(movie_dataset.index, index=movie_dataset["title"]).drop_duplicates()
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

# import numpy as np
# import pandas as pd
# import difflib
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# movies_data = pd.read_csv(r"C:/Users/user/Desktop/college/pro coder alka/ML/movie_recommendation/movies_data.csv")
# movies_data.head()
# selected_features = ['genres','keywords','tagline','cast','director']
# print(selected_features)

# for feature in selected_features:
#   movies_data[feature] = movies_data[feature].fillna('')
# combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
# print(combined_features)
# vectorizer = TfidfVectorizer()
# feature_vectors = vectorizer.fit_transform(combined_features)
# print(feature_vectors)
# similarity = (int)cosine_similarity(feature_vectors)
# print(similarity)
# print(similarity.shape)
# movie_name = input(' Enter your favourite movie name : ')
# list_of_all_titles = movies_data['title'].tolist()
# titles_np=np.asarray(list_of_all_titles)
# find_close_match = difflib.get_close_matches(movie_name, titles_np)
# print(find_close_match)
# close_match = find_close_match[0]
# print(close_match)
# index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
# print(index_of_the_movie)
# similarity_score = list(enumerate(similarity[index_of_the_movie]))
# print(similarity_score)
# len(similarity_score)
# sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
# print(sorted_similar_movies)
# print('Movies suggested for you : /n')

# i = 1

# for movie in sorted_similar_movies:
#   index = movie[0]
#   title_from_index = movies_data[movies_data.index==index]['title'].values[0]
#   if (i<30):
#     print(i, '.',title_from_index)
#     i+=1
# movie_name = input(' Enter your favourite movie name : ')

# list_of_all_titles = movies_data['title'].tolist()

# find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

# close_match = find_close_match[0]

# index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
# # for i in enumerate(list[index_of_the_movie]):
# #   similarity_score= similarity[index_of_the_movie]
# similarity_score = similarity[list(enumerate([index_of_the_movie]))]
# # similarity_score = list(enumerate(similarity[index_of_the_movie]))

# sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

# print('Movies suggested for you : /n')

# i = 1

# for movie in sorted_similar_movies:
#   index = movie[0]
#   title_from_index = movies_data[movies_data.index==index]['title'].values[0]
#   if (i<30):
#     print(i, '.',title_from_index)
#     i+=1