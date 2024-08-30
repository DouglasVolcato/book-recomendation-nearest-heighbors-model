import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


class NearestNeighborsModel:
    def getRecommendations(self, book_title: str) -> list:
        model = self.__getModel()
        books = self.__getBooksData()
        ratings_pivot = self.__getRatingsData()

        # Find the index of the book
        book_index = books[books['title'] == book_title].index[0]

        # Get the distances and indices of the neighbors
        distances, indices = model.kneighbors(
            ratings_pivot.iloc[book_index, :].values.reshape(1, -1), n_neighbors=6)

        # Get the book titles for the recommendations
        recommended_books = []
        for i in range(1, len(distances.flatten())):
            recommended_books.append(
                [books.iloc[indices.flatten()[i]]['title'], distances.flatten()[i]])

        return [book_title, recommended_books]

    def trainAndSaveModel(self) -> None:
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        data = self.__getRatingsData()
        model.fit(data.values)

        np.save('cache/models/model.npy', model)

    def __getModel(self) -> NearestNeighbors:
        return np.load('cache/models/model.npy', allow_pickle=True).item()

    def __getRatingsData(self) -> pd.DataFrame:
        ratings = pd.read_csv(
            'cache/data/BX-Book-Ratings.csv',
            encoding="ISO-8859-1",
            sep=";",
            header=0,
            names=['user', 'isbn', 'rating'],
            usecols=['user', 'isbn', 'rating'],
            dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})
        user_counts = ratings['user'].value_counts()

        ratings = ratings[ratings['user'].isin(
            user_counts[user_counts >= 200].index)]

        book_counts = ratings['isbn'].value_counts()
        ratings = ratings[ratings['isbn'].isin(
            book_counts[book_counts >= 100].index)]

        return ratings.pivot(index='user', columns='isbn', values='rating').fillna(0)

    def __getUsersData(self) -> pd.DataFrame:
        return pd.read_csv(
            'cache/data/BX-Users.csv',
            encoding="ISO-8859-1",
            sep=";",
            header=0,
            names=["User-ID", "Location", "Age"],
            usecols=["User-ID", "Location", "Age"],
            dtype={'user': 'int32', 'isbn': 'str', 'rating': 'int32'})

    def __getBooksData(self) -> pd.DataFrame:
        return pd.read_csv(
            'cache/data/BX-Books.csv',
            encoding="ISO-8859-1",
            sep=";",
            header=0,
            names=['isbn', 'title', 'author'],
            usecols=['isbn', 'title', 'author'],
            dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})
