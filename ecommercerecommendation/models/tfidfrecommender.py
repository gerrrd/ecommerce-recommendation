"""
This script has the class TfIdfRecommender for the tf-idf based
recommendations.

"""

from __future__ import annotations

import pickle
from typing import List, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class TfIdfRecommender(BaseEstimator, TransformerMixin):
    def __init__(
        self, columns: Union[str, List[str]] = "Description", top_n: int = 5
    ):
        # ngram_range=(1, 2) for expressions up to 2 words, such as "LUNCH BOX"
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        if isinstance(columns, str):
            columns = [columns]
        self.columns = columns
        self.top_n = top_n
        self.products = None

    def fit(self, X, y=None) -> TfIdfRecommender:
        self.products = (
            X[["StockCode"] + self.columns]
            .drop_duplicates("StockCode")
            .reset_index(drop=True)
            .copy()
        )
        self.tfidf_matrix = self.tfidf.fit_transform(
            self.products[self.columns].apply(
                lambda row: " ".join(row), axis=1
            )
        )
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(
            self.products.index, index=self.products["StockCode"]
        ).drop_duplicates()

        return self

    def predict_element(self, stock_code: int, description: str = ""):
        """
        Method to perform the recommendation with using results of the
        training. It predicts either based on a selected stock or on the
        description given.

        :param stock_code: currently selected stock/product
        :param description: description of th selected item
        :return: the recommended items
        """

        if stock_code not in self.indices:
            if description == "":
                return self.products[["StockCode", "Description"]].head(0)
            new_element = self.tfidf.transform([description])
            similar_scores_sorted = (
                linear_kernel(new_element, self.tfidf_matrix)
                .flatten()
                .argsort()[-5:][::-1]
            )
            recommendations = self.products.iloc[similar_scores_sorted]
            return recommendations[["StockCode", "Description"]]

        idx = self.indices[stock_code]
        similar_scores = list(enumerate(self.cosine_sim[idx]))
        similar_scores = sorted(
            similar_scores, key=lambda x: x[1], reverse=True
        )[1 : self.top_n + 1]

        product_indices = [i[0] for i in similar_scores]

        return self.products.iloc[product_indices][
            ["StockCode", "Description"]
        ]

    def transform(self, X):
        return X["StockCode"].apply(self.predict_element)

    def save_pickle(self, filename: str):
        """
        Save the model to the given filename.

        :param filename: str, path where to store the model
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_pickle(filename) -> TfIdfRecommender:
        """
        Load the model from the given filename.

        :param filename: str, path from where to load the model
        :return: the loaded model
        """
        with open(filename, "rb") as f:
            return pickle.load(f)
