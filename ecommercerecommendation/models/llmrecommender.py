"""
This script has the class LLMRecommender for the LLM based
recommendations.

"""

from __future__ import annotations

import pickle
from typing import List, Union

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity


class LLMRecommender(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        columns: Union[str, List[str]] = "Description",
        top_n: int = 5,
    ):
        # ngram_range=(1, 2) for expressions up to 2 words, such as "LUNCH BOX"
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.products = None
        self.indices = None
        self.embeddings = None
        self.cosine_sim = None
        self.similar_items = None
        if isinstance(columns, str):
            columns = [columns]
        self.columns = columns
        self.top_n = top_n

    def fit(self, X, y=None) -> LLMRecommender:
        self.products = (
            X[["StockCode"] + self.columns]
            .drop_duplicates("StockCode")
            .reset_index(drop=True)
            .copy()
        )

        self.embeddings = self.model.encode(
            self.products[self.columns]
            .apply(lambda row: " ".join(row), axis=1)
            .to_list(),
            show_progress_bar=True,
        )
        cosine_sim = cosine_similarity(self.embeddings)

        self.similar_items = pd.DataFrame(
            cosine_sim,
            index=self.products["StockCode"],
            columns=self.products["StockCode"],
        )

        self.indices = pd.Series(
            self.products.index, index=self.products["StockCode"]
        ).drop_duplicates()

        return self

    def predict_element(self, stock_code, description: str = ""):
        """
        Method to perform the recommendation with using results of the
        training. It predicts either based on a selected stock or on the
        description given.

        :param stock_code: currently selected stock/product
        :param description: description of th selected item
        :return: the recommended items
        """

        if stock_code not in self.indices:
            # embedding prediction
            new_embedding = self.model.encode([description])
            similar_scores_sorted = (
                cosine_similarity(new_embedding, self.embeddings)
                .flatten()
                .argsort()[-5:][::-1]
            )

            recommendations = self.products.iloc[similar_scores_sorted]
            return recommendations[["StockCode", "Description"]]

        similar_stock_codes = (
            self.similar_items[stock_code]
            .sort_values(ascending=False)[1 : 5 + 1]
            .index
        )

        return self.products[
            self.products["StockCode"].isin(similar_stock_codes)
        ][["StockCode", "Description"]]

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
    def load_pickle(filename) -> LLMRecommender:
        """
        Load the model from the given filename.

        :param filename: str, path from where to load the model
        :return: the loaded model
        """
        with open(filename, "rb") as f:
            return pickle.load(f)
