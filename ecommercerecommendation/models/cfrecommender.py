"""
This script has the class CFRecommender for the collaborative filtering based
recommendations.

"""

from __future__ import annotations

import pickle
from typing import List, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity

SIMILAR_CUSTOMERS = 10


class CFRecommender(BaseEstimator, TransformerMixin):
    def __init__(self, top_n: int = 5):
        self.top_n = top_n
        self.customer_stock = None
        self.customer_similarity = None
        self.customer_ids = None
        self.stock_ids = None
        self.customer_indices = None
        self.stock_indices = None

    def fit(self, X, y=None) -> CFRecommender:
        df_cf = (
            X.groupby(["CustomerID", "StockCode"])
            .size()
            .reset_index(name="interaction")
        )

        self.customer_ids = df_cf["CustomerID"].astype("category")
        self.stock_ids = df_cf["StockCode"].astype("category")

        # customer stocks as a sparse matrix
        self.customer_stock = csr_matrix(
            (
                df_cf["interaction"].astype(float),
                (self.customer_ids.cat.codes, self.stock_ids.cat.codes),
            )
        )

        self.customer_indices = self.customer_ids.cat.categories
        self.stock_indices = self.stock_ids.cat.categories

        # customer x customer similarity, significantly smaller than per stocks
        self.customer_similarity = pd.DataFrame(
            cosine_similarity(self.customer_stock),
            index=self.customer_indices,
            columns=self.customer_indices,
        )

        return self

    def transform(self, X):
        return X["CustomerID"].apply(
            lambda customer_id: self.get_recommendations(
                target_user=customer_id
            )
        )

    def get_recommendations(
        self, target_user: int, selected_stocks: Union[List[int], None] = None
    ) -> List[int]:
        """
        Method to perform the recommendation with using results of the
        training. It predicts either based on the target user or on the
        already selected items

        :param target_user: target user CustomerID
        :param selected_stocks: currently selected stocks/products
        :return: the recommended items
        """

        if target_user not in self.customer_indices:
            if (selected_stocks is None) or (
                isinstance(selected_stocks, list)
                and (len(selected_stocks) == 0)
            ):
                # unknown customer, no selected products
                return []
            # unknown customer, existing selected products

            selected_stock_indices = [
                self.stock_indices.get_loc(s)
                for s in selected_stocks
                if s in self.stock_indices
            ]
            if not selected_stock_indices:
                # unknown customer, no selected products are known
                return []

            new_customer_stocks = np.ones(len(selected_stock_indices))
            new_customer_columns = np.array(selected_stock_indices)
            new_customer_rows = np.zeros(len(selected_stock_indices))

            user_vector = csr_matrix(
                (
                    new_customer_stocks,
                    (new_customer_rows, new_customer_columns),
                ),
                shape=(1, self.customer_stock.shape[1]),
            )

            similar_scores = cosine_similarity(
                user_vector,
                csr_matrix(self.customer_stock),
            ).flatten()

            customer_similarity = pd.Series(
                similar_scores, index=self.customer_indices
            )
        else:
            customer_similarity = self.customer_similarity[target_user]
            user_vector = self.customer_stock[
                self.customer_indices.get_loc(target_user)
            ]

        similar_customers = customer_similarity.sort_values(
            ascending=False
        ).index[0 : SIMILAR_CUSTOMERS + 1]

        sim_user_indices = [
            self.customer_indices.get_loc(c) for c in similar_customers
        ]
        stock_weights = self.customer_stock[sim_user_indices, :].sum(axis=0).A1

        stock_weights[user_vector.indices] = 0

        top_stock_indices = np.argsort(stock_weights)[-self.top_n :][::-1]

        return [
            self.stock_indices[i]
            for i in top_stock_indices
            if stock_weights[i] > 0
        ]

    def save_pickle(self, filename: str):
        """
        Save the model to the given filename.

        :param filename: path where to store the model
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_pickle(filename) -> CFRecommender:
        """
        Load the model from the given filename.

        :param filename: path from where to load the model
        :return: the loaded model
        """
        with open(filename, "rb") as f:
            return pickle.load(f)
