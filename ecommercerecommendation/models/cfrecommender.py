"""
This script has the class CFRecommender for the collaborative filtering based
recommendations.

"""

from __future__ import annotations

import pickle
from typing import List, Union

import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity

SIMILAR_CUSTOMERS = 10


class CFRecommender(BaseEstimator, TransformerMixin):
    def __init__(self, top_n: int = 5):
        # ngram_range=(1, 2) for expressions up to 2 words, such as "LUNCH BOX"
        self.top_n = top_n
        self.customer_stock = None
        self.customer_similarity = None

    def fit(self, X, y=None) -> CFRecommender:
        df_cf = (
            X.groupby(["CustomerID", "StockCode"])
            .size()
            .reset_index(name="interaction")
        )

        self.customer_stock = (
            df_cf.pivot_table(
                index="CustomerID",
                columns="StockCode",
                values="interaction",
                aggfunc="count",
            )
            .fillna(0)
            .astype(int)
        )

        self.customer_similarity = pd.DataFrame(
            cosine_similarity(sparse.csr_matrix(self.customer_stock)),
            index=self.customer_stock.index,
            columns=self.customer_stock.index,
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
        :param current_selection: currently selected stocks/products
        :return: the recommended items
        """

        if target_user not in self.customer_stock.index:
            if (selected_stocks is None) or (
                isinstance(selected_stocks, list)
                and (len(selected_stocks) == 0)
            ):
                # unknown customer, no selected products
                return []
            # unknown customer, existing selected products
            new_line = pd.DataFrame(
                [1] * len(selected_stocks),
                index=selected_stocks,
                columns=[target_user],
            ).T

            use_customer_stock = pd.concat(
                [self.customer_stock, new_line]
            ).fillna(0)

            use_customer_similarity = pd.DataFrame(
                cosine_similarity(sparse.csr_matrix(use_customer_stock)),
                index=use_customer_stock.index,
                columns=use_customer_stock.index,
            )
        else:
            use_customer_similarity = self.customer_similarity
            use_customer_stock = self.customer_stock

        similar_customers = (
            use_customer_similarity[target_user]
            .sort_values(ascending=False)
            .index[1 : SIMILAR_CUSTOMERS + 1]
        )
        sim_customer_stocks = use_customer_stock.loc[similar_customers].sum()
        already_bought = use_customer_stock.loc[target_user]
        recommendations = sim_customer_stocks[already_bought == 0].sort_values(
            ascending=False
        )
        return list(recommendations.head(self.top_n).index)

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
