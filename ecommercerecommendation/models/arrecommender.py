"""
This script has the class ARRecommender for the association rules based
recommendations.

"""

from __future__ import annotations

import pickle
from typing import List, Tuple

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.base import BaseEstimator, TransformerMixin

SIMILAR_CUSTOMERS = 10


class ARRecommender(BaseEstimator, TransformerMixin):
    def __init__(self, top_n: int = 5):
        self.rules = None
        self.top_n = top_n

    def fit(self, X, y=None) -> ARRecommender:
        transaction_encoder = TransactionEncoder()
        transactions = (
            X.groupby("InvoiceNo")["StockCode"].unique().apply(list).to_list()
        )
        transaction_encoder.fit(transactions)
        encoded_transactions = transaction_encoder.transform(transactions)
        df_transactions = pd.DataFrame(
            encoded_transactions, columns=transaction_encoder.columns_
        )
        frequent_items = fpgrowth(
            df_transactions, min_support=0.01, use_colnames=True
        )
        self.rules = association_rules(
            frequent_items, metric="lift", min_threshold=0.5
        )

        self.rules = self.rules.sort_values(
            ["confidence", "lift"], ascending=[False, False]
        )

        return self

    def transform(self, X):
        return X["StockCode"].apply(
            lambda stock_code: self.get_recommendations(
                current_selection=[stock_code]
            )
        )

    def get_recommendations(self, current_selection: List[int]) -> List[int]:
        """
        Method to perform the recommendation with using the rules dataframe
        from the training.

        :param current_selection: currently selected stocks/products
        :return: the recommended items
        """

        current_selection = set(current_selection)
        matches = self.rules[
            self.rules["antecedents"].apply(
                lambda x: set(x).issubset(current_selection)
            )
        ]
        recommendations = matches.sort_values(
            by=["lift", "confidence"], ascending=False
        )

        suggested_items = []
        for cons in recommendations["consequents"]:
            for item in cons:
                if (
                    item not in current_selection
                    and item not in suggested_items
                ):
                    suggested_items.append(item)

        return suggested_items[: self.top_n]

    def evaluate_rules(
        self, X, sample_perc: float = 1
    ) -> Tuple[float, int, int]:
        """
        Evaluates the recommender system.

        :param X: the test set
        :param sample_perc: to downsample, in case it would last too long
        :return: the stats
        """

        test_df = (
            X[
                X["StockCode"].isin(
                    X["StockCode"].sample(frac=sample_perc, random_state=81)
                )
            ]
            .groupby("InvoiceNo")["StockCode"]
            .unique()
            .reset_index()
        )

        hits = 0
        opportunities = 0

        # Iterate through each transaction in the test set
        for _, transaction in test_df.iterrows():
            items_bought = {"StockCode"}

            # Check each rule
            for _, rule in self.rules.iterrows():
                antecedent = set(rule["antecedents"])
                consequent = set(rule["consequents"])

                # Opportunity: Did the customer buy the 'If' part?
                if antecedent.issubset(items_bought):
                    opportunities += 1
                    # Hit: Did they also buy the 'Then' part?
                    if consequent.issubset(items_bought):
                        hits += 1

        hit_rate = hits / opportunities if opportunities > 0 else 0
        return hit_rate, hits, opportunities

    def save_pickle(self, filename: str):
        """
        Save the model to the given filename.

        :param filename: path where to store the model
        """

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_pickle(filename) -> ARRecommender:
        """
        Load the model from the given filename.

        :param filename: path from where to load the model
        :return: the loaded model
        """

        with open(filename, "rb") as f:
            return pickle.load(f)
