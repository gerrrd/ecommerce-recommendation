"""
This script has the class ARRecommender for the association rules based
recommendations extended by an LLM-embedding instead of direct matching.

"""

from __future__ import annotations

import pickle
from typing import List, Tuple

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances

from ecommercerecommendation.utils.data import (
    clean_entry,
    find_minimax_assignment,
)

SIMILAR_CUSTOMERS = 10


class ARLLMRecommender(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", top_n: int = 5):
        self.rules = None
        self.model_name = model_name
        self.transformer = SentenceTransformer(self.model_name)
        self.top_n = top_n

    def fit(self, X, y=None) -> ARLLMRecommender:
        transaction_encoder = TransactionEncoder()

        # we use code + description as unique identifier of a stock, as
        # some of the products have slightly different descriptions
        X["Stock"] = X["StockCode"].astype(str) + " " + X["Description"]
        transactions = (
            X.groupby("InvoiceNo")["Stock"].unique().apply(list).to_list()
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

        # getting rid of the StockCode for embedding and embedding:
        self.rules["antecedents_embedded"] = self.rules["antecedents"].apply(
            lambda antecedents: self.transformer.encode(
                [" ".join(stock.split(" ")[1:]) for stock in antecedents]
            )
        )

        return self

    def transform(self, X):
        return X["StockCode"].apply(
            lambda stock_code: self.get_recommendations(
                current_selection=[stock_code]
            )
        )

    def get_recommendations(
        self, current_selection: List[str], min_confidence: float = 0.5
    ) -> List[str]:
        """
        Method to perform the recommendation with using the rules dataframe
        from the training.

        :param current_selection: descriptions of currently selected stocks
        :param min_confidence: to use rules over a certain confidence level
        :return: the recommended items
        """

        # as training data is clean, we preprocess the input data the same way:
        current_selection_clean = [
            clean_entry(product) for product in current_selection
        ]
        current_selection_embedded = self.transformer.encode(
            current_selection_clean
        )
        current_selection_clean = [
            text.lower() for text in current_selection_clean
        ]

        # in association rules: antecedents is a subset of selected.
        # combining with LLM, subset (meaning 0 euclidian distance
        # between antecedent(s) and selected) becomes a minimum
        # distance among the possible matches: we check how far
        # what the closest (minimum) maximum distance is pairwise.
        # Having 2 antecedents and 3 selected elements, we can have the
        # following distance matrix:
        # [[1.2131854, 1.0794144, 1.2798889],
        #  [1.2439317, 1.1904092, 1.3334384]]
        # from which antecedent[0]-selected[0] and antecedent[1]-selected[1]
        # is the best combination: having the minimum of their max distance.
        # This is calculated by

        consequents = self.rules[self.rules["confidence"] >= min_confidence][
            ["consequents", "antecedents_embedded"]
        ].copy()
        consequents["minimax_distances"] = (
            consequents["antecedents_embedded"]
            .apply(
                lambda embedded: euclidean_distances(
                    embedded, current_selection_embedded
                )
            )
            .apply(find_minimax_assignment)
        )
        consequents.drop(columns="antecedents_embedded", inplace=True)

        consequents = consequents.sort_values(by="minimax_distances").explode(
            "consequents"
        )
        consequents["consequents_text"] = consequents["consequents"].apply(
            lambda text: " ".join(text.split(" ")[1:])
        )
        consequents = consequents[
            ~consequents["consequents_text"]
            .str.lower()
            .isin(current_selection_clean)
        ]

        return (
            consequents["consequents"]
            .drop_duplicates()
            .to_list()[: self.top_n]
        )

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
    def load_pickle(filename) -> ARLLMRecommender:
        """
        Load the model from the given filename.

        :param filename: path from where to load the model
        :return: the loaded model
        """

        with open(filename, "rb") as f:
            return pickle.load(f)
