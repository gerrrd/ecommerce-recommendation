"""
This script has the class ARLLMLLMRecommender for the association rules based
recommendations extended by an LLM-embedding on both sides, before and after
Association Rules.

"""

from __future__ import annotations

import pickle
from typing import List, Tuple

from sklearn.base import BaseEstimator, TransformerMixin

from ecommercerecommendation.models.arllmrecommender import ARLLMRecommender

# from sklearn.metrics.pairwise import euclidean_distances


SIMILAR_CUSTOMERS = 10


class ARLLMLLMRecommender(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", top_n: int = 5):
        self.rules = None
        self.model_name = model_name
        self.ar_llm_recommender = ARLLMRecommender(
            model_name=model_name, top_n=top_n
        )
        self.top_n = top_n

    def fit(self, X, y=None) -> ARLLMLLMRecommender:
        self.ar_llm_recommender.fit(X=X, y=y)
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
        Method to perform the recommendation with using the ARLLMRecommender,
        enriching its results by semantically similar products.

        :param current_selection: descriptions of currently selected stocks
        :param min_confidence: to use rules over a certain confidence level
        :return: the recommended items
        """

        ar_llm_recommendation = self.ar_llm_recommender.get_recommendation(
            current_selection=current_selection,
            min_confidence=min_confidence,
        )

        # TODO: enrich the prediction with similar stocks their by embeddings
        # ar_llm_recommendation.append(  __some more products__  )

        return ar_llm_recommendation

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
    def load_pickle(filename) -> ARLLMLLMRecommender:
        """
        Load the model from the given filename.

        :param filename: path from where to load the model
        :return: the loaded model
        """

        with open(filename, "rb") as f:
            return pickle.load(f)
