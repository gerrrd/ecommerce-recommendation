"""
This script has the class ARLLMLLMRecommender for the association rules based
recommendations extended by an LLM-embedding on both sides, before and after
Association Rules.

"""

from __future__ import annotations

import pickle
from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ecommercerecommendation.models.arllmrecommender import ARLLMRecommender
from ecommercerecommendation.models.llmrecommender import LLMRecommender
from ecommercerecommendation.utils.constants import LOCAL_MODEL_NAME


def merge_code_description(row: pd.Series) -> str:
    """
    Utility function for merging to columns when predicting.

    :param row: pd.Series
    :return:
    """
    return str(row["StockCode"]) + " " + row["Description"]


class ARLLMLLMRecommender(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str = LOCAL_MODEL_NAME, top_n: int = 5):
        self.rules = None
        self.model_name = model_name
        self.ar_llm_recommender = ARLLMRecommender(
            model_name=model_name, top_n=top_n
        )
        # TODO: like this we store the LLM model twice, it should be simplified
        self.llm_recommender = LLMRecommender(
            model_name=LOCAL_MODEL_NAME, top_n=top_n
        )
        self.top_n = top_n

    def fit(self, X, y=None) -> ARLLMLLMRecommender:
        self.ar_llm_recommender.fit(X=X, y=y)
        self.llm_recommender.fit(X=X, y=y)
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

        ar_llm_recommendation = self.ar_llm_recommender.get_recommendations(
            current_selection=current_selection,
            min_confidence=min_confidence,
        )

        return sorted(
            set(
                sum(
                    [
                        self.llm_recommender.predict_element(
                            int(rec.split(" ")[0])
                        )
                        .apply(merge_code_description, axis=1)
                        .to_list()
                        for rec in ar_llm_recommendation
                    ],
                    [],
                )
            )
        )

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
