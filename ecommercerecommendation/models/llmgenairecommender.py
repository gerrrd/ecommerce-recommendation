"""
This script has the class LLMGenAIRecommender, that uses ARLLMLLMRecommender,
and then calls the Gemini API to finalize the recommendation.

"""

from __future__ import annotations

import json
import pickle
from typing import List

from google import genai
from sklearn.base import BaseEstimator, TransformerMixin

from ecommercerecommendation.models.arllmllmrecommender import (
    ARLLMLLMRecommender,
)
from ecommercerecommendation.models.json_response import JSONResponse
from ecommercerecommendation.utils.constants import (
    BULLET_POINT,
    GEMINI_MODEL_ID,
    LOCAL_MODEL_NAME,
    PROMPT_TEMPLATE,
)
from ecommercerecommendation.utils.prompts import Prompt


class LLMGenAIRecommender(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        local_model_name: str = LOCAL_MODEL_NAME,
        gemini_model_id: str = GEMINI_MODEL_ID,
        top_n: int = 5,
    ):
        self.rules = None
        self.local_model_name = local_model_name
        self.gemini_model_id = gemini_model_id
        self.ar_llm_recommender = ARLLMLLMRecommender(
            model_name=local_model_name, top_n=top_n
        )
        self.ar_llm_llm_recommender = ARLLMLLMRecommender(
            model_name=local_model_name, top_n=top_n
        )
        self.top_n = top_n

    def fit(self, X, y=None) -> LLMGenAIRecommender:
        self.ar_llm_llm_recommender.fit(X=X, y=y)
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
        Method to perform the recommendation with using the
        ARLLMLLMRecommender, selecting from its results the most probably
        meaningful ones based on Gemini API call.

        :param current_selection: descriptions of currently selected stocks
        :param min_confidence: to use rules over a certain confidence level
        :return: the recommended items
        """

        # even though all these steps could be written as one big formula,
        # we keep them separated for better readability
        client = genai.Client()

        candidates = self.ar_llm_llm_recommender.get_recommendations(
            current_selection=current_selection,
            min_confidence=min_confidence,
        )

        # removing stock codes and converting int bullet points
        candidates_str = BULLET_POINT.join(
            [" ".join(c.split(" ")[1:]) for c in candidates]
        )
        selected_str = (BULLET_POINT.join(current_selection),)

        prompt_format = {
            "top_n": 5,
            "selected": selected_str,
            "recommendation_candidates": candidates_str,
        }

        prompt = Prompt(prompt_str=PROMPT_TEMPLATE, replace=prompt_format)

        response = client.models.generate_content(
            model=self.gemini_model_id,
            contents=prompt.to_string(),
            config={
                "response_mime_type": "application/json",
                "response_json_schema": JSONResponse.model_json_schema(),
            },
        )

        # TODO: exception handling in case it's not the correct form:
        return json.loads(response.text).get("recommendations", [])

    def save_pickle(self, filename: str):
        """
        Save the model to the given filename.

        :param filename: path where to store the model
        """

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_pickle(filename) -> LLMGenAIRecommender:
        """
        Load the model from the given filename.

        :param filename: path from where to load the model
        :return: the loaded model
        """

        with open(filename, "rb") as f:
            return pickle.load(f)
