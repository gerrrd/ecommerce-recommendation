"""
This script has a utility function for the API.

"""

from typing import Dict, Union

from ecommercerecommendation.models.arrecommender import ARRecommender
from ecommercerecommendation.models.cfrecommender import CFRecommender
from ecommercerecommendation.models.tfidfrecommender import TfIdfRecommender

# from ecommercerecommendation.models.llmrecommender import LLMRecommender


def load_models() -> Dict[
    str,
    Union[ARRecommender, CFRecommender, TfIdfRecommender, None],
]:
    """
    Loads and returns the models used in predictions. Skips the LLM model.

    :return:
    """
    return {
        "association_rules": ARRecommender().load_pickle(
            "models/AR_model.pkl"
        ),
        "collaborative_filtering": CFRecommender().load_pickle(
            "models/CF_model.pkl"
        ),
        # due to memory issues on my old laptop, we skip LLM recommendation
        # in the API. It is still available in the notebook.
        "llm": None,
        "tfidf": TfIdfRecommender().load_pickle("models/TF_model.pkl"),
    }
