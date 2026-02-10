"""
Intput request and output schemas for the API

"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class RecommenderModel(str, Enum):
    association_rules = "association_rules"
    collaborative_filtering = "collaborative_filtering"
    tfidf = "tfidf"
    llm = "llm"


class InputRequest(BaseModel):
    recommender_model: RecommenderModel = Field(
        description="Chosen model to recommend."
    )
    customer_id: Optional[int] = Field(
        description="Optionally given customerId for collaborative filtering.",
        default=-1,
    )
    selected_products: List[int] = Field(
        description="Selected products to predict for."
    )
    description: Optional[str] = Field(
        description="Text input in case of predicting for a new product.",
        default="",
    )


class Output(BaseModel):
    results: List[int] = Field(
        description="List of recommended products by StockID.",
    )


class Response(BaseModel):
    input: InputRequest
    output: Output
    estimation_id: str = Field(
        description="Estimation uuid or any string if given.",
    )
    processing_time: int = Field(
        description="Processing time in seconds.",
    )
