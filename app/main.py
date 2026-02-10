"""
Code of the fastAPI component serving predictions.

"""

import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from schemas import InputRequest, Response

from ecommercerecommendation.models.predict import load_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    print("Loading models...")
    app.state.models = load_models()

    yield

    # shutdown
    print("Shutting down and clearing memory...")
    app.state.models = None


app = FastAPI(lifespan=lifespan)


@app.post("/predict", response_model=Response)
def predict(input_request: InputRequest):
    input_request = input_request.model_dump()
    model_name = input_request["recommender_model"]
    customer_id = input_request["customer_id"]
    selected_products = input_request["selected_products"]
    description = input_request["description"]

    estimation_id = str(uuid4())
    start = time.time()
    logging.info(f"{estimation_id} | Predicting with {model_name}.")
    if model_name == "association_rules":
        output = app.state.models[model_name].get_recommendations(
            current_selection=selected_products
        )
    elif model_name == "collaborative_filtering":
        output = app.state.models[model_name].get_recommendations(
            target_user=customer_id,
            selected_stocks=selected_products,
        )
    elif model_name == "tfidf":
        if len(selected_products) == 0:
            output = (
                app.state.models[model_name]
                .predict_element(stock_code=-1, description=description)[
                    "StockCode"
                ]
                .drop_duplicates()
                .to_list()
            )
        else:
            output = list(
                set(
                    sum(
                        [
                            app.state.models[model_name]
                            .predict_element(
                                stock_code=stock_code, description=description
                            )["StockCode"]
                            .drop_duplicates()
                            .to_list()
                            for stock_code in selected_products
                        ],
                        [],
                    )
                )
            )

    else:  # == "llm"
        logging.info(
            f"{estimation_id} | LLM based predictions are not "
            "implemented, returning [0]."
        )

        output = [0]

    processing_time = int(time.time() - start)

    response = {
        "input": input_request,
        "output": output,
        "estimation_id": estimation_id,
        "processing_time": processing_time,
    }
    logging.info(f"{estimation_id} | Finished prediction.")
    return JSONResponse(content=jsonable_encoder(response))


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    LOGGER = logging.getLogger("ecommercerecommendation")
    LOGGER.info("Starting the app.")

    logging.getLogger("uvicorn.access")

    uvicorn.run("main:app", host="0.0.0.0", port=8080)
