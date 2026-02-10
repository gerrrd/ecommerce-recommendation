import json
import os
import pickle
from typing import Dict, List, Tuple

import requests
import streamlit as st

from ecommercerecommendation.utils.data import clean_entry

MODELS = ["association_rules", "collaborative_filtering", "tfidf", "llm"]
URL_API = os.environ.get("API_URL", "http://0.0.0.0:8080")

st.set_page_config(layout="wide")
st.sidebar.write("# Recommendation API Simulator")


@st.cache_data
def load_data() -> Tuple[Dict[int, List[str]], Dict[int, str], List[int]]:
    # read data for easier stock searches
    with open("descriptions.pkl", "rb") as file:
        description_ = pickle.load(file)
    with open("unique_descriptions.pkl", "rb") as file:
        unique_descriptions_ = pickle.load(file)
    with open("customers.pkl", "rb") as file:
        customers_ = pickle.load(file)
    return description_, unique_descriptions_, customers_


description, unique_descriptions, customers = load_data()


def get_stock():
    return st.sidebar.multiselect(
        label="Select StockCode", options=description.keys()
    )


def get_customer():
    return int(
        st.sidebar.selectbox(label="Select Customer", options=customers)
    )


def call_api():
    if st.session_state["model"] == "association_rules":
        input_request = {
            "recommender_model": "association_rules",
            "selected_products": st.session_state.get("stock_code", []),
        }
    elif st.session_state["model"] == "collaborative_filtering":
        input_request = {
            "recommender_model": "collaborative_filtering",
            "customer_id": st.session_state.get("customer_id", -1),
            "selected_products": st.session_state.get("stock_code", []),
        }
    else:  # llm or tfidf
        input_request = {
            "recommender_model": st.session_state["model"],
            "selected_products": st.session_state.get("stock_code", []),
            "description": st.session_state.get("description", ""),
        }

    data_json = json.dumps(input_request)
    response = requests.post(f"{URL_API}/predict", data=data_json)

    # let's skip for now error/exception management
    return response.json()


st.session_state["model"] = st.sidebar.radio(
    "Model:",
    MODELS,
    key="model selection",
    # on_change=clear_content,
)

if st.session_state["model"] == "association_rules":
    st.session_state["stock_code"] = get_stock()
elif st.session_state["model"] == "collaborative_filtering":
    collab_choice = st.sidebar.radio(
        "Customer or product?", ["CustomerId", "StockCode"]
    )
    if collab_choice == "CustomerId":
        st.session_state["customer_id"] = get_customer()
        st.session_state["stock_code"] = []
    else:  # StockCode
        st.session_state["stock_code"] = get_stock()
        st.session_state["customer_id"] = -1
else:  # llm or tfidf
    collab_choice = st.sidebar.radio(
        "Product or free description", ["StockCode", "Description"]
    )
    if collab_choice == "StockCode":
        st.session_state["stock_code"] = get_stock()
        st.session_state["description"] = ""
    else:  # Description
        st.session_state["description"] = clean_entry(
            st.sidebar.text_input("Description:").upper()
        )
        st.session_state["stock_code"] = []

if st.sidebar.button(
    f"Call recommender API for **{st.session_state['model']}** prediction."
):
    st.session_state["response"] = call_api()

if st.sidebar.button("Clear"):
    st.session_state["response"] = None

if st.session_state.get("response", None) is not None:
    st.write(
        f"#### Results from the **{st.session_state['model']}**"
        " recommendation."
    )
    unique_bool = st.checkbox("Unique description", value=False)
    st.write(
        ", ".join(
            [str(code) for code in st.session_state["response"]["output"]]
        )
    )
    st.write(
        {
            key: (unique_descriptions if unique_bool else description)[key]
            for key in st.session_state["response"]["output"]
        }
    )
    st.write("---")
    st.write("#### Input request sent")
    st.write(st.session_state["response"]["input"])
else:
    st.write("Waiting for results.")
