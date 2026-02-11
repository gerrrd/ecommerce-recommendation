# Introduction

This repository contains a recommender system for the following Ecommerce dataset on Kaggle: https://www.kaggle.com/datasets/carrie1/ecommerce-data

## Idea

Five recommender systems are implemented (and deployed locally in a fastAPI together with a UI with docker-compose): Association Rules, Collaborative Filtering, 2 content-based recommenders with tf-idf and LLM-embedding, and an extension of Association Rules using LLM-embedding.

A simple streamlit app is also implemented as simulator to test the different models. I have used an old laptop with limited resources, therefore I have not included the models with LLM in the API and the UI. They can be added easily or tested in a notebook.

## Set-up

### Locally

In order to have everything running the repo, after cloning it, run the following commands.
They will create a conda environment (you need to have conda or miniconda installed),
activate it, install requirements and make precommit (to make sure all committed code is well formatted).

```shell
make create_environment
conda activate ecomrec
make requirements
make precommit
```

I have used my old, 2012 Macbook pro with Intel processor.
Therefore, I had to use older versions of Python and other packages.
If you use a newer computer, you may have to adjust the requirements and/or the Dockerfile.

### Google Gemini

If you do not have any, generate your Google Gemini API key at https://aistudio.google.com/app/api-keys and save it in `~/.config/.gemini`. Then, and run
```shell
GEMINI_API_KEY=`cat ~/.config/.gemini`
export GEMINI_API_KEY
```

# Solution

I have implemented 5 different recommender systems and services to perform it.

## Models

I have chosen the following 5 models:

- *Association rules:* based market basket analysis -- which products are generally sold together. Fast, easy to interpret (rules are human-readable)
- *Collaborative filtering:* represents user behaviour, what other users with similar basket have bought
- *Tf-idf (Term Frequency-Inverse Document Frequency):* content-based, using text (description) based on their frequency (per word), fast, but depends highly on the training set (i.e. words)
- *LLM (Sentence Transformers):* content-based, using the semantic meaning, it can predict for new words/descriptions, computationally heavy
- *LLM-embedding-boosted Association rules:* instead of direct matching among the rules, we use the (Euclidian) distance from their text, gaining recommendations also having synonyms or similar words
- #TODO *LLM-Association rules-LLM:* as the previous one but also add recommendations based on the embedding distance from the results.
- #TODO2 *LLM-Association rules-LLM-LLM:* as the previous one but followed by an API call to Google Gemini 1.5 Flash via `google-genai` to understand which of the recommended elements are most probable/logical lifestyle bundle.

## Architecture

It has two components, three parts in total.

Firstly, there are 5 notebooks with
- 1_investigation: getting to know the data, features
- 2_text_cleaning: to understand and get rid of non-alphanumerical characters. Text was relatively clean, I did not implement a sophisticated text preprocessing, I have used everything upper case, as it was given
- 3_modelling: model training and saving them
- 4_test_api: to be able to test the API quickly in a notebook
- 5_llm_association: for the last model (association rules with LLM embedding)

Then, there is an API serving the predictions based on the 4 chosen models. As my laptop has limited resources, I have skipped the LLM based predictions, but it can be included with minor change to the code.

Lastly, there is a `streamlit` app as a Recommender System Simulator, to be able to test it clicking around, writing our ad-hoc descriptions.

## Investigation and training

### Notebooks
In order to generate all the models (pickle files) necessary to run the API, run all lines in notebooks 1, 2 and 3.
There are comments about decisions taken among the code lines.

In some of the models, I had to downsample the training and/or the test set, that takes away statistical credibility, but it was necessary on my laptop I have used. In a real life scenario, we would use such instances that can handle the given data.

### Copy models
When done, you can copy all the models to their corresponding folders with `copy_models.sh`.
It will copy the used pickle files for both the API and the streamlit app (UI).

## Evaluation - TODO

Proper evaluation of all models.

# Run the components

```shell
docker-compose up
# use it
docker-compose down
```

When everything is up, you can test the API with notebook 4, or you can open the streamlit app at
http://localhost:8501/

If the API is running, you can find the documentation of the API at
http://0.0.0.0:8080/docs

# Notes

I have used GenAI input as
- part of Google searches,
- to brainstorm certain topics and
- for requirements to use relatively new packages in the old laptop
