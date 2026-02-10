cp notebooks/*.pkl app/models/
# we remove the LLM and AR_LLM models as they won't be used for now in the API
rm app/models/llm_model.pkl
rm app/models/ARLLM_model.pkl
cp notebooks/cache/*.pkl app/models/
cp notebooks/cache/*.pkl ui/
