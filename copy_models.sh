cp notebooks/*.pkl app/models/
# we remove the LLM model as it won't be used for now in the API
rm app/models/llm_model.pkl
cp notebooks/cache/*.pkl app/models/
cp notebooks/cache/*.pkl ui/
