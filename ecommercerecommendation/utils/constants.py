"""
This script has constants to be imported by other modules.

"""

BULLET_POINT = "\n- "

REMOVE_CHARS = ".,'\"&/- !()"

PROMPT_TEMPLATE = """# Lines starting with "#" will be removed like comments.
# Prompt for the GenAI API call to select the final recommendations from a previously curated candidate list, given the already selected (by the user) products:
# - recommendations: List[str] - final recommendation
#
# Parameters (in curly brackets as in Python's formatted strings)
# - selected: "\\n - "-separated list of already selected products
# - top_n: maximum number of recommendation needed
# - recommendation_candidates: "\\n - "-separated list of possible candidates to be recommended

<role>You are part of a bundle recommender system. You are given a list of already selected products by the user, and also a list of candidates to be recommended to buy together.</role>

<task>Your task is to select those elements of the candidate list, which are the most probable to be bought together, knowing what the user have already selected.
The user has already selected the following products:
- "{selected}"
Output is in JSON format with one key: “recommendations” (list of strings), such as:
{
    "recommendations": [],
}

You are a JSON generator. Return only valid JSON.</task>

<candidate_list>
- "{recommendation_candidates}"
</candidate_list>

<recommendations rules>
Select up to {top_n} elements of the candidate_list that are the most probable to buy together with the already selected products. Each returned recommendation string has to coincide with one of the candidates.
</recommendations rules>
"""
