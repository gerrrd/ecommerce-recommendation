"""
This script has the class representation of the GenAI API's JSON response.

"""

from typing import List

from pydantic import BaseModel


class JSONResponse(BaseModel):
    recommendations: List[str]
