from typing import Dict, Union
from pydantic import BaseModel


# RestAPI routing
class BccaasResults(BaseModel):
    completed_timestamp: str
    results: Dict[str, Dict[str, Union[int, str]]] = None
