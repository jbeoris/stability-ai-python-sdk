import requests
from enum import Enum
from pydantic import BaseModel
from typing import (
    List
)

from stability_ai.util import (
    make_url,
    APIVersion
)

from stability_ai.error import (
    StabilityAIError
)
from stability_ai.client_interface import ClientInterface

resource = 'engines'

class Endpoint(str, Enum):
    LIST = 'list'

class EngineType(str, Enum):
    AUDIO = 'AUDIO'
    CLASSIFICATION = 'CLASSIFICATION'
    PICTURE = 'PICTURE'
    STORAGE = 'STORAGE'
    TEXT = 'TEXT'
    VIDEO = 'VIDEO'

class Engine(BaseModel):
    description: str
    id: str
    name: str
    type: EngineType

class ListResponse(BaseModel):
    engines: List[Engine]

class Engines():
    def __init__(self, client: ClientInterface) -> None:
        self.client = client
  
    def list(self) -> ListResponse:
        url = make_url(APIVersion.V1, resource=resource, endpoint=Endpoint.LIST)
        response = requests.get(url, headers=self.client.headers)

        if response.status_code == 200:
            try:
                engines_data = response.json()
                if isinstance(engines_data, list):
                    return ListResponse(engines=[Engine(**engine) for engine in engines_data])
                else:
                    raise ValueError("Unexpected response format")
            except ValueError as e:
                raise StabilityAIError(response.status_code, f"Failed to parse response: {str(e)}", response.text)
        else:
            raise StabilityAIError(response.status_code, 'Failed to list engines', response.text)