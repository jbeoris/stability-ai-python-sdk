import requests
from enum import Enum
from pydantic import BaseModel
from typing import (
    Optional,
    List
)

from stability_ai.util import (
    make_url,
    APIVersion,
    StabilityAIError
)
from stability_ai.client_interface import ClientInterface

resource = 'user'

class Endpoint(str, Enum):
    ACCOUNT = 'account'
    BALANCE = 'balance'

class Organization(BaseModel):
    id: str
    is_default: bool
    name: str
    role: str

class AccountResponse(BaseModel):
    email: str
    id: str
    organizations: List[Organization]
    profile_picture: Optional[str]

class BalanceResponse(BaseModel):
    credits: float

class User():
    def __init__(self, client: ClientInterface) -> None:
        self.client = client
  
    def account(self) -> AccountResponse:
        url = make_url(APIVersion.V1, resource=resource, endpoint=Endpoint.ACCOUNT)
        response = requests.get(url, headers=self.client.headers)

        if response.status_code == 200:
            try:
                return AccountResponse(**response.json())
            except ValueError as e:
                raise StabilityAIError(response.status_code, f"Failed to parse response: {str(e)}", response.text)
        else:
            raise StabilityAIError(response.status_code, 'Failed to gather user balance', response.text)
  
    def balance(self) -> BalanceResponse:
        url = make_url(APIVersion.V1, resource=resource, endpoint=Endpoint.BALANCE)
        response = requests.get(url, headers=self.client.headers)

        if response.status_code == 200:
            try:
                return BalanceResponse(**response.json())
            except ValueError as e:
                raise StabilityAIError(response.status_code, f"Failed to parse response: {str(e)}", response.text)
        else:
            raise StabilityAIError(response.status_code, 'Failed to gather user balance', response.text)