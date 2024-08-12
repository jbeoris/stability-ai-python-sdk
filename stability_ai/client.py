from stability_ai.client_interface import ClientInterface
from stability_ai.v1 import V1
from typing import ( Optional )

class Client(ClientInterface):
    def __init__(
        self,
        api_key: str,
        organization: Optional[str] = None,
        client_id: Optional[str] = None,
        client_version: Optional[str] = None,
    ) -> None:
        self.api_key = api_key
        self.organization = organization
        self.client_id = client_id
        self.client_version = client_version

    @property
    def headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}"
        }

    @property
    def v1(self):
        return V1(client=self)