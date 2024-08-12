from stability_ai.client_interface import ClientInterface
from stability_ai.v1.engines import Engines
from stability_ai.v1.user import User
from stability_ai.v1.generation import Generation

class V1:
    def __init__(
        self,
        client: ClientInterface
    ) -> None:
        self.client = client

    @property
    def engines(self):
        return Engines(client=self.client)

    @property
    def user(self):
        return User(client=self.client)

    @property
    def generation(self):
        return Generation(client=self.client)