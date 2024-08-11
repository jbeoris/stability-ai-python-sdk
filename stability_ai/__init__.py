from dotenv import load_dotenv
import os

from stability_ai.client import Client

load_dotenv()

api_key = os.environ.get("STABILITY_AI_API_KEY")

default_client = Client(api_key=api_key)

v1 = default_client.v1