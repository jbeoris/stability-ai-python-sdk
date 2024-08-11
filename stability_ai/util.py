import json
from enum import Enum

STABILITY_AI_BASE_URL = "https://api.stability.ai"

class APIVersion(Enum):
    V1 = "v1"
    V2_BETA = "v2beta"

class OutputFormat(Enum):
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"

def make_url(
    version: APIVersion,
    resource: str,
    endpoint: str
) -> str:
    return f"{STABILITY_AI_BASE_URL}/{version.value}/{resource}{f'/{endpoint}' if endpoint.__len__() > 0  else ''}"

class StabilityAIErrorName(Enum):
    INVALID_REQUEST_ERROR = 'StabilityAIInvalidRequestError'
    UNAUTHORIZED_ERROR = 'StabilityAIUnauthorizedError'
    CONTENT_MODERATION_ERROR = 'StabilityAIContentModerationError'
    RECORD_NOT_FOUND_ERROR = 'StabilityAIRecordNotFoundError'
    UNKNOWN_ERROR = 'StabilityAIUnknownError'

class StabilityAIError(Exception):
    def __init__(self, status: int, message: str, data: any = None):
        try:
            data_message = json.dumps(data)
        except:
            data_message = ''

        full_message = f"{message}: {data_message}"
        super().__init__(full_message)

        name = StabilityAIErrorName.UNKNOWN_ERROR

        if status == 400:
            name = StabilityAIErrorName.INVALID_REQUEST_ERROR
        elif status == 401:
            name = StabilityAIErrorName.UNAUTHORIZED_ERROR
        elif status == 403:
            name = StabilityAIErrorName.CONTENT_MODERATION_ERROR
        elif status == 404:
            name = StabilityAIErrorName.RECORD_NOT_FOUND_ERROR

        self.name = name