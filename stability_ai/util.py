import uuid
import os
import tempfile
import requests
import base64
from pathlib import Path
from enum import Enum
from pydantic import BaseModel
from typing import (Optional, Union, Set)
from urllib.parse import urlparse

STABILITY_AI_BASE_URL = "https://api.stability.ai"

class APIVersion(Enum):
    V1 = "v1"
    V2_BETA = "v2beta"

class OutputFormat(Enum):
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    MP4 = "mp4"
    GLB = "glb"

class ContentType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    GL3D = "3d"

DEFAULT_OUTPUT_FORMAT = OutputFormat.PNG

class StabilityAIContentResponse(BaseModel):
    filepath: str
    filename: str
    content_type: ContentType
    output_format: OutputFormat
    content_filtered: bool
    errored: bool
    seed: int

class StabilityAIStatus(Enum):
    IN_PROGRESS = "in-progress"

class StabilityAIStatusResult(BaseModel):
    id: str
    status: StabilityAIStatus

def make_url(
    version: APIVersion,
    resource: str,
    endpoint: str
) -> str:
    return f"{STABILITY_AI_BASE_URL}/{version.value}/{resource}{f'/{endpoint}' if endpoint.__len__() > 0  else ''}"

def is_valid_http_url(resource: str) -> bool: 
    try:
        result = urlparse(resource)
        return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
    except ValueError:
        return False

def is_valid_file(resource: str) -> bool:
    try:
        return Path(resource).is_file()
    except:
        return False

class ImagePathType(Enum):
    DOWNLOAD = "download"
    LOCAL = "local"

class ImagePath:
    resource: str
    type: ImagePathType
    download_filepath: Optional[str]

    def __init__(self, resource:str) -> None:
        self.resource = resource
        self.download_filepath = None
        if is_valid_http_url(resource=resource):
            self.type = ImagePathType.DOWNLOAD
        elif is_valid_file(resource=resource):
            self.type = ImagePathType.LOCAL
        else:
            raise Exception("Invalid image resource. Must be a local filepath or public URL.")
        
    def filepath(self) -> str:
        match self.type:
            case ImagePathType.LOCAL:
                return self.resource
            case ImagePathType.DOWNLOAD:
                if self.download_filepath is not None:
                    return self.download_filepath
                else:
                    self.download_filepath = download_image(url=self.resource)
                    return self.download_filepath
        
    def cleanup(self) -> None:
        match self.type:
            case ImagePathType.DOWNLOAD:
                if self.download_filepath is not None:
                    delete_file(filepath=self.download_filepath)
            case _:
                return
            
def get_file_extension(url) -> str:
    path = urlparse(url).path
    path_without_params, _ = os.path.splitext(path.split('?')[0])
    _, file_extension = os.path.splitext(path_without_params)
    return file_extension

def delete_file(filepath: str):
    try:
        os.remove(filepath)
        print(f"Successfully deleted the file: {filepath}")
    except FileNotFoundError:
        print(f"The file {filepath} does not exist.")
    except PermissionError:
        print(f"Permission denied: Unable to delete the file {filepath}")
    except Exception as e:
        print(f"An error occurred while deleting the file {filepath}: {str(e)}")

def download_image(url: str):
    filename = f"{uuid.uuid4()}.{get_file_extension(url)}"

    temp_dir = get_persistent_temp_dir()
    filepath = os.path.join(temp_dir, filename)

    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, 'wb') as file:
            file.write(response.content)

        return filepath
    else:
        raise Exception(f"Failed to download image. Url: {url}, Status code: {response.status_code}")
        
class FinishReason(Enum):
    SUCCESS = "SUCCESS"
    CONTENT_FILTERED = "CONTENT_FILTERED"
    ERROR = "ERROR"

def filter_params(params: dict, filters: Set[str]):
    return {k: v.value if isinstance(v, Enum) else v for k, v in params.items() if k not in filters}

def get_content_type(output_format: OutputFormat):
    match output_format:
        case OutputFormat.JPEG | OutputFormat.PNG | OutputFormat.WEBP:
            return ContentType.IMAGE
        case OutputFormat.MP4:
            return ContentType.VIDEO
        case OutputFormat.GLB:
            return ContentType.GL3D
        
def get_persistent_temp_dir():
    temp_base = tempfile.gettempdir()
    temp_dir = os.path.join(temp_base, "stability_ai")
    os.makedirs(temp_dir, exist_ok=True)
    
    return temp_dir
        
def process_content_response(data: dict, output_format: OutputFormat, resource: str):
    file_data = data.get('video') if output_format == OutputFormat.MP4 else data.get('image')
    if file_data is None:
        file_data = data.get('base64')
    if file_data is None:
        raise Exception('No valid data found in the response')
    
    finish_reason = FinishReason(data.get('finish_reason', FinishReason.SUCCESS))
        
    filename = f"{resource}_{uuid.uuid4()}.{output_format.value}"

    temp_dir = get_persistent_temp_dir()
    filepath = os.path.join(temp_dir, filename)
    with open(filepath, 'wb') as file:
        file.write(base64.b64decode(file_data))

    return StabilityAIContentResponse(
        filepath=filepath,
        filename=filename,
        content_type=get_content_type(output_format=output_format),
        output_format=output_format,
        content_filtered=True if finish_reason == FinishReason.CONTENT_FILTERED else False,
        errored=True if finish_reason == FinishReason.ERROR else False,
        seed=data.get("seed", 0)
    )
        
def process_array_buffer_response(data: Union[str, bytes], output_format: OutputFormat, resource: str):
    filename = f"{resource}_{uuid.uuid4()}.{output_format}"

    temp_dir = get_persistent_temp_dir()
    filepath = os.path.join(temp_dir, filename)
    with open(filepath, 'wb') as file:
        file.write(data)

    return StabilityAIContentResponse(
        filepath=filepath,
        filename=filename,
        content_type=get_content_type(output_format=output_format),
        output_format=output_format,
        content_filtered=False,
        errored=False,
        seed=0
    )