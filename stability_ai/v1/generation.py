import requests
from enum import Enum
from typing import (
    List,
    Optional,
    TypedDict
)
from typing_extensions import Unpack
from stability_ai.util import (
    make_url,
    process_content_response,
    APIVersion,
    OutputFormat,
    ImagePath,
    filter_params,
    StabilityAIContentResponse
)
from stability_ai.error import (
    StabilityAIError
)
from stability_ai.client_interface import ClientInterface

resource = 'generation'

class Endpoint(str, Enum):
    TEXT_TO_IMAGE = "text-to-image",
    IMAGE_TO_IMAGE = "image-to-image",
    IMAGE_TO_IMAGE_UPSCALE = "image-to-image/upscale",
    IMAGE_TO_IMAGE_MASKING = "image-to-image/masking",

class EngineId(str, Enum):
    ESRGAN_V1_X2PLUS = "esrgan-v1-x2plus"
    STABLE_DIFFUSION_XL_1024_V0_9 = "stable-diffusion-xl-1024-v0-9"
    STABLE_DIFFUSION_XL_1024_V1_0 = "stable-diffusion-xl-1024-v1-0"
    STABLE_DIFFUSION_V1_6 = "stable-diffusion-v1-6"
    STABLE_DIFFUSION_512_V2_1 = "stable-diffusion-512-v2-1"
    STABLE_DIFFUSION_XL_BETA_V2_2_2 = "stable-diffusion-xl-beta-v2-2-2"

class ClipGuidancePreset(str, Enum):
    FAST_BLUE = "FAST_BLUE"
    FAST_GREEN = "FAST_GREEN"
    NONE = "NONE"
    SIMPLE = "SIMPLE"
    SLOW = "SLOW"
    SLOWER = "SLOWER"
    SLOWEST = "SLOWEST"

class Sampler(str, Enum):
    DDIM = "DDIM"
    DDPM = "DDPM"
    K_DPMPP_2M = "K_DPMPP_2M"
    K_DPMPP_2S_ANCESTRAL = "K_DPMPP_2S_ANCESTRAL"
    K_DPM_2 = "K_DPM_2"
    K_DPM_2_ANCESTRAL = "K_DPM_2_ANCESTRAL"
    K_EULER = "K_EULER"
    K_EULER_ANCESTRAL = "K_EULER_ANCESTRAL"
    K_HEUN = "K_HEUN"
    K_LMS = "K_LMS"

class StylePreset(str, Enum):
    MODEL_3D = "3d-model"
    ANALOG_FILM = "analog-film"
    ANIME = "anime"
    CINEMATIC = "cinematic"
    COMIC_BOOK = "comic-book"
    DIGITAL_ART = "digital-art"
    ENHANCE = "enhance"
    FANTASY_ART = "fantasy-art"
    ISOMETRIC = "isometric"
    LINE_ART = "line-art"
    LOW_POLY = "low-poly"
    MODELING_COMPOUND = "modeling-compound"
    NEON_PUNK = "neon-punk"
    ORIGAMI = "origami"
    PHOTOGRAPHIC = "photographic"
    PIXEL_ART = "pixel-art"
    TILE_TEXTURE = "tile-texture"

class TextPrompt(TypedDict):
    text: str
    weight: float

class V1GenerationRequiredParams(TypedDict):
    engine_id: EngineId
    text_prompts: List[TextPrompt]

class V1GenerationOptionalParams(TypedDict):
    cfg_scale: Optional[float]
    clip_guidance_preset: Optional[ClipGuidancePreset]
    sampler: Optional[Sampler]
    samples: Optional[int]
    seed: Optional[int]
    steps: Optional[int]
    style_preset: Optional[StylePreset]
    extra: Optional[dict]

class TextToImageOptions(V1GenerationRequiredParams, V1GenerationOptionalParams):
    height: Optional[int]
    width: Optional[int]

class ImageToImageMode(str, Enum):
    IMAGE_STRENGTH = 'IMAGE_STRENGTH'
    STEP_SCHEDULE = 'STEP_SCHEDULE'

class ImageToImageStrengthOptions(TypedDict):
    mode: ImageToImageMode = ImageToImageMode.IMAGE_STRENGTH
    image_strength: float

class ImageToImageStepScheduleOptions(TypedDict):
    mode: ImageToImageMode = ImageToImageMode.STEP_SCHEDULE
    step_schedule_start: float
    step_schedule_end: float

class ImageToImageOptions(V1GenerationRequiredParams, V1GenerationOptionalParams, ImageToImageStrengthOptions, ImageToImageStepScheduleOptions):
    init_image: str

class ImageToImageUpscaleOptions(TypedDict):
    image: str
    height: Optional[int]
    width: Optional[int]

class ImageToImageMaskSource(str, Enum):
    MASK_IMAGE_WHITE = 'MASK_IMAGE_WHITE'
    MASK_IMAGE_BLACK = 'MASK_IMAGE_BLACK'
    INIT_IMAGE_ALPHA = 'INIT_IMAGE_ALPHA'

class ImageToImageMaskingOptions(V1GenerationRequiredParams, V1GenerationOptionalParams):
    init_image: str
    mask_source: Optional[ImageToImageMaskSource]
    mask_image: Optional[str]

def get_multi_part_text_prompts(text_prompts: List[TextPrompt]):
    multi_part_text_prompts = {}

    for index, text_prompt in enumerate(text_prompts):
        multi_part_text_prompts[f'text_prompts[{index}][text]'] = text_prompt.get('text')
        multi_part_text_prompts[f'text_prompts[{index}][weight]'] = text_prompt.get('weight')

    return multi_part_text_prompts

def process_articafts(artifacts: List[dict], endpoint: Endpoint) -> List[StabilityAIContentResponse]:
    results: List[StabilityAIContentResponse] = []

    for artifact in artifacts:
        results.append(
            process_content_response(
                data=artifact,
                output_format=OutputFormat.PNG,
                resource=f"v1_generation_{endpoint.replace('/', '_').replace('-', '_')}"
            )
        )
    
    return results

class Generation():
    def __init__(self, client: ClientInterface) -> None:
        self.client = client
  
    def text_to_image(
        self, 
        **params: Unpack[TextToImageOptions]
    ) -> StabilityAIContentResponse:
        url = make_url(
            version=APIVersion.V1, 
            resource=resource, 
            endpoint=f"{params.get('engine_id')}/{Endpoint.TEXT_TO_IMAGE}"
        )
        
        filtered_params = filter_params(params=params, filters={'engine_id'})

        response = requests.post(
            url,
            json={
                **filtered_params
            },
            headers={
                **self.client.headers,
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
        )
        
        if response.status_code == 200 \
            and isinstance(response.json().get('artifacts'), list):
            return process_articafts(
                artifacts=response.json().get('artifacts'),
                endpoint=Endpoint.TEXT_TO_IMAGE
            )

        raise StabilityAIError(
            response.status_code,
            'Failed to run v1 generation text to image',
            response.json()
        )
  
    def image_to_image(
        self, 
        **params: Unpack[ImageToImageOptions]
    ) -> StabilityAIContentResponse:
        image_path = ImagePath(params.get('init_image'))
        
        filtered_params = filter_params(params=params, filters={'init_image', 'engine_id', 'text_prompts'})

        url = make_url(
            version=APIVersion.V1, 
            resource=resource, 
            endpoint=f"{params.get('engine_id')}/{Endpoint.IMAGE_TO_IMAGE}"
        )

        text_prompts = get_multi_part_text_prompts(params.get('text_prompts'))

        response = requests.post(
            url,
            files={
                "init_image": open(image_path.filepath(), "rb")
            },
            data={
                **filtered_params,
                **text_prompts
            },
            headers={
                **self.client.headers,
                'Accept': 'application/json'
            }
        )

        image_path.cleanup()
        
        if response.status_code == 200 \
            and isinstance(response.json().get('artifacts'), list):
            return process_articafts(
                artifacts=response.json().get('artifacts'),
                endpoint=Endpoint.IMAGE_TO_IMAGE
            )

        raise StabilityAIError(
            response.status_code,
            'Failed to run v1 generation image to image',
            response.json()
        )
  
    def image_to_image_upscale(
        self, 
        **params: Unpack[ImageToImageUpscaleOptions]
    ) -> StabilityAIContentResponse:
        image_path = ImagePath(params.get('image'))

        engine_id = EngineId.ESRGAN_V1_X2PLUS.value
        
        filtered_params = filter_params(params=params, filters={'image'})

        url = make_url(
            version=APIVersion.V1, 
            resource=resource, 
            endpoint=f"{engine_id}/{Endpoint.IMAGE_TO_IMAGE_UPSCALE}"
        )

        response = requests.post(
            url,
            files={
                "image": open(image_path.filepath(), "rb")
            },
            data={
                **filtered_params
            },
            headers={
                **self.client.headers,
                'Accept': 'application/json'
            }
        )

        image_path.cleanup()
        
        if response.status_code == 200 \
            and isinstance(response.json().get('artifacts'), list):
            return process_articafts(
                artifacts=response.json().get('artifacts'),
                endpoint=Endpoint.IMAGE_TO_IMAGE_UPSCALE
            )

        raise StabilityAIError(
            response.status_code,
            'Failed to run v1 generation image to image',
            response.json()
        )
  
    def image_to_image_masking(
        self, 
        **params: Unpack[ImageToImageMaskingOptions]
    ) -> StabilityAIContentResponse:
        image_path = ImagePath(params.get('init_image'))
        mask_path: Optional[ImagePath] = None

        if params.get('mask_image') is not None:
            mask_path = ImagePath(params.get('mask_image'))
        
        filtered_params = filter_params(params=params, filters={'init_image', 'mask_image', 'engine_id', 'text_prompts'})

        text_prompts = get_multi_part_text_prompts(params.get('text_prompts'))

        url = make_url(
            version=APIVersion.V1, 
            resource=resource, 
            endpoint=f"{params.get('engine_id')}/{Endpoint.IMAGE_TO_IMAGE_MASKING}"
        )

        files = {
            "init_image": open(image_path.filepath(), "rb")
        }

        if mask_path is not None:
            files['mask_image'] = open(mask_path.filepath(), "rb")

        response = requests.post(
            url,
            files=files,
            data={
                **filtered_params,
                **text_prompts
            },
            headers={
                **self.client.headers,
                'Accept': 'application/json'
            }
        )

        image_path.cleanup()
        if mask_path is not None:
            mask_path.cleanup()
        
        if response.status_code == 200 \
            and isinstance(response.json().get('artifacts'), list):
            return process_articafts(
                artifacts=response.json().get('artifacts'),
                endpoint=Endpoint.IMAGE_TO_IMAGE_MASKING
            )

        raise StabilityAIError(
            response.status_code,
            'Failed to run v1 generation image to image masking',
            response.json()
        )