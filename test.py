import stability_ai
from stability_ai.util import ( StabilityAIContentResponse )
from stability_ai.v1.generation import ( TextPrompt, EngineId, ImageToImageMaskSource )

txt2img = stability_ai.v1.generation.text_to_image(
    engine_id=EngineId.STABLE_DIFFUSION_XL_BETA_V2_2_2,
    text_prompts=[
        TextPrompt(text="a big goat", weight=0.5)
    ]
)

print(txt2img[0].filepath)

img2img = stability_ai.v1.generation.image_to_image(
    engine_id=EngineId.STABLE_DIFFUSION_XL_BETA_V2_2_2,
    text_prompts=[
        TextPrompt(text="a big goat", weight=0.5)
    ],
    init_image="https://storage.googleapis.com/storage.catbird.ai/test-data/bird.png"
)

print(img2img[0].filepath)

img2img_upscale = stability_ai.v1.generation.image_to_image_upscale(
    image="https://storage.googleapis.com/storage.catbird.ai/test-data/bird.png",
    width=1024
)

print(img2img_upscale[0].filepath)

img2img_masking = stability_ai.v1.generation.image_to_image_masking(
    engine_id=EngineId.STABLE_DIFFUSION_XL_BETA_V2_2_2,
    text_prompts=[
        TextPrompt(text="a beautiful ocean", weight=0.5)
    ],
    init_image="https://storage.googleapis.com/storage.catbird.ai/test-data/bird.png",
    mask_source=ImageToImageMaskSource.INIT_IMAGE_ALPHA
)

print(img2img_masking[0].filepath)