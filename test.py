import stability_ai
from stability_ai.util import ( StabilityAIContentResponse )
from stability_ai.v1.generation import ( TextPrompt, EngineId )

value = stability_ai.v1.generation.text_to_image(
    engine_id=EngineId.STABLE_DIFFUSION_XL_BETA_V2_2_2,
    text_prompts=[
        TextPrompt(text="a big goat", weight=0.5)
    ]
)

print(value[0].filepath)