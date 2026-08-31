"""The validated LTX 2.3 stack: one configuration, every pose, every character.

Measured across all 24 recipes, every field here had exactly ONE value. It lives once
rather than being copied per recipe, because storing a global 24 times is how it quietly
stops being global -- one row gets edited, nothing complains, and two recipes that should
be identical are not.

NOT app/seeds.py, which is about NOISE seeds. Different sense of the word entirely.
"""

# content_lora is "none" on purpose: dropping DR34ML4Y is what removed the motion horror.
# It was a third LoRA competing for the same layers as the character LoRA, and the
# checkpoint already carries the NSFW training it was providing.
LTX_STACK = {
    'checkpoint': 'sulphur_dev_bf16',
    'content_lora': 'none',
    'distill': 'sulphur_distill_lora_condsafe',
    'distill_stage_1': 0.3,
    'distill_stage_2': 0.6,
    'frames': 241,
    'frame_rate': 24,
    'steps_stage_1': 20,
    'sigmas_stage_2': '0.85, 0.7250, 0.4219, 0.0',
    'cfg': 3,
    'stg': 1,
    'rescale': 0.9,
    'stg_blocks': '28',
    'negative': 'static, still image, frozen, no motion, slideshow, identity change, different person, face distortion, warped anatomy, extra limbs, deformed hands, merged limbs, mangled body, blurry, low quality',
}
