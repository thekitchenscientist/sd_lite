SPLIT_ATTENTION= True
MEMORY_EFFICIENT_ATTENTION = True
HALF_PRECISION = True
MODEL_ID = 'stabilityai/stable-diffusion-2-1-base'
DEPTH_MODEL_ID = 'stabilityai/stable-diffusion-2-depth'
IMAGE_INPUT_FOLDER = None
IMAGE_OUTPUT_FOLDER = None
IMAGE_FORMAT = 'PNG'
IMAGE_SCHEDULER = 'EulerAncestralDiscrete'
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
IMAGE_SEED = 1000000007
IMAGE_SCALE = 7.5
IMAGE_STEPS = 20
IMAGE_SCALE_OFFSET = 3
IMAGE_STEPS_OFFSET = 5
IMAGE_COUNT = 1
IMAGE_STRENGTH = 0.85
IMAGE_STRENGTH_OFFSET = 0.15
IMAGE_BRACKETING = True
SAVE_METADATA_TO_IMAGE = True
loaded_pipe = None
txt2img_pipe = None
img2img_pipe = None
depth2img_pipe = None
walk_pipe = None
