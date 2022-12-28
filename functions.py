from diffusers import StableDiffusionPipeline, DiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import random
import config
import uuid
import sqlite3
import os

if not os.path.isdir(config.IMAGE_OUTPUT):
    os.mkdir(config.IMAGE_OUTPUT)

#def load_txt2img_pipe():

scheduler = DPMSolverMultistepScheduler.from_pretrained(config.MODEL_ID, subfolder="scheduler")

pipe = StableDiffusionPipeline.from_pretrained(
      config.MODEL_ID,
      revision="fp16" if torch.cuda.is_available() else "fp32",
      torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
      scheduler=scheduler
    ).to("cuda")

if config.SPLIT_ATTENTION:
    pipe.enable_attention_slicing()
if config.MEMORY_EFFICIENT_ATTENTION:
    pipe.enable_xformers_memory_efficient_attention()



def inference(prompt, neg_prompt=""):

    n_images = config.IMAGE_COUNT
    guidance = config.IMAGE_CFG
    steps = config.IMAGE_STEPS
    width= config.IMAGE_WIDTH
    height= config.IMAGE_HEIGHT
    seed= config.IMAGE_SEED

    if seed == 0:
        seed = random.randint(0, 2147483647)

    generator = torch.Generator('cuda').manual_seed(seed)
    prompt = prompt

    try:
        return txt_to_img(prompt, n_images, neg_prompt, guidance, steps, width, height, generator, seed)

    except:
        return None

def txt_to_img(prompt, n_images, neg_prompt, guidance, steps, width, height, generator, seed):

    output_name = str(uuid.uuid4())
    
    result = pipe(
      prompt,
      num_images_per_prompt = n_images,
      negative_prompt = neg_prompt,
      num_inference_steps = int(steps),
      guidance_scale = guidance,
      width = width,
      height = height,
      generator = generator).images

    try:
        for i, image in enumerate(result):
            #print(i, image)
            image.save( config.IMAGE_OUTPUT+'/'+output_name+'_'+str(i)+config.IMAGE_FORMAT, 'PNG')
    except:
        image.save( str(i)+'.png', 'PNG')

    return result