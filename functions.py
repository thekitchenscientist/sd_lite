from diffusers import StableDiffusionPipeline, DiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import PngImagePlugin, Image
import random
import config
import uuid
import sqlite3
import os


loaded_pipe = config.loaded_pipe

# ensure correct folders exist and are used
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__)))

if config.IMAGE_INPUT_FOLDER is not None:
    if not os.path.isdir(config.IMAGE_INPUT_FOLDER):
        os.mkdir(config.IMAGE_INPUT_FOLDER)
    IMAGE_INPUT_FOLDER = config.IMAGE_INPUT_FOLDER
else:
    if not os.path.isdir(ROOT_DIR+"/inputs"):
        os.mkdir(ROOT_DIR+"/inputs")
    IMAGE_INPUT_FOLDER = ROOT_DIR+"/inputs"

if config.IMAGE_OUTPUT_FOLDER is not None:
    if not os.path.isdir(config.IMAGE_OUTPUT_FOLDER):
        os.mkdir(config.IMAGE_OUTPUT_FOLDER)
    IMAGE_OUTPUT_FOLDER = config.IMAGE_OUTPUT_FOLDER
else:
    if not os.path.isdir(ROOT_DIR+"/outputs"):
        os.mkdir(ROOT_DIR+"/outputs")
    IMAGE_OUTPUT_FOLDER = ROOT_DIR+"/outputs"

if not os.path.isdir(ROOT_DIR+"/prompts"):
    os.mkdir(ROOT_DIR+"/prompts")


# assign unique ID's to each configuration chosen
def generate_session_UUID():
    config.session_UUID = str(uuid.uuid4())
    #save the config data to history.db
    
if config.session_UUID == None:
    generate_session_UUID()



def load_txt2img_pipe(loaded_pipe):

    if loaded_pipe != 'txt2img':

        scheduler = DPMSolverMultistepScheduler.from_pretrained(config.MODEL_ID, subfolder="scheduler")
        
        print("Loading txt2img model into memory... This may take 5 minutes depending on available RAM.")
        
        config.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
              config.MODEL_ID,
              revision="fp16" if config.HALF_PRECISION else "fp32",
              torch_dtype=torch.float16 if config.HALF_PRECISION else torch.float32,
              scheduler=scheduler
            ).to("cuda")

        loaded_pipe = 'txt2img'
        

        if config.SPLIT_ATTENTION:
            config.txt2img_pipe.enable_attention_slicing()
        if config.MEMORY_EFFICIENT_ATTENTION:
            config.txt2img_pipe.enable_xformers_memory_efficient_attention()
            
        print("txt2img model loaded")

if loaded_pipe == None:
    load_txt2img_pipe(loaded_pipe)

def inference(prompt="", anti_prompt="", n_images = config.IMAGE_COUNT, guidance = config.IMAGE_CFG, steps = config.IMAGE_STEPS, width= config.IMAGE_WIDTH, height= config.IMAGE_HEIGHT, seed= config.IMAGE_SEED):
    if seed == 0:
        seed = random.randint(0, 2147483647)

    generator = torch.Generator('cuda').manual_seed(seed)
    prompt = prompt

    try:
        return text_to_image(prompt, anti_prompt, n_images,  guidance, steps, width, height, generator, seed)

    except:
        return None

def text_to_image(prompt, anti_prompt, n_images,  guidance, steps, width, height, generator, seed):

    output_name = str(uuid.uuid4())
    result = []
    file_metadata = []
    
    for settings in range(-1,2):

        # setup image properties
        temp_steps = steps + 1 * settings * config.IMAGE_STEPS_OFFSET
        temp_guidance = guidance - 1 * settings * config.IMAGE_CFG_OFFSET
        
        if settings == -1:
            temp_prompt = prompt + " " + output_name
        else:
            temp_prompt = prompt

        temp_result = config.txt2img_pipe(
          temp_prompt,
          num_images_per_prompt = n_images,
          negative_prompt = anti_prompt,
          num_inference_steps = int(temp_steps),
          guidance_scale = temp_guidance,
          width = width,
          height = height,
          generator = generator).images

        result.extend(temp_result)
        
        # record image metadata
        temp_dict = {}
        temp_dict["prompt"] = temp_prompt
        temp_dict["anti_prompt"] = anti_prompt
        temp_dict["steps"] = str(temp_steps)
        temp_dict["CFG"] = str(temp_guidance)
        temp_dict["seed"] = str(seed)
        temp_dict["n_images"] = str(n_images)

        file_metadata.append(temp_dict)

    # save image to folder
    try:
        for i, image in enumerate(result):
            if config.SAVE_METADATA_TO_IMAGE and config.IMAGE_FORMAT == "PNG":
                metadata = PngImagePlugin.PngInfo()
                for key, value in file_metadata[i].items():
                    if isinstance(key, str) and isinstance(value, str):
                        metadata.add_text(key, value)

            image.save( IMAGE_OUTPUT_FOLDER+'/'+output_name+'_'+str(i)+'.'+config.IMAGE_FORMAT, config.IMAGE_FORMAT, pnginfo=(metadata if config.SAVE_METADATA_TO_IMAGE and config.IMAGE_FORMAT == "PNG" else None))
    except:
        print( "Error saving image" +output_name+'_'+str(i)+'.png')
        
 
    return result