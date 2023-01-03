from diffusers import StableDiffusionPipeline, DiffusionPipeline, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
import torch
from PIL import PngImagePlugin, Image
import random
import config
import copy
import os
import uuid
import hashlib
import sqlite3
from sqlite3 import Error


loaded_pipe = config.loaded_pipe

BUFFER_SIZE = 65536 
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
def check_config_hash(filepath):

    md5 = hashlib.md5()
    with open(filepath,'rb') as config_file:
        while True:
            data = config_file.read(BUFFER_SIZE)
            if not data:
                break
            md5.update(data)

    return md5.hexdigest()


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)

    return conn

#print(create_connection(ROOT_DIR+"/history.db"))

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def create_history_database():
    database = ROOT_DIR+"/history.db"

    sql_create_config_table = """ CREATE TABLE IF NOT EXISTS config (
                                        hash text PRIMARY KEY,
                                        SPLIT_ATTENTION integer NOT NULL,
                                        MEMORY_EFFICIENT_ATTENTION  integer NOT NULL,
                                        HALF_PRECISION integer NOT NULL,
                                        MODEL_ID text NOT NULL,
                                        IMAGE_INPUT_FOLDER text,
                                        IMAGE_OUTPUT_FOLDER text,
                                        IMAGE_FORMAT text NOT NULL,
                                        IMAGE_SCHEDULER text NOT NULL,
                                        IMAGE_WIDTH integer NOT NULL,
                                        IMAGE_HEIGHT integer NOT NULL,
                                        IMAGE_SEED integer NOT NULL,
                                        IMAGE_CFG real NOT NULL,
                                        IMAGE_STEPS integer NOT NULL,
                                        IMAGE_CFG_OFFSET real NOT NULL,
                                        IMAGE_STEPS_OFFSET integer NOT NULL,
                                        IMAGE_COUNT integer NOT NULL,
                                        IMAGE_STRENGTH real NOT NULL,
                                        IMAGE_STRENGTH_OFFSET real NOT NULL,
                                        IMAGE_BRACKETING integer NOT NULL,
                                        SAVE_METADATA_TO_IMAGE integer NOT NULL,
                                        app_status text,
                                        loaded_pipe text,
                                        txt2img_pipe text
                                    ); """

    sql_create_prompt_table = """CREATE TABLE IF NOT EXISTS prompts (
                                    id integer PRIMARY KEY,
                                    UUID text NOT NULL,
                                    scheduler text NOT NULL,
                                    prompt text,
                                    anti_prompt text,
                                    steps  integer NOT NULL,
                                    CFG real NOT NULL,
                                    seed integer NOT NULL,
                                    n_images integer NOT NULL,
                                    date_time text NOT NULL,
                                    FOREIGN KEY (config_hash) REFERENCES config (hash)
                                );"""

    # create a database connection
    conn = create_connection(database)

    # create tables
    if conn is not None:
        # create projects table
        create_table(conn, sql_create_config_table)
        # create tasks table
        create_table(conn, sql_create_prompt_table)
    else:
        print("Error! cannot create the database connection.")

#check if history.db exists if not dreate tables
#check_config_hash(ROOT_DIR+"/config.py")
#if it exists check if hash of config is alreay recorded, if not record it

def load_txt2img_pipe(loaded_pipe):

    if loaded_pipe != 'txt2img':

        if config.IMAGE_SCHEDULER == 'EulerAncestralDiscrete':
            scheduler = EulerAncestralDiscreteScheduler.from_pretrained(config.MODEL_ID, subfolder="scheduler")
        else:
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

        if config.IMAGE_BRACKETING == False and settings != 0:
            continue

        # setup image properties
        temp_guidance = guidance + config.IMAGE_CFG_OFFSET * settings
        temp_steps = steps + config.IMAGE_STEPS_OFFSET * settings
        
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
        temp_dict["scheduler"] = config.IMAGE_SCHEDULER
        temp_dict["prompt"] = temp_prompt
        temp_dict["anti_prompt"] = anti_prompt
        temp_dict["steps"] = str(temp_steps)
        temp_dict["CFG"] = str(temp_guidance)
        temp_dict["seed"] = str(seed)
        temp_dict["n_images"] = str(n_images)

        file_metadata.append(copy.deepcopy(temp_dict))

        if n_images > 1:
            for k in range(1,n_images):
                file_metadata.append(copy.deepcopy(temp_dict))
                file_metadata[k+1]["seed"] = str(seed+k)

    # save image to folder
    try:
        for i, image in enumerate(result):
            if config.SAVE_METADATA_TO_IMAGE and config.IMAGE_FORMAT == "PNG":
                metadata = PngImagePlugin.PngInfo()
                for key, value in file_metadata[i].items():
                    if isinstance(key, str) and isinstance(value, str):
                        metadata.add_text(key, value)
                        print(key, value)

            image.save( IMAGE_OUTPUT_FOLDER+'/'+output_name+'_'+str(i)+'.'+config.IMAGE_FORMAT, config.IMAGE_FORMAT, pnginfo=(metadata if config.SAVE_METADATA_TO_IMAGE and config.IMAGE_FORMAT == "PNG" else None))
    except:
        print( "Error saving image " +output_name+'_'+str(i)+'.png')
        
 
    return result