from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionDepth2ImgPipeline, DiffusionPipeline, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
import torch
from sld import SLDPipeline
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
def get_config_hash(filepath):

    md5 = hashlib.md5()
    with open(filepath,'rb') as config_file:
        while True:
            data = config_file.read(BUFFER_SIZE)
            if not data:
                break
            md5.update(data)

    return md5.hexdigest()

SESSION_ID = get_config_hash(ROOT_DIR+"/config.py")

CONFIG_LIST = (SESSION_ID,config.SPLIT_ATTENTION,config.MEMORY_EFFICIENT_ATTENTION,config.HALF_PRECISION,config.MODEL_ID,config.IMAGE_INPUT_FOLDER,config.IMAGE_OUTPUT_FOLDER,config.IMAGE_FORMAT,config.IMAGE_SCHEDULER,
                   config.IMAGE_WIDTH,config.IMAGE_HEIGHT,config.IMAGE_SEED,config.IMAGE_SCALE,config.IMAGE_STEPS,config.IMAGE_SCALE_OFFSET,config.IMAGE_STEPS_OFFSET,config.IMAGE_COUNT,config.IMAGE_STRENGTH,config.IMAGE_STRENGTH_OFFSET,
                   config.IMAGE_BRACKETING,config.SAVE_METADATA_TO_IMAGE)

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        #print(sqlite3.version)
    except Error as e:
        print(e)

    return conn

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
                                        IMAGE_SCALE real NOT NULL,
                                        IMAGE_STEPS integer NOT NULL,
                                        IMAGE_SCALE_OFFSET real NOT NULL,
                                        IMAGE_STEPS_OFFSET integer NOT NULL,
                                        IMAGE_COUNT integer NOT NULL,
                                        IMAGE_STRENGTH real NOT NULL,
                                        IMAGE_STRENGTH_OFFSET real NOT NULL,
                                        IMAGE_BRACKETING integer NOT NULL,
                                        SAVE_METADATA_TO_IMAGE integer NOT NULL
                                    ); """

    sql_create_prompt_table = """CREATE TABLE IF NOT EXISTS prompts (
                                    id integer PRIMARY KEY,
                                    config_hash text NOT NULL,
                                    UUID text NOT NULL,
                                    scheduler text NOT NULL,
                                    prompt text,
                                    anti_prompt text,
                                    steps  integer NOT NULL,
                                    scale real NOT NULL,
                                    strength real,
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


def add_config_hash(conn, config_list):
    """
    Create a entry in the config table
    :param conn:
    :param CONFIG_LIST:
    :return: row id
    """
    sql = ''' INSERT INTO config(hash, 
                                        SPLIT_ATTENTION,
                                        MEMORY_EFFICIENT_ATTENTION,
                                        HALF_PRECISION,
                                        MODEL_ID,
                                        IMAGE_INPUT_FOLDER,
                                        IMAGE_OUTPUT_FOLDER,
                                        IMAGE_FORMAT,
                                        IMAGE_SCHEDULER,
                                        IMAGE_WIDTH,
                                        IMAGE_HEIGHT,
                                        IMAGE_SEED,
                                        IMAGE_SCALE,
                                        IMAGE_STEPS,
                                        IMAGE_SCALE_OFFSET,
                                        IMAGE_STEPS_OFFSET,
                                        IMAGE_COUNT,
                                        IMAGE_STRENGTH,
                                        IMAGE_STRENGTH_OFFSET,
                                        IMAGE_BRACKETING,
                                        SAVE_METADATA_TO_IMAGE)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, config_list)
    conn.commit()
    return cur.lastrowid

def add_prompt_metadata(conn, output_name, file_metadata):
    """
    Create a entry in the config table
    :param conn:
    :param output_name:
    :param file_metadata:
    :return: None
    """
    time_stamp = 'today'
    for i in range(0,len(file_metadata)):
        prompt_metadata_list = (SESSION_ID, output_name+'_'+str(i)+'.'+config.IMAGE_FORMAT, file_metadata[i]["scheduler"],file_metadata[i]["prompt"],file_metadata[i]["negative_prompt"],file_metadata[i]["steps"],file_metadata[i]["scale"],file_metadata[i]["strength"],file_metadata[i]["seed"],file_metadata[i]["n_images"], time_stamp)
     
        sql = ''' INSERT INTO prompts(config_hash,
                                        UUID,
                                        scheduler,
                                        prompt,
                                        anti_prompt,
                                        steps,
                                        scale,
                                        strength,
                                        seed,
                                        n_images,
                                        date_time)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?) '''
        cur = conn.cursor()
        cur.execute(sql, prompt_metadata_list)
    conn.commit()

def read_config_metadata(db_file=ROOT_DIR+"/history.db"):
    """
    Read the entries in the config table
    :param conn:
    :return: table contents
    """
    conn = create_connection(db_file)
    cur = conn.cursor()
    cur.execute("SELECT hash, MODEL_ID,IMAGE_SCHEDULER,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_SEED,IMAGE_COUNT,IMAGE_BRACKETING FROM config")

    rows = cur.fetchall()
    
    return rows

def read_prompt_metadata(db_file=ROOT_DIR+"/history.db"):
    """
    Read the entries in the config table
    :param conn:
    :return: table contents
    """
    conn = create_connection(db_file)
    cur = conn.cursor()
    cur.execute("SELECT UUID, prompt, anti_prompt, steps, scale, strength, seed FROM prompts")

    rows = cur.fetchall()
    
    return rows

def lookup_config_hash(conn, config_hash):
    """
    Query tasks by priority
    :param conn: the Connection object
    :param priority:
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM config WHERE hash=?", (config_hash,))

    rows = cur.fetchall()

    return rows

#check if history.db exists if not create tables
if not os.path.exists(ROOT_DIR+"/history.db"):
    create_history_database()

#if it exists check if hash of config is already recorded, if not record it
def check_config_hash_exists(config_hash, config_list):
    if os.path.exists(ROOT_DIR+"/history.db"):
        conn = create_connection(ROOT_DIR+"/history.db")
        config_check = lookup_config_hash(conn, config_hash)

        if len(config_check) == 0:
            add_config_hash(conn,config_list)
            print("Added new config with hash",config_hash)

check_config_hash_exists(SESSION_ID, CONFIG_LIST)

if config.IMAGE_SCHEDULER == 'EulerAncestralDiscrete':
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(config.MODEL_ID, subfolder="scheduler")
else:
    scheduler = DPMSolverMultistepScheduler.from_pretrained(config.MODEL_ID, subfolder="scheduler") 

def set_mem_optimizations(pipe):
        if config.MEMORY_EFFICIENT_ATTENTION:
            pipe.enable_xformers_memory_efficient_attention()
        elif config.SPLIT_ATTENTION:
            pipe.enable_attention_slicing()


def load_txt2img_pipe(scheduler):

    if config.loaded_pipe != 'txt2img':

        print("Loading txt2img model into memory... This may take 5 minutes depending on available RAM.")

        pipe = SLDPipeline.from_pretrained(
            pretrained_model_name_or_path = config.MODEL_ID,
            revision="fp16" if config.HALF_PRECISION else "fp32",
            torch_dtype=torch.float16 if config.HALF_PRECISION else torch.float32,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None
            ).to("cuda")

        config.loaded_pipe = 'txt2img'
        print("txt2img model loaded")        
        set_mem_optimizations(pipe)
        pipe.to("cuda")
        return pipe

            


def load_img2img_pipe(loaded_pipe):

    if config.loaded_pipe != 'img2img':
        
        print("Loading img2img model into memory... This may take 5 minutes depending on available RAM.")
        
        config.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
              config.MODEL_ID,
              revision="fp16" if config.HALF_PRECISION else "fp32",
              torch_dtype=torch.float16 if config.HALF_PRECISION else torch.float32,
              scheduler=scheduler,
              safety_checker=None,
              feature_extractor=None
            ).to("cuda")

        config.loaded_pipe = 'img2img'
        print("imgimg model loaded")        
        set_mem_optimizations(pipe)
        pipe.to("cuda")
        return pipe


def load_depth2img_pipe(loaded_pipe):

    if config.loaded_pipe != 'depth2img':
        
        print("Loading depth2img model into memory... This may take 5 minutes depending on available RAM.")
        
        config.depth2img_pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
              config.DEPTH_MODEL_ID,
              revision="fp16" if config.HALF_PRECISION else "fp32",
              torch_dtype=torch.float16 if config.HALF_PRECISION else torch.float32,
              scheduler=scheduler
            ).to("cuda")

        config.loaded_pipe = 'depth2img'
        print("depth2img model loaded")        
        set_mem_optimizations(pipe)
        pipe.to("cuda")
        return pipe

def txt2img_inference(explore_prompt="", explore_anti_prompt="", n_images = config.IMAGE_COUNT, guidance = config.IMAGE_SCALE, steps = config.IMAGE_STEPS, width= config.IMAGE_WIDTH, height= config.IMAGE_HEIGHT, seed= config.IMAGE_SEED, strength=config.IMAGE_STRENGTH):
    if seed == 0:
        seed = random.randint(0, 2147483647)

    generator = torch.Generator('cuda').manual_seed(seed)

    if config.loaded_pipe == 'img2img':
        config.img2img_pipe.to("cpu")
    if config.loaded_pipe == 'depth2img':
        config.depth2img_pipe.to("cpu")
    if config.txt2img_pipe is not None:
        config.txt2img_pipe.to("cuda")

    try:
        prompt = explore_prompt
        anti_prompt = explore_anti_prompt
        return text_to_image(prompt, anti_prompt, n_images,  guidance, steps, width, height, generator, seed)

    except:
        return None


def img2img_inference(sketch_prompt="", sketch_anti_prompt="", sketch_image_input=None, n_images = config.IMAGE_COUNT, guidance = config.IMAGE_SCALE, steps = config.IMAGE_STEPS, width= config.IMAGE_WIDTH, height= config.IMAGE_HEIGHT, seed= config.IMAGE_SEED, strength=config.IMAGE_STRENGTH):
    if seed == 0:
        seed = random.randint(0, 2147483647)

    generator = torch.Generator('cuda').manual_seed(seed)

    if sketch_image_input is None:
        print('Error, no input image provided')
        return None

    if config.loaded_pipe == 'txt2img':
        config.txt2img_pipe.to("cpu")
    if config.loaded_pipe == 'depth2img':
        config.depth2img_pipe.to("cpu")
    if config.img2img_pipe is not None:
        config.img2img_pipe.to("cuda")

    try:
        prompt = sketch_prompt
        anti_prompt = sketch_anti_prompt
        input_image = sketch_image_input
        return image_to_image(prompt, anti_prompt, input_image, strength, n_images, guidance, steps, width, height, generator, seed)

    except:
        return None

def depth2img_inference(transform_prompt="", transform_anti_prompt="", transform_image_input=None, n_images = config.IMAGE_COUNT, guidance = config.IMAGE_SCALE, steps = config.IMAGE_STEPS, width= config.IMAGE_WIDTH, height= config.IMAGE_HEIGHT, seed= config.IMAGE_SEED, strength=config.IMAGE_STRENGTH):
    if seed == 0:
        seed = random.randint(0, 2147483647)

    generator = torch.Generator('cuda').manual_seed(seed)

    if transform_image_input is None:
        print('Error, no input image provided')
        return None

    if config.loaded_pipe == 'txt2img':
        config.txt2img_pipe.to("cpu")
    if config.loaded_pipe == 'img2img':
        config.img2img_pipe.to("cpu")
    if config.depth2img_pipe is not None:
        config.depth2img_pipe.to("cuda")

    try:
        prompt = transform_prompt
        anti_prompt = transform_anti_prompt
        input_image = transform_image_input
        return depth_to_image(prompt, anti_prompt, input_image, strength, n_images, guidance, steps, width, height, generator, seed)

    except:
        return None

def save_images(output_name, result, file_metadata):
    # save images to database
    try:
        conn = create_connection(ROOT_DIR+"/history.db")
        add_prompt_metadata(conn, output_name, file_metadata)
    except:
        print( "Error saving image metadata to history.db")

    # save image to folder
    try:
        for i, image in enumerate(result):
            if config.SAVE_METADATA_TO_IMAGE and config.IMAGE_FORMAT == "PNG":
                metadata = PngImagePlugin.PngInfo()
                for key, value in file_metadata[i].items():
                    if isinstance(key, str) and isinstance(value, str):
                        metadata.add_text(key, value)
                        #print(key, value)

            image.save( IMAGE_OUTPUT_FOLDER+'/'+output_name+'_'+str(i)+'.'+config.IMAGE_FORMAT, config.IMAGE_FORMAT, pnginfo=(metadata if config.SAVE_METADATA_TO_IMAGE and config.IMAGE_FORMAT == "PNG" else None))
    except:
        print( "Error saving image " +output_name+'_'+str(i)+'.png')


def text_to_image(prompt, anti_prompt, n_images,  guidance, steps, width, height, generator, seed):

    output_name = str(uuid.uuid4())
    result = []
    file_metadata = []

    if config.txt2img_pipe is None:
        config.txt2img_pipe = load_txt2img_pipe(scheduler)
    
    for settings in range(-1,2):

        if config.IMAGE_BRACKETING == False and settings != 0:
            continue

        # setup image properties
        temp_guidance = guidance + config.IMAGE_SCALE_OFFSET * settings
        temp_steps = steps + config.IMAGE_STEPS_OFFSET * settings
        temp_warm_up = int((temp_steps/10)+1)
        
        if settings == -1:
            temp_prompt = output_name + ". " + prompt
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
            generator = generator,
            sld_warmup_steps=temp_warm_up, #7,
            sld_guidance_scale= 5000,
            sld_threshold=0.025,
            sld_momentum_scale=0.5,
            sld_mom_beta=0.7).images

        result.extend(temp_result)
        
        # record image metadata
        temp_dict = {}
        temp_dict["scheduler"] = config.IMAGE_SCHEDULER
        temp_dict["prompt"] = temp_prompt
        temp_dict["negative_prompt"] = anti_prompt
        temp_dict["steps"] = str(temp_steps)
        temp_dict["scale"] = str(temp_guidance)
        temp_dict["strength"] = str(-1)
        temp_dict["seed"] = str(seed)
        temp_dict["n_images"] = str(n_images)

        file_metadata.append(copy.deepcopy(temp_dict))

        if n_images > 1:
            for k in range(1,n_images):
                file_metadata.append(copy.deepcopy(temp_dict))
                file_metadata[-1]["seed"] = str(int(file_metadata[-1]["seed"])+k)
    
    save_images(output_name, result, file_metadata)

        
    return result

def image_to_image(prompt, anti_prompt, input_image, strength, n_images, guidance, steps, width, height, generator, seed):

    output_name = str(uuid.uuid4())
    result = []
    file_metadata = []

    if config.img2img_pipe is None:
        config.img2img_pipe = load_img2img_pipe(scheduler)

    temp_image = input_image #['image']
    ratio = min(height / temp_image.height, width / temp_image.width)
    temp_image = temp_image.resize((int(temp_image.width * ratio), int(temp_image.height * ratio)), Image.LANCZOS)
    
    for settings in range(-1,2):

        if config.IMAGE_BRACKETING == False and settings != 0:
            continue

        # setup image properties
        temp_guidance = guidance + config.IMAGE_SCALE_OFFSET * settings
        temp_steps = steps + config.IMAGE_STEPS_OFFSET * settings
        temp_strength = strength + config.IMAGE_STRENGTH_OFFSET * settings
        
        if settings == -1:
            temp_prompt = output_name + ". " + prompt
        else:
            temp_prompt = prompt

        temp_result = config.img2img_pipe(
          temp_prompt,
          num_images_per_prompt = n_images,
          negative_prompt = anti_prompt,
          image = temp_image,
          num_inference_steps = int(temp_steps),
          strength = temp_strength,
          guidance_scale = temp_guidance,
          #width = width,
          #height = height,
          generator = generator).images

        result.extend(temp_result)
        
        # record image metadata
        temp_dict = {}
        temp_dict["scheduler"] = config.IMAGE_SCHEDULER
        temp_dict["prompt"] = temp_prompt
        temp_dict["negative_prompt"] = anti_prompt
        temp_dict["steps"] = str(temp_steps)
        temp_dict["scale"] = str(temp_guidance)
        temp_dict["strength"] = str(temp_strength)
        temp_dict["seed"] = str(seed)
        temp_dict["n_images"] = str(n_images)

        file_metadata.append(copy.deepcopy(temp_dict))

        if n_images > 1:
            for k in range(1,n_images):
                file_metadata.append(copy.deepcopy(temp_dict))
                file_metadata[-1]["seed"] = str(int(file_metadata[-1]["seed"])+k)

    try:
        temp_image.save( IMAGE_INPUT_FOLDER+'/'+output_name+'_input.'+config.IMAGE_FORMAT, config.IMAGE_FORMAT)
    except:
        print("Unable to save input image")

    save_images(output_name, result, file_metadata)
        
    return result

def depth_to_image(prompt, anti_prompt, input_image, strength, n_images, guidance, steps, width, height, generator, seed):

    output_name = str(uuid.uuid4())
    result = []
    file_metadata = []

    if config.depth2img_pipe is None:
        config.depth2img_pipe = load_depth2img_pipe(scheduler)

    temp_image = input_image #['image']
    ratio = min(height / temp_image.height, width / temp_image.width)
    temp_image = temp_image.resize((int(temp_image.width * ratio), int(temp_image.height * ratio)), Image.LANCZOS)
    
    for settings in range(-1,2):

        if config.IMAGE_BRACKETING == False and settings != 0:
            continue

        # setup image properties
        temp_guidance = guidance + config.IMAGE_SCALE_OFFSET * settings
        temp_steps = steps + config.IMAGE_STEPS_OFFSET * settings
        temp_strength = strength + config.IMAGE_STRENGTH_OFFSET * settings
        
        if settings == -1:
            temp_prompt = output_name + ". " + prompt
        else:
            temp_prompt = prompt

        temp_result = config.depth2img_pipe(
          temp_prompt,
          num_images_per_prompt = n_images,
          negative_prompt = anti_prompt,
          image = temp_image,
          num_inference_steps = int(temp_steps),
          strength = temp_strength,
          guidance_scale = temp_guidance,
          #width = width,
          #height = height,
          generator = generator).images

        result.extend(temp_result)
        
        # record image metadata
        temp_dict = {}
        temp_dict["scheduler"] = config.IMAGE_SCHEDULER
        temp_dict["prompt"] = temp_prompt
        temp_dict["negative_prompt"] = anti_prompt
        temp_dict["steps"] = str(temp_steps)
        temp_dict["scale"] = str(temp_guidance)
        temp_dict["strength"] = str(temp_strength)
        temp_dict["seed"] = str(seed)
        temp_dict["n_images"] = str(n_images)

        file_metadata.append(copy.deepcopy(temp_dict))

        if n_images > 1:
            for k in range(1,n_images):
                file_metadata.append(copy.deepcopy(temp_dict))
                file_metadata[-1]["seed"] = str(int(file_metadata[-1]["seed"])+k)

    try:
        temp_image.save( IMAGE_INPUT_FOLDER+'/'+output_name+'_input.'+config.IMAGE_FORMAT, config.IMAGE_FORMAT)
    except:
        print("Unable to save input image")

    save_images(output_name, result, file_metadata)
        
    return result