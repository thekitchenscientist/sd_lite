from PIL import Image
import functions
import config as config
from argparse import ArgumentParser
import os
import importlib


def main(args):

	#load config if given
	if args.config is not None:	
		if os.path.exists(functions.ROOT_DIR+"/prompts/"+args.config+".py"):
			from prompts import config as config
			functions.SESSION_ID = functions.get_config_hash(functions.ROOT_DIR+"/prompts/"+args.config+".py")
			functions.CONFIG_LIST = (functions.SESSION_ID,config.SPLIT_ATTENTION,config.MEMORY_EFFICIENT_ATTENTION,config.HALF_PRECISION,config.MODEL_ID,config.IMAGE_INPUT_FOLDER,config.IMAGE_OUTPUT_FOLDER,config.IMAGE_FORMAT,config.IMAGE_SCHEDULER,
                   config.IMAGE_WIDTH,config.IMAGE_HEIGHT,config.IMAGE_SEED,config.IMAGE_SCALE,config.IMAGE_STEPS,config.IMAGE_SCALE_OFFSET,config.IMAGE_STEPS_OFFSET,config.IMAGE_COUNT,config.IMAGE_STRENGTH,config.IMAGE_STRENGTH_OFFSET,
                   config.IMAGE_BRACKETING,config.SAVE_METADATA_TO_IMAGE)
			functions.check_config_hash_exists(functions.SESSION_ID, functions.CONFIG_LIST)
	else:
		import config as config

	
	#load prompt_list if given
	prompt_list = ["cheese and wine","chickens firing guns","polar bear cubs on an iceberg"]
	
	if args.prompts is not None:		
		if os.path.exists(args.prompts):
			prompt_list = open(args.prompts)
		elif os.path.exists(functions.ROOT_DIR+"/prompts/"+args.prompts+".txt"):
			prompt_list = open(functions.ROOT_DIR+"/prompts/"+args.prompts+".txt")


	#parse prompt_list
	for prompt_parts in prompt_list:
		prompt = ""
		anti_prompt = ""
		prompt_text = prompt_parts.split("|")
		prompt = prompt_text[0]
		if len(prompt_text) > 1:
			anti_prompt = prompt_text[-1]

		print(prompt,anti_prompt)

		functions.txt2img_inference(explore_prompt=prompt, explore_anti_prompt=anti_prompt, n_images = config.IMAGE_COUNT, guidance = config.IMAGE_SCALE, steps = config.IMAGE_STEPS, width= config.IMAGE_WIDTH, height= config.IMAGE_HEIGHT, seed= config.IMAGE_SEED)


#parse supplied data
parser = ArgumentParser()
parser.add_argument("--mode", type=str, default="text", help="inference mode: text, image, depth")
parser.add_argument("--prompts", type=str, default=None, help="txt prompt file name (and URL if not .\prompts). One line per prompt, use | to add anti-prompts")
parser.add_argument("--config", type=str, default=None, help="(None|config) alternative configuration file (in .\prompts)")
args = parser.parse_args()
main(args)

