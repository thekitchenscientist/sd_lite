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

		functions.inference(prompt=prompt, anti_prompt=anti_prompt, n_images = config.IMAGE_COUNT, guidance = config.IMAGE_CFG, steps = config.IMAGE_STEPS, width= config.IMAGE_WIDTH, height= config.IMAGE_HEIGHT, seed= config.IMAGE_SEED)


#parse supplied data
parser = ArgumentParser()
parser.add_argument("--prompts", type=str, default=None, help="txt prompt file name (and URL if not .\prompts). One line per prompt, use | to add anti-prompts")
parser.add_argument("--config", type=str, default=None, help="(None|config) alternative configuration file (in .\prompts)")
args = parser.parse_args()
main(args)

