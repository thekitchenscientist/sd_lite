from PIL import Image
import functions
import config as config
from argparse import ArgumentParser
import os



def main(args):

	#load config if given
	if args.config != None:	
		if os.path.exists(functions.ROOT_DIR+"/prompts/"+args.config+".py"):
			from prompts import config as config
	
	#load prompt_list
	prompt_list = ["cheese and wine","chickens firing guns at polar bears"]
		
	if os.path.exists(args.prompts):
		prompt_list = open(args.prompts)
	elif os.path.exists(functions.ROOT_DIR+"/prompts/"+args.prompts):
		prompt_list = open(functions.ROOT_DIR+"/prompts/"+args.prompts)


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


#load supplied data
parser = ArgumentParser()
parser.add_argument("--prompts", type=str, default="prompt_list.txt", help="prompt file")
parser.add_argument("--config", type=str, default="config", help="alternative configuration file")
args = parser.parse_args()
main(args)

