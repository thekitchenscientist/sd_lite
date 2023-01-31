import gradio as gr
from PIL import Image
import functions


CONFIG_HISTORY = functions.read_config_metadata()
def get_confighash_for_history(hash=None):
    # add error code if no history file
    if hash == None:
        return CONFIG_HISTORY[0][0]
#argument to return only prompts with a given hash
PROMPT_HISTORY = functions.read_prompt_metadata(config_hash = get_confighash_for_history())

sd_lite = gr.Blocks()

with sd_lite:
    gr.Markdown(
        f"""
          ## Stable Diffusion 2.1

          Press enter after typing in a text box to start the image generation process. The first time you trigger generation on the Sketch or Transform tab it will fail because the model is loading. Once its loaded simply click into a text box again and press enter.
        """)
    with gr.Tabs():
        with gr.TabItem("Explore"):
            with gr.Column():
                with gr.Row():
                    explore_prompt = gr.Textbox(label="Prompt", show_label=False, lines=1, placeholder=f"What do you want to see?")
                    explore_anti_prompt = gr.Textbox(label="Negative prompt", show_label=False, lines=1, placeholder="What should the image avoid including?")
                with gr.Row():
                    explore_alt_prompt = gr.Textbox(label="Alternative prompt", show_label=False, lines=1, placeholder=f"How do you want it styled?")
                    explore_alt_mode = gr.Dropdown(label="Mode", show_label=False,choices=["delay adding style until 15%", "delay adding style until 30%", "delay adding style until 45%","mirror up:down","mirror left:right","rotate 90","rotate 180"], value="delay adding style until 15%")
                explore_gallery = gr.Gallery(label="Generated images", show_label=False).style(grid=[3], height="auto")
            with gr.Accordion("The text to image method is using Safer Diffusion which aims to supress inappropriate content", open=False):
                gr.Markdown("After (n_steps*0.1)+1 to allow the composition to form, the model then will steer the image diffusion away from the categories of 'hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use, theft, vandalism, weapons, child abuse, brutality & cruelty.")
            gr.Markdown("The model should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes. **This model exacerbates biases to such a degree that viewer discretion must be advised irrespective of the input or its intent. The training data is lacking in knowledge of communities and cultures that use languages other than English and as a result the default outputs are from a white and western cultural perspective. Furthermore any representations of other languages or cultures is likely to be mischaracterised and biased.**")
            gr.Markdown("[The full terms of the Open RAIL++-M licence which governs the model use are available here](https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL)")
        with gr.TabItem("Sketch"):
            with gr.Column():
                with gr.Row():
                    sketch_prompt = gr.Textbox(label="Prompt", show_label=False, lines=1, placeholder=f"What do you want to see?")
                    sketch_anti_prompt = gr.Textbox(label="Negative prompt", show_label=False, lines=1, placeholder="What should the image avoid including?")  
                with gr.Row():  
                    with gr.Column(scale=1.2):
                        sketch_image_input = gr.Image(label="Image", show_label=False, type="pil", tool="color-sketch")
                    with gr.Column(scale=2.8):
                        sketch_gallery = gr.Gallery(label="Generated images", show_label=False).style(grid=[3], height="auto")
            gr.Markdown("The model licence prohibits alterations of copyrighted or licensed material for which you do not have the rights. Read the conditions of the [Open RAIL++-M](https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL) licence.")
        with gr.TabItem("Transform"):
            with gr.Column():
                with gr.Row():
                    transform_prompt = gr.Textbox(label="Prompt", show_label=False, lines=1, placeholder=f"What do you want to see?")
                    transform_anti_prompt = gr.Textbox(label="Negative prompt", show_label=False, lines=1, placeholder="What should the image avoid including?")  
                with gr.Row():  
                    with gr.Column(scale=1.2):
                        transform_image_input = gr.Image(label="Image", show_label=False, type="pil")#, tool="sketch")
                    with gr.Column(scale=2.8):
                        transform_gallery = gr.Gallery(label="Generated images", show_label=False).style(grid=[3], height="auto")
            gr.Markdown("The model licence prohibits alterations of copyrighted or licensed material for which you do not have the rights. Read the conditions of the [Open RAIL++-M](https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL) licence.")
        with gr.TabItem("Hybrid"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        hybrid_prompt = gr.Textbox(label="Prompt", show_label=False, lines=1, placeholder=f"What do you want to use for image A? (press enter to preview)")
                        hybrid_prompt_gallery = gr.Gallery(label="Generated images", show_label=False).style(grid=[3], height="auto")
                    with gr.Column():
                        hybrid_alt_prompt = gr.Textbox(label="Prompt", show_label=False, lines=1, placeholder=f"What do you want to use for image B? (press enter to preview)")
                        hybrid_alt_prompt_gallery = gr.Gallery(label="Generated images", show_label=False).style(grid=[3], height="auto")
                with gr.Row():
                    hybrid_anti_prompt = gr.Textbox(label="Negative prompt (applied to all images)", show_label=True, lines=1, placeholder="What should the images avoid including?")
                    hybrid_alt_mode = gr.Dropdown(label="Morph Mode (pick one to start image generation)",choices=["alternating", "increasing B", "decreasing B",  "switch A:B 25%","switch A:B 50%","switch A:B 75%", "weight A:B 75:25", "weight A:B 50:50", "weight A:B 25:75"])
                with gr.Column():
                    hybrid_gallery = gr.Gallery(label="Generated images", show_label=False).style(grid=[3], height="auto")
        with gr.TabItem("Morph"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        morph_prompt = gr.Textbox(label="Prompt", show_label=False, lines=1, placeholder=f"What do you want to use for image A? (press enter to preview)")
                        morph_prompt_gallery = gr.Gallery(label="Generated images", show_label=False).style(grid=[3], height="auto")
                    with gr.Column():
                        morph_alt_prompt = gr.Textbox(label="Prompt", show_label=False, lines=1, placeholder=f"What do you want to use for image B? (press enter to preview)")
                        morph_alt_prompt_gallery = gr.Gallery(label="Generated images", show_label=False).style(grid=[3], height="auto")
                with gr.Row():
                    morph_anti_prompt = gr.Textbox(label="Negative prompt (applied to all images)", show_label=True, lines=1, placeholder="What should the images avoid including?")
                    morph_alt_mode = gr.Dropdown(label="Morph Mode (pick one to start image generation)",choices=["alternating", "increasing B", "decreasing B",  "switch A:B 25%","switch A:B 50%","switch A:B 75%", "weight A:B 75:25", "weight A:B 50:50", "weight A:B 25:75"])
                with gr.Column():
                    morph_gallery = gr.Gallery(label="Generated images", show_label=False).style(grid=[3], height="auto")
        with gr.TabItem("History"):
            with gr.Column():
                with gr.Row():
                    history_config_choice = gr.Dropdown(label="Config",choices=[item[0] for item in CONFIG_HISTORY])
                    prompt_config_choice = gr.Dropdown(label="Image",choices=[item[0] for item in functions.read_prompt_metadata()])
                with gr.Accordion("Selected image:", open=False) as history_show_image:
                    with gr.Row():
                        with gr.Column():
                            history_chosen_image = gr.Image(label="Image", interactive=False, show_label=False, type="pil")
                        with gr.Column():
                            history_to_explore = gr.Button(value="Send Settings to Explore")
                            history_to_sketch = gr.Button(value="Send Image to Sketch (double click or image will be zoomed-in")
                            history_to_transform = gr.Button(value="Send Image to Transform")
                history_config_table = gr.DataFrame(CONFIG_HISTORY,  type="numpy", max_rows=5, overflow_row_behaviour="paginate", label="Config Settings", wrap=True, headers=["hash","MODEL","SCHEDULER","WIDTH","HEIGHT","SEED","IMAGE_COUNT","IMAGE_BRACKETING"])  
                history_prompt_table = gr.DataFrame(PROMPT_HISTORY,  type="numpy", max_rows=10, overflow_row_behaviour="paginate", label="Prompt Settings", wrap=True, headers=["UUID","prompt","anti_prompt","alt_prompt","mode","steps","scale","strength","seed"])

    explore_inputs = [explore_prompt, explore_anti_prompt,explore_alt_prompt,explore_alt_mode]
    explore_outputs = [explore_gallery]
    sketch_inputs = [sketch_prompt, sketch_anti_prompt, sketch_image_input]
    sketch_outputs = [sketch_gallery]
    transform_inputs = [transform_prompt, transform_anti_prompt, transform_image_input]
    transform_outputs = [transform_gallery]
    hybrid_prompt_inputs = [hybrid_prompt, hybrid_anti_prompt]
    hybrid_prompt_outputs = [hybrid_prompt_gallery]
    hybrid_alt_prompt_inputs = [hybrid_alt_prompt, hybrid_anti_prompt]
    hybrid_alt_prompt_outputs = [hybrid_alt_prompt_gallery]
    hybrid_inputs = [hybrid_prompt, hybrid_anti_prompt, hybrid_alt_prompt, hybrid_alt_mode]
    hybrid_outputs = [hybrid_gallery]
    morph_prompt_inputs = [morph_prompt, morph_anti_prompt]
    morph_prompt_outputs = [morph_prompt_gallery]
    morph_alt_prompt_inputs = [morph_alt_prompt, morph_anti_prompt]
    morph_alt_prompt_outputs = [morph_alt_prompt_gallery]
    morph_inputs = [morph_prompt, morph_anti_prompt, morph_alt_prompt, morph_alt_mode]
    morph_outputs = [morph_gallery]
    explore_prompt.submit(functions.txt2img_inference, inputs=explore_inputs, outputs=explore_outputs)
    explore_anti_prompt.submit(functions.txt2img_inference, inputs=explore_inputs, outputs=explore_outputs)
    explore_alt_prompt.submit(functions.txt2img_inference, inputs=explore_inputs, outputs=explore_outputs)
    explore_alt_mode.change(functions.txt2img_inference, inputs=explore_inputs, outputs=explore_outputs)
    sketch_prompt.submit(functions.img2img_inference, inputs=sketch_inputs, outputs=sketch_outputs)
    sketch_anti_prompt.submit(functions.img2img_inference, inputs=sketch_inputs, outputs=sketch_outputs)
    transform_prompt.submit(functions.depth2img_inference, inputs=transform_inputs, outputs=transform_outputs)
    transform_anti_prompt.submit(functions.depth2img_inference, inputs=transform_inputs, outputs=transform_outputs)
    hybrid_prompt.submit(functions.txt2img_inference, inputs=hybrid_prompt_inputs, outputs=hybrid_prompt_outputs)
    hybrid_alt_prompt.submit(functions.txt2img_inference, inputs=hybrid_alt_prompt_inputs, outputs=hybrid_alt_prompt_outputs)
    hybrid_alt_mode.change(functions.txt2img_inference, inputs=hybrid_inputs, outputs=hybrid_outputs)
    morph_prompt.submit(functions.txt2img_inference, inputs=morph_prompt_inputs, outputs=morph_prompt_outputs)
    morph_alt_prompt.submit(functions.txt2img_inference, inputs=morph_alt_prompt_inputs, outputs=morph_alt_prompt_outputs)
    morph_alt_mode.change(functions.txt2img_inference, inputs=morph_inputs, outputs=morph_outputs)
    history_config_choice.change(fn=lambda value: gr.update(value=functions.read_prompt_metadata(value)), inputs=history_config_choice, outputs=history_prompt_table)
    history_config_choice.change(fn=lambda value: gr.update(choices=[item[0] for item in functions.read_prompt_metadata(value)]), inputs=history_config_choice, outputs=prompt_config_choice)
    prompt_config_choice.change(fn=lambda value: gr.update(open=True), inputs=prompt_config_choice, outputs=history_show_image)
    prompt_config_choice.change(fn=lambda value: gr.update(value=functions.IMAGE_OUTPUT_FOLDER+'/'+value), inputs=prompt_config_choice, outputs=history_chosen_image)
    history_to_explore.click(fn=lambda value: gr.update(value=functions.get_prompt_metadata(value)[0][1]), inputs=prompt_config_choice, outputs=explore_prompt)
    history_to_explore.click(fn=lambda value: gr.update(value=functions.get_prompt_metadata(value)[0][2]), inputs=prompt_config_choice, outputs=explore_anti_prompt)
    history_to_sketch.click(fn=lambda value: gr.update(value=functions.IMAGE_OUTPUT_FOLDER+'/'+value), inputs=prompt_config_choice, outputs=sketch_image_input)
    history_to_sketch.click(fn=lambda value: gr.update(value=functions.get_prompt_metadata(value)[0][1]), inputs=prompt_config_choice, outputs=sketch_prompt)
    history_to_sketch.click(fn=lambda value: gr.update(value=functions.get_prompt_metadata(value)[0][2]), inputs=prompt_config_choice, outputs=sketch_anti_prompt)
    history_to_transform.click(fn=lambda value: gr.update(value=functions.IMAGE_OUTPUT_FOLDER+'/'+value), inputs=prompt_config_choice, outputs=transform_image_input)
    history_to_transform.click(fn=lambda value: gr.update(value=functions.get_prompt_metadata(value)[0][1]), inputs=prompt_config_choice, outputs=transform_prompt)
    history_to_transform.click(fn=lambda value: gr.update(value=functions.get_prompt_metadata(value)[0][2]), inputs=prompt_config_choice, outputs=transform_anti_prompt)

sd_lite.queue()
sd_lite.launch(debug=True, show_api=False)