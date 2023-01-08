import gradio as gr
from PIL import Image
import functions


CONFIG_HISTORY = functions.read_config_metadata()
def get_confighash_for_history():
    return CONFIG_HISTORY[0][0]
#argument to return only prompts with a given hash
PROMPT_HISTORY = functions.read_prompt_metadata()

sd_lite = gr.Blocks()

with sd_lite:
    gr.Markdown(
        f"""
          ## Stable Diffusion 2.1

          Press enter after typing in either text box to start the image generation process. The first time you trigger generation on any tab it will fail because the model is loading. Once its loaded simply click into a text box again and press enter.
        """)
    with gr.Tabs():
        with gr.TabItem("Explore"):
            with gr.Column():
                with gr.Row():
                    explore_prompt = gr.Textbox(label="Prompt", show_label=False, lines=1, placeholder=f"What do you want to see?")
                    explore_anti_prompt = gr.Textbox(label="Negative prompt", show_label=False, lines=1, placeholder="What should the image avoid including?")
                explore_gallery = gr.Gallery(label="Generated images", show_label=False).style(grid=[3], height="auto")
            with gr.Accordion("The text to image method has been replaced with Safer Diffusion which aims to supress inappropriate content", open=False):
                gr.Markdown("in the categories of 'hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use, theft, vandalism, weapons, child abuse, brutality & cruelty")
            gr.Markdown("The model should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes. **This model exacerbates biases to such a degree that viewer discretion must be advised irrespective of the input or its intent. The training data is lacking in knowledge of communities and cultures that use languages other than English and as a result the default outputs are from a white and western cultural perspective. Furthermore any representations of other languages or cultures is likely to be mischaracterised and biased.**")
        with gr.TabItem("Sketch"):
            with gr.Column():
                with gr.Row():
                    sketch_prompt = gr.Textbox(label="Prompt", show_label=False, lines=1, placeholder=f"What do you want to see?")
                    sketch_anti_prompt = gr.Textbox(label="Negative prompt", show_label=False, lines=1, placeholder="What should the image avoid including?")  
                with gr.Row():  
                    with gr.Column(scale=1.2):
                        sketch_image_input = gr.Image(label="Image", show_label=False, type="pil")#, tool="sketch")
                    with gr.Column(scale=2.8):
                        sketch_gallery = gr.Gallery(label="Generated images", show_label=False).style(grid=[3], height="auto")
            gr.Markdown("The model licence prohibits alterations of copyrighted or licensed material for which you do not have the rights.")
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
            gr.Markdown("The model licence prohibits alterations of copyrighted or licensed material for which you do not have the rights.")
        with gr.TabItem("History"):
            with gr.Column():
                with gr.Row():
                    history_config_choice = gr.Dropdown(label="Config",choices=[item[0] for item in CONFIG_HISTORY])
                    prompt_config_choice = gr.Dropdown(label="Image",choices=[item[0] for item in PROMPT_HISTORY])
                with gr.Accordion("Selected image:", open=False):
                    with gr.Row():
                        with gr.Column():
                            history_gallery = gr.Gallery(label="Chosen image", show_label=False).style(grid=[1], height="auto")
                        with gr.Column():
                            history_to_explore = gr.Button(value="Send Settings to Explore")
                            history_to_explore = gr.Button(value="Send Image to Sketch")
                            history_to_explore = gr.Button(value="Send Image to Transform")
                history_config_table = gr.DataFrame(CONFIG_HISTORY,  type="numpy", max_rows=5, overflow_row_behaviour="paginate", label="Config Settings", wrap=True, headers=["hash","MODEL","SCHEDULER","WIDTH","HEIGHT","SEED","IMAGE_COUNT","IMAGE_BRACKETING"])  
                history_prompt_table = gr.DataFrame(PROMPT_HISTORY,  type="numpy", max_rows=10, overflow_row_behaviour="paginate", label="Prompt Settings", wrap=True, headers=["UUID","prompt","anti_prompt","steps","scale","strength","seed"])  
                
            gr.Markdown("The model licence prohibits alterations of copyrighted or licensed material for which you do not have the rights.")

    explore_inputs = [explore_prompt, explore_anti_prompt]
    explore_outputs = [explore_gallery]
    sketch_inputs = [sketch_prompt, sketch_anti_prompt, sketch_image_input]
    sketch_outputs = [sketch_gallery]
    transform_inputs = [transform_prompt, transform_anti_prompt, transform_image_input]
    transform_outputs = [transform_gallery]
    explore_prompt.submit(functions.txt2img_inference, inputs=explore_inputs, outputs=explore_outputs)
    explore_anti_prompt.submit(functions.txt2img_inference, inputs=explore_inputs, outputs=explore_outputs)
    sketch_prompt.submit(functions.img2img_inference, inputs=sketch_inputs, outputs=sketch_outputs)
    sketch_anti_prompt.submit(functions.img2img_inference, inputs=sketch_inputs, outputs=sketch_outputs)
    transform_prompt.submit(functions.depth2img_inference, inputs=transform_inputs, outputs=transform_outputs)
    transform_anti_prompt.submit(functions.depth2img_inference, inputs=transform_inputs, outputs=transform_outputs)
    # functions to take selected config and return associated prompts
    # functions to take selected image and load, if load then add onclick to pass to other tabs

sd_lite.queue()
sd_lite.launch(debug=True)