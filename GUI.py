import gradio as gr
from PIL import Image
import functions




css = """.main-div div{display:inline-flex;align-items:center;gap:.8rem;font-size:1.75rem}.main-div div h1{font-weight:900;margin-bottom:7px}.main-div p{margin-bottom:10px;font-size:94%}a{text-decoration:underline}.tabs{margin-top:0;margin-bottom:0}#gallery{min-height:20rem}
"""

sd_lite = gr.Blocks(css=css)

with sd_lite:
    gr.HTML(
        f"""
          <div class="main-div">
            <div>
              <h1>Stable Diffusion 2.1</h1>
            </div><br>
<p>Press enter after typing in either text box to start the image generation process. The first generation on any tab will fail whilst the model is loading. Once it is loaded simply click into a text box again and press enter.</p>
          </div>
        """
    )
    with gr.Tabs():
        with gr.TabItem("Explore"):
            with gr.Column():
                with gr.Row():
                    explore_prompt = gr.Textbox(label="Prompt", show_label=False, lines=1, placeholder=f"What do you want to see?")
                    explore_anti_prompt = gr.Textbox(label="Negative prompt", show_label=False, lines=1, placeholder="What should the image avoid including?")
                explore_gallery = gr.Gallery(label="Generated images", show_label=False).style(grid=[3], height="auto")
        with gr.TabItem("Sketch"):
            with gr.Column():
                with gr.Row():
                    sketch_prompt = gr.Textbox(label="Prompt", show_label=False, lines=1, placeholder=f"What do you want to see?")
                    sketch_anti_prompt = gr.Textbox(label="Negative prompt", show_label=False, lines=1, placeholder="What should the image avoid including?")  
                with gr.Row():  
                    with gr.Column():
                        sketch_image_input = gr.Image(label="Image", show_label=False, type="pil", tool="sketch")
                        #sketch_image_button = gr.Button("Generate Image")
                    sketch_gallery = gr.Gallery(label="Generated images", show_label=False).style(grid=[3], height="auto")
            gr.HTML(
                    f"""
                        <div>
            <p>The model licence requires you own the copyright to any images you input, in order for you to also have copyright over the output.</p>
                        </div>
                    """
                )
        with gr.TabItem("Transform"):
            with gr.Column():
                with gr.Row():
                    transform_prompt = gr.Textbox(label="Prompt", show_label=False, lines=1, placeholder=f"What do you want to see?")
                    transform_anti_prompt = gr.Textbox(label="Negative prompt", show_label=False, lines=1, placeholder="What should the image avoid including?")  
                with gr.Row():  
                    with gr.Column():
                        transform_image_input = gr.Image(label="Image", show_label=False, type="pil", tool="sketch")
                    transform_gallery = gr.Gallery(label="Generated images", show_label=False).style(grid=[3], height="auto")
    explore_inputs = [explore_prompt, explore_anti_prompt]
    explore_outputs = [explore_gallery]
    sketch_inputs = [sketch_prompt, sketch_anti_prompt, sketch_image_input]
    sketch_outputs = [sketch_gallery]
    transform_inputs = [transform_prompt, transform_anti_prompt, transform_image_input]
    transform_outputs = [transform_gallery]
    explore_prompt.submit(functions.txt2img_inference, inputs=explore_inputs, outputs=explore_outputs)
    explore_anti_prompt.submit(functions.txt2img_inference, inputs=explore_inputs, outputs=explore_outputs)
    #sketch_image_button.click(functions.img2img_inference, inputs=sketch_inputs, outputs=sketch_outputs)
    sketch_prompt.submit(functions.img2img_inference, inputs=sketch_inputs, outputs=sketch_outputs)
    sketch_anti_prompt.submit(functions.img2img_inference, inputs=sketch_inputs, outputs=sketch_outputs)
    transform_prompt.submit(functions.depth2img_inference, inputs=transform_inputs, outputs=transform_outputs)
    transform_anti_prompt.submit(functions.depth2img_inference, inputs=transform_inputs, outputs=transform_outputs)

sd_lite.queue()
sd_lite.launch(debug=True)