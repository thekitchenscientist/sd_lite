import gradio as gr
from PIL import Image
import functions


css = """.main-div div{display:inline-flex;align-items:center;gap:.8rem;font-size:1.75rem}.main-div div h1{font-weight:900;margin-bottom:7px}.main-div p{margin-bottom:10px;font-size:94%}a{text-decoration:underline}.tabs{margin-top:0;margin-bottom:0}#gallery{min-height:20rem}
"""
with gr.Blocks(css=css) as sd_lite:
    gr.HTML(
        f"""
          <div class="main-div">
            <div>
              <h1>Stable Diffusion 2.1</h1>
            </div><br>
          </div>
        """
    )
    with gr.Row():
        
        with gr.Column(scale=70):
          with gr.Group():
              with gr.Row():
                prompt = gr.Textbox(label="Prompt", show_label=False,placeholder=f"What do you want to see?")
                anti_prompt = gr.Textbox(label="Negative prompt", show_label=False, placeholder="What should the image avoid including?")
              #load_txt2img_model = gr.Button(value="Load Model")
              explore = gr.Button(value="Explore")
              gallery = gr.Gallery(label="Generated images", show_label=False).style(grid=[2], height="auto")
          
    inputs = [prompt, anti_prompt]
    outputs = [gallery]
    prompt.submit(functions.inference, inputs=inputs, outputs=outputs)
    explore.click(functions.inference, inputs=inputs, outputs=outputs)
    #load_txt2img_model.click(functions.load_txt2img_pipe)

sd_lite.queue()
sd_lite.launch(debug=True) #, share=True)