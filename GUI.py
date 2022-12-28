import gradio as gr
from PIL import Image
import functions

#functions.load_txt2img_pipe()

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
                prompt = gr.Textbox(label="Prompt", show_label=False, max_lines=2,placeholder=f"Enter prompt").style(container=False)
                generate = gr.Button(value="Explore")

              neg_prompt = gr.Textbox(label="Negative prompt", placeholder="What to exclude from the image")
              gallery = gr.Gallery(label="Generated images", show_label=False).style(grid=[2], height="auto")
          
    inputs = [prompt, neg_prompt]
    outputs = [gallery]
    prompt.submit(functions.inference, inputs=inputs, outputs=outputs)
    generate.click(functions.inference, inputs=inputs, outputs=outputs)

sd_lite.queue()
sd_lite.launch(debug=True) #, share=True, height=768)