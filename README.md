**Latent Exploration**

Have you considered the vastness of latent space in large diffusion models? openCLIP which encodes the text in Stable Diffusion 2.1 has a vocabulary of 49409 tokens and knowledge of 2 billion concepts. Each basic 75 token prompt can be paired with one of 2.1 billion seeds. Generating 100 images per second it would still take 8 months try each seed once.

Having analysed over 2 million stable diffusion prompts, it is clear humans are rubbish at exploring latent space. Despite the nearly limitless possibilities we stick with the familiar and once we have a formula that works we stop experimenting. Can this knowledge of what people typically look for be reversed to push into the unknown? How can the image generation interface be designed to encourage exploration, rather than making more of the same?

The guiding principal of this project is to be able to run Stable Diffusion with minimalism. This means:

* taking steps to minimise harmful outputs and combatting the model bias
* as few options as possible (abstract away CFG, steps, sampler, height, width, model and prompt refining, etc)
* as few dependencies as possible (currently 8 if you want to use a Graphical User Interface)
* as few resources as possible, e.g. systems with 7+ year old processors, only 8GB RAM and 2GB VRAM.

The recommendation at the end of 2022 from StabilityAI is to use xformers for a 25-40 % boost in diffusion speed. This requires Microsoft Visual Studio, Nvidia CUDA, git, ninja & wheel in order to build the .whl file compatible with your system. The payback on the additional 11GB hard drive space and 2 hours setup comes after generating about 3000 images.

[Installation instructions](https://github.com/thekitchenscientist/sd_lite/wiki/Installation) are available in the project wiki, along with a detailed roadmap and guiding prinicples.

## Citations
The Explore (text to image) and Sketch (image to image) modules makes use of Safe Latent Diffusion:  

@article{schramowski2022safe,  
      title={Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models},   
      author={Patrick Schramowski and Manuel Brack and Björn Deiseroth and Kristian Kersting},  
      year={2022},  
      journal={arXiv preprint arXiv:2211.05105}  
} 

The Morph (latent walk) module makes use of stable-diffusion-videos, with the audio/video code removed to just return a series of images.  
[https://github.com/nateraw/stable-diffusion-videos](https://github.com/nateraw/stable-diffusion-videos)

The Edit module makes use of instructpix2pix:  

@InProceedings{brooks2022instructpix2pix,
    author    = {Brooks, Tim and Holynski, Aleksander and Efros, Alexei A.},
    title     = {InstructPix2Pix: Learning to Follow Image Editing Instructions},
    month     = {November},
    year      = {2022},
} 
