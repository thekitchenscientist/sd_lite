The aim of this project is to be able to run Stable Diffusion with minimalism. This means:

* as few options as possible (abstract away CFG, steps, sampler, height, width, model and prompt refining)
* as few dependencies as possible ()
* on systems with 7 year old processors, only 8GB RAM and 6GB VRAM (making it work on 3G VRAM is something I'll let someone else try).

The recommendation at the end of 2022 from StabilityAI is to use xformers for a 25% boost in diffusion speed. This requires Microsoft Visual Studio, Nvidia CUDA, git, ninja & wheel in order to build the .whl file compatible with your system. The payback on the additional 11GB hard drive space and 2 hours setup comes after generating about 3000 images.
