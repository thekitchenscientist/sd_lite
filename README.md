Setting up Stable Diffusion 2.1 with the minimum dependancies

I was given an old PC from 2015 equipped with an A10-7870K (3.9GHz, 4 threads) and 8 GB RAM. The GA-F2A78M-HD2 motherboard is compatible with newer NVIDIA cards, so I added a second hand RTX 2060 (6GB VRAM) and upgraded the PSU to a 600W model.

These are the steps I took to get Stable diffusion 2.1 running at 4.6 it/s. I was unable to get most of the pre-built community GUI to work due to the high RAM requirements. The only one that was close was NMKD on the low-VRAM mode but that took 52/s per image!

Commands to enter into the wondows cmd tool are quoted with "" and should be entered without the quotes.

1) Reset Windows 10 to its orginal state. The processor is not supported by Windows 11 due to lack of a TPM 2.0.
2) Install git (https://gitforwindows.org/)
3) Run cmd (as admin), enter "git config --system core.longpaths true" (to avoid a later error message)
3) Install Visual Studio Community Edition with C++ tools (required to build xformers, https://visualstudio.microsoft.com/)
4) Launch visual studio and Select Tools, Options. In the Options dialog, expand Projects and Solutions, select Build and Run. Set the maximum number of parallel project builds to 1 (to avoid a potential later error message).
5) Install python 3.9 (xformers github says it supports python 3.7-3.9, https://www.python.org/downloads/release/python-390/)
6) In the final step of the installer choose the "enable long paths" option.
7) Install CUDA 11.7 (xformers github says it doesn't support 12 yet, https://developer.nvidia.com/cuda-11-7-0-download-archive)
8) Create the python virtual environment to hold all the dependancies for running Stable Diffusion. Run cmd enter "python -m venv c:\sd_lite" (keep the name short)
9) "cd c:\sd_lite\scripts"
10) "activate"
11) "C:\sd_lite\Scripts\python.exe -m pip install --upgrade pip"
12) "pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117" This will take a while as it is a 2.3 GB file
13) check it has worked. From the command line, type:
"python"
then enter the following code:
import torch
x = torch.rand(5, 3)
print(x)

The output should be something similar to:

tensor([[0.3380, 0.3845, 0.3217],
        [0.8337, 0.9050, 0.2650],
        [0.2979, 0.7141, 0.9069],
        [0.1449, 0.1132, 0.1375],
        [0.4675, 0.3947, 0.1426]])
Additionally, to check if your GPU driver and CUDA is enabled and accessible by PyTorch, run the following commands to return whether or not the CUDA driver is enabled:

import torch
torch.cuda.is_available()

14) "pip install jupyterlab"
15) "pip install --upgrade git+https://github.com/huggingface/diffusers.git"
16) "pip install --upgrade git+https://github.com/huggingface/transformers/"
17) "pip install accelerate==0.12.0"
18) "pip install scipy"
19) "pip install ftfy"
20) "pip install gradio -q"

21) Getting xformers to compile took over ten attempts and some serious debugging. Without it the model runs at 3.3 it/s rather than 4.6 it/s. This is what finally worked:
"cd c:\sd_lite\scripts"
"activate"
"cd c:\sd_lite\"
"git clone https://github.com/facebookresearch/xformers.git"
"cd xformers"
"git submodule update --init --recursive"
"set TORCH_CUDA_ARCH_LIST=7.5 (look up your card in the GeForce table to set the correct number, https://developer.nvidia.com/cuda-gpus)"
"pip install ninja"
"pip install wheel"

"python setup.py build" (could take 30+ minutes to complete)
"python setup.py bdist_wheel"

get the .whl file and move it to c:\sd_lite

"cd ../"
22) "pip install 'xformers-0.0.15+ea1048b.d20221221-cp39-cp39-win_amd64.whl'" (yours will be named based on the system you are on)

"python -m xformers.info" will tell you if it has been sucessful (the memory_efficient_attention will list as available):

A matching Triton is not available, some optimizations will not be enabled.
Error caught was: No module named 'triton'
xFormers 0.0.15+ea1048b.d20221221
memory_efficient_attention.cutlassF:               available
memory_efficient_attention.cutlassB:               available
memory_efficient_attention.flshattF:               available
memory_efficient_attention.flshattB:               available
memory_efficient_attention.smallkF:                available
memory_efficient_attention.smallkB:                available
memory_efficient_attention.tritonflashattF:        unavailable
memory_efficient_attention.tritonflashattB:        unavailable
swiglu.fused.p.cpp:                                available
is_triton_available:                               False
is_functorch_available:                            False
pytorch.version:                                   1.13.1+cu117
pytorch.cuda:                                      available
gpu.compute_capability:                            7.5
gpu.name:                                          NVIDIA GeForce RTX 2060

You now have a ~6GB folder containing all the prequisites for Stable Diffusion 2.1

23) Optional: Save the SD 2.1 model files locally (5.1 GB)
"git lfs install"
"git clone --branch fp16 https://huggingface.co/stabilityai/stable-diffusion-2-1-base"
Download LFS model files seperately (unet, VAE, text encoder) and put in the correct folders
Move the entire folder from the C:\user\<user name> to where you actually want to keep it.

24) Save the .ipynb from "https://github.com/qunash/stable-diffusion-2-gui" to c:\sd_lite
25) Run cmd and paste in the following commands:
"cd c:\sd_lite\scripts"
"activate"
"cd c:\sd_lite\"
"jupyter-lab"

26) You can now browse for the notebook and run the app section (if you managed to build xformers set "mem_eff_attn_enabled = True" on line 11). It might take 5 minutes to load the models into VRAM.
27) I'm using the 512x512 model so on line 13 the model_id = 'stabilityai/stable-diffusion-2-1-base' or the folder you created in step 23 e.g. model_id = 'C:/Models/stable-diffusion-2-1-base'
28) Visit the link created by Gradio and start prompting. The first run will be a bit slower, but once it's warmed up you will get the full speed available from your card.

