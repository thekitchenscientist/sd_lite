# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from itertools import repeat
from typing import Callable, List, Optional, Union

import torch
torch.backends.cudnn.benchmark = True

import math
import numpy as np
import PIL

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import PIL_INTERPOLATION, deprecate, logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image

### Start stable-diffusion-videos ###
# Copyright 2023 MultiDiffusion Authors. All rights reserved."
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2
### End stable-diffusion-videos ###

### Start Panorama Diffusion ###
# Copyright 2023 MultiDiffusion Authors. All rights reserved."
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def get_views(pan_height, pan_width, window_size=64, stride=8):
    #pan_height /= 8
    #pan_width /= 8
    num_blocks_height = (pan_height - window_size) // stride + 1
    num_blocks_width = (pan_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views
### End Panorama Diffusion ###

def sigmoid(x):
  y = np.zeros(len(x))
  for i in range(len(x)):
    y[i] = 1 / (1 + math.exp(-x[i]))
  return y

### Start Dynamic Scale ###
"""The MIT License (MIT)

Copyright (c) 2023 Alex "mcmonkey" Goodwin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
def modify_scale(num_inference_steps,step, min_scale, dynamic_scale_factor, scale):
    scale -= min_scale
    max_scale = float(num_inference_steps - 1)
    scale *= math.pow(float(step) / max_scale, dynamic_scale_factor)
    scale += min_scale
    return round(scale,3)

def get_dynamic_scale(num_inference_steps,guidance_scale,dynamic_scale_factor):
    guidance_list=[]
    for i in range (num_inference_steps):
        if dynamic_scale_factor == 0:
            modified_scale = guidance_scale
        elif guidance_scale >=9:
            modified_scale = modify_scale(num_inference_steps,i, guidance_scale/2, dynamic_scale_factor, guidance_scale)          
        #elif guidance_scale <=5:
        #    modified_scale = modify_scale(num_inference_steps,i, guidance_scale, dynamic_scale_factor, guidance_scale*1.5)            
        else: 
            modified_scale = modify_scale(num_inference_steps,i, guidance_scale-1, dynamic_scale_factor, guidance_scale+1)
        guidance_list.extend([modified_scale]) 
    return guidance_list
### End Dynamic Scale ###

class StableDiffusionMultiPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            # TODO(Patrick) - there is currently a bug with cpu offload of nn.Parameter in accelerate
            # fix by only offloading self.safety_checker for now
            cpu_offload(self.safety_checker.vision_model, device)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, alt_prompt_list, enable_edit_guidance, editing_prompt_prompt_embeddings, editing_prompt):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            
            text_embeddings_list = [uncond_embeddings, text_embeddings]            
            ### AlternativePrompt ###
            
            # get alt_prompt text embeddings
            if len(alt_prompt_list) > 0:
                for alt_prompt in alt_prompt_list:
                    alt_prompt_input = self.tokenizer(
                        alt_prompt,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                    if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                        attention_mask = alt_prompt_input.attention_mask.to(device)
                    else:
                        attention_mask = None            

                    alt_prompt_embeddings = self.text_encoder(
                        alt_prompt_input.input_ids.to(device),
                        attention_mask=attention_mask,
                    )
                    alt_prompt_embeddings = alt_prompt_embeddings[0]

                    # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                    seq_len = alt_prompt_embeddings.shape[1]
                    alt_prompt_embeddings = alt_prompt_embeddings.repeat(1, num_images_per_prompt, 1)
                    alt_prompt_embeddings = alt_prompt_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)
                    
                    text_embeddings_list.append(alt_prompt_embeddings)
            ### End AlternativePrompt ###   
            
            ### SEGA ###
            if enable_edit_guidance:
                # get safety text embeddings
                if editing_prompt_prompt_embeddings is None:
                    edit_concepts_input = self.tokenizer(
                        [x for item in editing_prompt for x in repeat(item, batch_size)],
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        return_tensors="pt",
                    )

                    edit_concepts_input_ids = edit_concepts_input.input_ids

                    if edit_concepts_input_ids.shape[-1] > self.tokenizer.model_max_length:
                        removed_text = self.tokenizer.batch_decode(
                            edit_concepts_input_ids[:, self.tokenizer.model_max_length :]
                        )
                        logger.warning(
                            "The following part of your input was truncated because CLIP can only handle sequences up to"
                            f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                        )
                        edit_concepts_input_ids = edit_concepts_input_ids[:, : self.tokenizer.model_max_length]
                    edit_concepts = self.text_encoder(edit_concepts_input_ids.to(self.device))[0]
                else:
                    edit_concepts = editing_prompt_prompt_embeddings.to(self.device).repeat(batch_size, 1, 1)

                # duplicate text embeddings for each generation per prompt, using mps friendly method
                bs_embed_edit, seq_len_edit, _ = edit_concepts.shape
                edit_concepts = edit_concepts.repeat(1, num_images_per_prompt, 1)
                edit_concepts = edit_concepts.view(bs_embed_edit * num_images_per_prompt, seq_len_edit, -1)
            
                text_embeddings_list.append(edit_concepts)                    
                    
            text_embeddings = torch.cat(text_embeddings_list)

                        
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            #text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
            
    def get_timesteps_img2img(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def prepare_latents_img2img(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)

        init_latents = 0.18215 * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        rand_device = "cpu" if device.type == "mps" else device
        shape = init_latents.shape
        if isinstance(generator, list):
            shape = (1,) + shape[1:]
            noise = [
                torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype) for i in range(batch_size)
            ]
            noise = torch.cat(noise, dim=0).to(device)
        else:
            noise = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents
    
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        alt_prompt: Optional[Union[str, List[str]]] = None,
        alt_mode: Optional[str] = "0.2",
        alt_mode_warm_up: Optional[int] = 3,
        alt_mode_cool_down: Optional[int] = 0,
        image: Optional[Union[torch.FloatTensor, PIL.Image.Image]] = None,
        strength: Optional[float] = 0.0,
        cross_fade: Optional[Union[int, List[int]]] = 7,        
        dynamic_scale_factor: Optional[int] = 4,
        dynamic_scale_mimic: Optional[float] = 7,  
        dynamic_scale_threshold_percentile: Optional[float] = 0.9,   
        pan_window_size: Optional[int] = 512, 
        pan_stride: Optional[int] = 0,
        post_process_window_size: Optional[int] = 512,
        post_process_recombine_image: bool = True,
        editing_prompt: Optional[Union[str, List[str]]] = None,
        editing_prompt_prompt_embeddings=None,
        reverse_editing_direction: Optional[Union[bool, List[bool]]] = False,
        edit_guidance_scale: Optional[Union[float, List[float]]] = 500,
        edit_warmup_steps: Optional[Union[int, List[int]]] = 10,
        edit_cooldown_steps: Optional[Union[int, List[int]]] = None,
        edit_threshold: Optional[Union[float, List[float]]] = None,
        edit_momentum_scale: Optional[float] = 0.1,
        edit_mom_beta: Optional[float] = 0.4,
        edit_weights: Optional[List[float]] = None,
        sem_guidance = None,
        sld_concept: Optional[str] = 'hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use, theft, vandalism, weapons, child abuse, brutality, cruelty',
        sld_guidance_scale: Optional[int] = 5000,
        sld_warmup_steps = 7,
        sld_threshold: Optional[float] = 0.01,
        sld_momentum_scale: Optional[float] = 0.5,
        sld_mom_beta: Optional[float] = 0.7,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            alt_prompt  (`str` or `List[str]`, *optional*):
                An additonal noise channel to guide the image creation.
            alt_mode  (`str`, *optional*, defaults to "0.15"):
                How the additonal noise channel should be used.
                #64, 128, 192
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            editing_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to use for Semantic guidance. Semantic guidance is disabled by setting
                `editing_prompt = None`. Guidance direction of prompt should be specified via
                `reverse_editing_direction`.
            reverse_editing_direction (`bool` or `List[bool]`, *optional*):
                Whether the corresponding prompt in `editing_prompt` should be increased or decreased.
            edit_guidance_scale (`float` or `List[float]`, *optional*, defaults to 5):
                Guidance scale for semantic guidance. If provided as list values should correspond to `editing_prompt`.
            edit_warmup_steps (`float` or `List[float]`, *optional*, defaults to 10):
                Number of diffusion steps (for each prompt) for which semantic guidance will not be applied. Momentum
                will still be calculated for those steps and applied once all warmup periods are over.
            edit_cooldown_steps (`float` or `List[float]`, *optional*, defaults to 10):
                Number of diffusion steps (for each prompt) after which semantic guidance will no longer be applied.
            edit_threshold (`float` or `List[float]`, *optional*, defaults to `None`):
                Threshold of semantic guidance.
            edit_momentum_scale (`float`, *optional*, defaults to 0.1):
                Scale of the momentum to be added to the semantic guidance at each diffusion step. If set to 0.0 momentum
                will be disabled. Momentum is already built up during warmup, i.e. for diffusion steps smaller than
                `sld_warmup_steps`. Momentum will only be added to latent guidance once all warmup periods are
                finished.
            edit_mom_beta (`float`, *optional*, defaults to 0.4):
                Defines how semantic guidance momentum builds up. `edit_mom_beta` indicates how much of the previous
                momentum will be kept. Momentum is already built up during warmup, i.e. for diffusion steps smaller
                than `edit_warmup_steps`.
            edit_weights (`List[float]`, *optional*, defaults to `None`):
                Indicates how much each individual concept should influence the overall guidance. If no weights are
                provided all concepts are applied equally.
            sld_concept (`str`, *optional*, defaults to "harms identified by research")
                Steer the generation away from the listed concepts. Requires to be active for at least 15 steps to be fully effective
            sld_guidance_scale (`float`, *optional*, defaults to 1000):
                The guidance scale of safe latent diffusion. If set to be less than 1, safety guidance will be disabled.
            sld_warmup_steps (`int`, *optional*, defaults to 10):
                Number of warmup steps for safety guidance. SLD will only be applied for diffusion steps greater
                than `sld_warmup_steps`.
            sld_threshold (`float`, *optional*, defaults to 0.01):
                Threshold that separates the hyperplane between appropriate and inappropriate images.
            sld_momentum_scale (`float`, *optional*, defaults to 0.3):
                Scale of the SLD momentum to be added to the safety guidance at each diffusion step.
                If set to 0.0 momentum will be disabled.  Momentum is already built up during warmup,
                i.e. for diffusion steps smaller than `sld_warmup_steps`.
            sld_mom_beta (`float`, *optional*, defaults to 0.4):
                Defines how safety guidance momentum builds up. `sld_mom_beta` indicates how much of the previous
                momentum will be kept. Momentum is already built up during warmup, i.e. for diffusion steps smaller than
                `sld_warmup_steps`.                
            

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        ### SaferDiffusion & SEGA ###
        """MIT License
        Copyright (c) 2022 Manuel Brack

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        
        arXiv:2211.05105
        Config δ sS λ sm βm
        Hyp-Weak 15 200 0.0 0.0 -
        Hyp-Medium 10 1000 0.01 0.3 0.4
        Hyp-Strong 7 2000 0.025 0.5 0.7
        Hyp-Max 0 5000 1.0 0.5 0.7
        """   
        
           
        safety_momentum = None
        if num_inference_steps < 30 & sld_warmup_steps == 7:
            sld_warmup_steps = int(num_inference_steps*0.2)
            
        #enabled_editing_prompts = 0
        if editing_prompt:
            enable_edit_guidance = True
            if isinstance(editing_prompt, str):
                editing_prompt = [editing_prompt]
            enabled_editing_prompts = len(editing_prompt)
        elif editing_prompt_prompt_embeddings is not None:
            enable_edit_guidance = True
            enabled_editing_prompts = editing_prompt_prompt_embeddings.shape[0]
        else:
            enabled_editing_prompts = 0
            enable_edit_guidance = False
        ### End SEGA ###
         
        ### AlternativePrompt ###
        if sld_guidance_scale > 0 and pan_stride ==0:
            alt_prompt_list = [sld_concept]
        else:
            alt_prompt_list = []

        temporal_prompt_weight_list = []
        if isinstance(cross_fade,list):
            prompt_blend_start = cross_fade[0]
            prompt_blend_end = cross_fade[1]
        else:
            prompt_blend_start = prompt_blend_end = cross_fade    
        
        change_over_step = int(num_inference_steps+1)

        # prepare the switch over for alt_prompt
        blend_step_size = (prompt_blend_start+prompt_blend_end)/num_inference_steps

        if alt_prompt is not None:
            if isinstance(alt_prompt, str):
                alt_prompt = [[alt_prompt]]
            elif isinstance(alt_prompt[0], str):
                alt_prompt = [alt_prompt]
                
            alt_prompt_to_embed =  [item for sublist in alt_prompt for item in sublist]
            #print(alt_prompt_to_embed)
            if alt_mode=="stack":
                height = pan_stride*len(alt_prompt)+pan_stride//3
                width = pan_stride*len(alt_prompt[0])+pan_stride//3
            elif alt_mode=="multifade":
                height = pan_stride*len(alt_prompt)+(pan_window_size-pan_stride)
                width = pan_stride*len(alt_prompt[0])+(pan_window_size-pan_stride)

            for current_prompt in alt_prompt_to_embed:
                if len(alt_mode) <= 2:
                    current_prompt = prompt + ". "+ current_prompt
                    change_over_step = float(alt_mode)
                    temporal_prompt_weight_list = [0 if i >= change_over_step else 1 for i in range(num_inference_steps)]
                elif len(alt_mode) <= 4:
                    current_prompt = prompt + ". "+ current_prompt
                    change_over_step = int(num_inference_steps*float(alt_mode))
                    temporal_prompt_weight_list = [0 if i >= change_over_step else 1 for i in range(num_inference_steps)]
                elif alt_mode[:4]=="walk" or alt_mode=="crossfade":
                    current_prompt = prompt + ", "+ current_prompt
                    temporal_prompt_weight_list = [1 for i in range(num_inference_steps)]
                elif alt_mode=="stack" or alt_mode=="multifade":
                    current_prompt = current_prompt + ", "+ prompt
                    temporal_prompt_weight_list = [1 for i in range(num_inference_steps)]
                elif alt_mode == "alternating":
                    temporal_prompt_weight_list = [0 if i % 2 == 0 else 1 for i in range(num_inference_steps)]      
                elif alt_mode == "decreasing":
                    temporal_prompt_weight_list = sigmoid(np.arange(start=1*prompt_blend_start, stop=-1*prompt_blend_end, step=-1*blend_step_size))      
                elif alt_mode == "increasing":
                    temporal_prompt_weight_list = sigmoid(np.arange(start=1*prompt_blend_start, stop=-1*prompt_blend_end, step=-1*blend_step_size))
                    temporal_prompt_weight_list = temporal_prompt_weight_list[::-1]
                elif alt_mode[:6] == "switch" or alt_mode[:5] == "delay":
                    change_over_step = int(num_inference_steps*float(alt_mode[-3:-1])/100)
                    temporal_prompt_weight_list = [0 if i >= change_over_step else 1 for i in range(num_inference_steps)]      
                elif alt_mode[:6] == "weight":
                    weight = 1-float(alt_mode[-2:])/100
                    temporal_prompt_weight_list = [weight for i in range(num_inference_steps)]  
                elif alt_mode[:6] == "mirror" or alt_mode[:6] == "rotate":
                    current_prompt = prompt + ". "+ current_prompt
                    change_over_step = alt_mode_warm_up    
                else:
                    temporal_prompt_weight_list = [1 for i in range(num_inference_steps)]
                alt_prompt_list.append(current_prompt)
            #print(alt_mode, alt_prompt_list)
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, alt_prompt_list, enable_edit_guidance, editing_prompt_prompt_embeddings, editing_prompt
        )
        
        latent_multiplier = 2
        if sld_guidance_scale > 0 and pan_stride ==0:
            latent_multiplier +=1
        if alt_prompt is not None:
            latent_multiplier +=len(alt_prompt_to_embed)  

        guidance_scale_mimic = dynamic_scale_mimic  
        threshold_percentile = dynamic_scale_threshold_percentile        
        guidance_scale_list = []
        
        guidance_scale_list = get_dynamic_scale(num_inference_steps,guidance_scale,dynamic_scale_factor)
        #guidance_scale_list.append(guidance_scale_)
        
        if strength>0.0:
            # 4. Preprocess image
            image = preprocess(image)

            # 5. set timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps, num_inference_steps = self.get_timesteps_img2img(num_inference_steps, strength, device)
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

            # 6. Prepare latent variables
            latents = self.prepare_latents_img2img(
                image, latent_timestep, batch_size, num_images_per_prompt, text_embeddings.dtype, device, generator
            )
        else:
            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # 5. Prepare latent variables
            num_channels_latents = self.unet.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents,
            )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop 
             
        ### SEGA ###
        latents_shape = (batch_size * num_images_per_prompt, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype
        edit_momentum = None
        #print(enable_edit_guidance,enabled_editing_prompts,editing_prompt)

        self.uncond_estimates = None
        self.text_estimates = None
        self.edit_estimates = None
        self.sem_guidance = None
        
        ### End SaferDiffusion & SEGA ###

        ### Start Panorama Diffusion ###
        if  pan_stride > 0:
            if alt_mode[:4]=="walk":
                walk_steps = int(alt_mode[5:])
                walk_stops = len(alt_prompt[0])
                width = int(width * (walk_steps*(walk_stops-1)+walk_stops))
                temp_canvas = torch.zeros(num_images_per_prompt,4,height//8,width//8).cuda()
                temp_views = get_views(height//8,width//8, pan_window_size//8, pan_stride//8)
                for h_start, h_end, w_start, w_end in temp_views:
                    temp_canvas[:, :, h_start:h_end, w_start:w_end] += latents
                latents=temp_canvas.to(torch.float16)
                
            window_size=pan_window_size//8
            pan_height = height//8
            pan_width = width//8
            # stride cant be half size of window
            if pan_stride > pan_window_size:
                stride=window_size
            else:
                stride=pan_stride//8

            views = get_views(pan_height, pan_width, window_size, stride)
            count = torch.zeros_like(latents)
            value = torch.zeros_like(latents)

            # prompt blending
            num_blocks_height = (pan_height - window_size) // stride + 1
            num_blocks_width = (pan_width - window_size) // stride + 1
            overlap = 0
                        
            prompt_weight_list = [1 for i in range(len(views))]
            
            if alt_mode[:4]=="walk":
                walk_weight_list = np.linspace(0.0, 1.0, walk_steps+2)
                walk_embeds_list = []
                walk_text_embeddings = text_embeddings.chunk(latent_multiplier)
                current_stop=2
                for i in range(walk_stops-1):
                    for j, weights in enumerate(walk_weight_list):
                        if j ==walk_steps+1 and i < walk_stops-2:
                            break
                        embeds = torch.lerp(walk_text_embeddings[current_stop], walk_text_embeddings[current_stop+1], weights)
                        walk_embeds_list.append(embeds)
                    current_stop +=1
                latent_multiplier = 2
            elif alt_mode=="stack":
                stack_text_embeddings = text_embeddings.chunk(latent_multiplier)
                latent_multiplier = 2 
            elif alt_mode=="multifade":
                stack_text_embeddings = text_embeddings.chunk(latent_multiplier)
                latent_multiplier = 2 
                #workout region of overlap
                overlap = (window_size-stride)
                #create fade 
                start = 1
                stop = 0.1
                step_size = (overlap)/(start+stop)

                # create masks with gradients
                left_latent_fade = np.arange(start=start, stop=stop, step=1/step_size)
                left_latent_fade =left_latent_fade.reshape(-1,1)
                up_alpha = np.repeat(left_latent_fade, repeats=window_size, axis=1)
                #extent to size of window
                pass_through = np.ones((stride+overlap,window_size))
                top_mask = np.concatenate((up_alpha, pass_through), axis=0)
                #rotate by 90
                left_mask = np.rot90(top_mask)
                left_top_mask = top_mask*left_mask
                top_mask = torch.from_numpy(top_mask.copy()).cuda() 
                left_mask = torch.from_numpy(left_mask.copy()).cuda() 
                left_top_mask = torch.from_numpy(left_top_mask.copy()).cuda()
                
            elif alt_mode=="crossfade":
                if isinstance(cross_fade,list):
                    prompt_blend_start = cross_fade[0]
                    prompt_blend_end = cross_fade[1]
                else:
                    prompt_blend_start = prompt_blend_end = cross_fade

                crossfade_text_embeddings = text_embeddings.chunk(latent_multiplier)
                latent_multiplier = 3 

                if height == width or height > width:
                    blend_step_size = (prompt_blend_start+prompt_blend_end)/num_blocks_height
                    prompt_weight_list = sigmoid(np.arange(start=1*prompt_blend_start, stop=-1*prompt_blend_end, step=-1*blend_step_size))
                    prompt_weight_list =prompt_weight_list.round(2)
                    prompt_weight_list = np.repeat(prompt_weight_list, repeats=num_blocks_width, axis=0)
                else:
                    blend_step_size = (prompt_blend_start+prompt_blend_end)/num_blocks_width
                    prompt_weight_list = sigmoid(np.arange(start=1*prompt_blend_start, stop=-1*prompt_blend_end, step=-1*blend_step_size))
                    prompt_weight_list =prompt_weight_list.reshape(-1,1).round(2)
                    prompt_weight_list = np.repeat(prompt_weight_list, repeats=num_blocks_height, axis=1)
                    prompt_weight_list = prompt_weight_list.T.flatten()
                                            
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    count.zero_()
                    value.zero_()
                    #print(i)
                    slice_count = 0
                    for h_start, h_end, w_start, w_end in views:
                        # TODO we can support batches, and pass multiple views at once to the unet
                        latent_view = latents[:, :, h_start:h_end, w_start:w_end]
                        
                        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                        latent_model_input = torch.cat([latent_view] * latent_multiplier)
                        latent_model_input = latent_model_input.to(torch.float16)
                        # predict the noise residual
                        if alt_mode[:4]=="walk":
                            new_text_embeddings=torch.cat([walk_text_embeddings[0],walk_embeds_list[slice_count]])
                        elif alt_mode=="stack" or alt_mode=="multifade":
                            new_text_embeddings=torch.cat([stack_text_embeddings[0],stack_text_embeddings[slice_count+2]])
                            #print(alt_prompt_list[slice_count])
                        elif alt_mode=="crossfade":
                            new_text_embeddings=torch.cat([crossfade_text_embeddings[0],crossfade_text_embeddings[2],crossfade_text_embeddings[3]])
                        else:
                            new_text_embeddings=text_embeddings
                            
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=new_text_embeddings).sample

                        # perform guidance
                        noise_pred_out = noise_pred.chunk(latent_multiplier)  # [b,4, 64, 64]
                        noise_pred_uncond, noise_pred_text = noise_pred_out[0], noise_pred_out[1]
                        if alt_prompt is not None and (alt_mode=="crossfade" or alt_mode=="stack"):
                            noise_pred_alt_text = noise_pred_out[latent_multiplier-1]
                            noise_pred_text = noise_pred_text*prompt_weight_list[slice_count]+noise_pred_alt_text*(1-prompt_weight_list[slice_count])

                                            
                        guidance_scale = guidance_scale_list[i]
                        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        # compute the denoising step with the reference model
                        latents_view_denoised = self.scheduler.step(noise_pred, t, latent_view).prev_sample
                        if alt_mode=="multifade":
                            if h_start==0 and w_start >0:
                                latents_view_denoised=latents_view_denoised*left_mask
                            elif h_start>0 and w_start ==0:
                                latents_view_denoised=latents_view_denoised*top_mask
                            elif h_start>0 and w_start >0:
                                latents_view_denoised=latents_view_denoised*left_top_mask

                        value[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised   
                        count[:, :, h_start:h_end, w_start:w_end] += 1
                        slice_count += 1
                    # take the MultiDiffusion step
                    latents = torch.where(count > 0, value / count, value)
                    
                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)
        ### End Panorama Diffusion ###
        else:
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * (latent_multiplier+ enabled_editing_prompts)) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    #print(latent_model_input.shape, t, text_embeddings.shape)
                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                    
                    # perform guidance
                    if do_classifier_free_guidance:
                        #noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        #noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        
                        noise_pred_out = noise_pred.chunk(latent_multiplier + enabled_editing_prompts)  # [b,4, 64, 64]
                        noise_pred_uncond, noise_pred_text = noise_pred_out[0], noise_pred_out[1]

                        ### Alternative Prompt ###
                        if alt_prompt is not None:
                            noise_pred_alt_text = noise_pred_out[latent_multiplier-1]
                            noise_pred_text = (noise_pred_text*(temporal_prompt_weight_list[i])+noise_pred_alt_text*(1-temporal_prompt_weight_list[i]))
                        ### End Alternative Prompt ###
                        
                        noise_guidance = (noise_pred_text - noise_pred_uncond)

                        ### Mirroring and Rotation ###
                        if alt_prompt is None:
                            None
                        elif alt_mode == 'mirror up:down' and i > change_over_step:# and i % 3 == 0:
                            noise_guidance = torch.flipud(noise_guidance)#, [3])
                        elif alt_mode == 'mirror left:right' and i > change_over_step:# and i % 3 == 0:
                            noise_guidance = torch.fliplr(noise_guidance)#, [2])
                        elif alt_mode == 'rotate 90' and i > change_over_step:# and i % 3 == 0:
                            noise_guidance = torch.rot90(noise_guidance, dims=[2, 3])
                        elif alt_mode == 'rotate 180' and i > change_over_step:# and i % 3 == 0:
                            noise_guidance = torch.rot90(torch.rot90(noise_guidance, dims=[2, 3]),dims=[2, 3])
                        ### End Mirroring and Rotation ###

                        ### Start SaferDiffusion ###
                        if sld_guidance_scale > 0:
                            noise_pred_safety_concept = noise_pred_out[2]
                            if safety_momentum is None:
                                safety_momentum = torch.zeros_like(noise_guidance)
                            #noise_pred_safety_concept = noise_pred_out[2]

                            # Equation 6
                            scale = torch.clamp(
                                torch.abs((noise_pred_text - noise_pred_safety_concept)) * sld_guidance_scale, max=1.)

                            # Equation 6
                            safety_concept_scale = torch.where(
                                (noise_pred_text - noise_pred_safety_concept) >= sld_threshold,
                                torch.zeros_like(scale), scale)

                            # Equation 4
                            noise_guidance_safety = torch.mul(
                                (noise_pred_safety_concept - noise_pred_uncond), safety_concept_scale)

                            # Equation 7
                            noise_guidance_safety = noise_guidance_safety + sld_momentum_scale * safety_momentum

                            # Equation 8
                            safety_momentum = sld_mom_beta * safety_momentum + (1 - sld_mom_beta) * noise_guidance_safety

                            if i >= sld_warmup_steps: # Warmup
                                # Equation 3
                                noise_guidance = noise_guidance - noise_guidance_safety
                        ### End SaferDiffusion ###
                            
                        ### Start Dynamic Scale ###                    
                        guidance_scale = guidance_scale_list[i]
                        noise_pred = noise_pred_uncond + guidance_scale * noise_guidance
                        noise_pred_mimic = noise_pred_uncond + guidance_scale_mimic * noise_guidance
                        ### If we weren't doing mimic scale, we'd just return noise_pred here

                        ### Now recenter the values relative to their average rather than absolute, to allow scaling from average
                        mim_flattened = noise_pred_mimic.flatten(2)
                        cfg_flattened = noise_pred.flatten(2)
                        mim_means = mim_flattened.mean(dim=2).unsqueeze(2)
                        cfg_means = cfg_flattened.mean(dim=2).unsqueeze(2)
                        mim_centered = mim_flattened - mim_means
                        cfg_centered = cfg_flattened - cfg_means

                        ### Get the maximum value of all datapoints (with an optional threshold percentile on the uncond)
                        mim_max = mim_centered.abs().max(dim=2).values.unsqueeze(2)
                        orig_dtype = cfg_centered.dtype
                        if orig_dtype not in [torch.float, torch.double]:
                            cfg_centered = cfg_centered.float()
                        cfg_max = torch.quantile(cfg_centered.abs(), threshold_percentile, dim=2).unsqueeze(2)
                        actualMax = torch.maximum(cfg_max, mim_max)

                        ### Clamp to the max
                        cfg_centered = cfg_centered.type(orig_dtype)
                        cfg_clamped = cfg_centered.clamp(-actualMax, actualMax)
                        ### Now shrink from the max to normalize and grow to the mimic scale (instead of the CFG scale)
                        cfg_renormalized = (cfg_clamped / actualMax) * mim_max

                        ### Now add it back onto the averages to get into real scale again and return
                        result = cfg_renormalized + cfg_means
                        noise_pred = result.unflatten(2, noise_pred_mimic.shape[2:])
                        if orig_dtype not in [torch.float, torch.double]:
                            noise_pred = noise_pred.to(torch.float16)
                        ### End Dynamic Scale ###
                                                
                        
                        ### SEGA ###
                        noise_pred_edit_concepts = noise_pred_out[latent_multiplier:]    
                        if self.uncond_estimates is None:
                            self.uncond_estimates = torch.zeros((num_inference_steps+1, *noise_pred_uncond.shape))
                        self.uncond_estimates[i] = noise_pred_uncond.detach().cpu()

                        if self.text_estimates is None:
                            self.text_estimates = torch.zeros((num_inference_steps+1, *noise_pred_text.shape))
                        self.text_estimates[i] = noise_pred_text.detach().cpu()

                        if self.edit_estimates is None and enable_edit_guidance:
                            self.edit_estimates = torch.zeros((num_inference_steps+1, len(noise_pred_edit_concepts), *noise_pred_edit_concepts[0].shape))

                        if self.sem_guidance is None:
                            self.sem_guidance = torch.zeros((num_inference_steps + 1, *noise_pred_text.shape))
                            
                        if edit_momentum is None:
                            edit_momentum = torch.zeros_like(noise_guidance)
                    
                        if enable_edit_guidance:
                            noise_guidance = guidance_scale * (noise_pred_text - noise_pred_uncond)

                            concept_weights = torch.zeros(
                                (len(noise_pred_edit_concepts), noise_guidance.shape[0]), device=self.device
                            )
                            noise_guidance_edit = torch.zeros(
                                (len(noise_pred_edit_concepts), *noise_guidance.shape), device=self.device
                            )
                            # noise_guidance_edit = torch.zeros_like(noise_guidance)
                            warmup_inds = []
                            for c, noise_pred_edit_concept in enumerate(noise_pred_edit_concepts):
                                self.edit_estimates[i, c] = noise_pred_edit_concept
                                if isinstance(edit_guidance_scale, list):
                                    edit_guidance_scale_c = edit_guidance_scale[c]
                                else:
                                    edit_guidance_scale_c = edit_guidance_scale

                                if isinstance(edit_threshold, list):
                                    edit_threshold_c = edit_threshold[c]
                                else:
                                    edit_threshold_c = edit_threshold
                                if isinstance(reverse_editing_direction, list):
                                    reverse_editing_direction_c = reverse_editing_direction[c]
                                else:
                                    reverse_editing_direction_c = reverse_editing_direction
                                if edit_weights:
                                    edit_weight_c = edit_weights[c]
                                else:
                                    edit_weight_c = 1.0
                                if isinstance(edit_warmup_steps, list):
                                    edit_warmup_steps_c = edit_warmup_steps[c]
                                else:
                                    edit_warmup_steps_c = edit_warmup_steps

                                if isinstance(edit_cooldown_steps, list):
                                    edit_cooldown_steps_c = edit_cooldown_steps[c]
                                elif edit_cooldown_steps is None:
                                    edit_cooldown_steps_c = i + 1
                                else:
                                    edit_cooldown_steps_c = edit_cooldown_steps
                                if i >= edit_warmup_steps_c:
                                    warmup_inds.append(c)
                                if i >= edit_cooldown_steps_c:
                                    noise_guidance_edit[c, :, :, :, :] = torch.zeros_like(noise_pred_edit_concept)
                                    continue

                                noise_guidance_edit_tmp = noise_pred_edit_concept - noise_pred_uncond
                                # tmp_weights = (noise_pred_text - noise_pred_edit_concept).sum(dim=(1, 2, 3))
                                tmp_weights = (noise_guidance - noise_pred_edit_concept).sum(dim=(1, 2, 3))

                                tmp_weights = torch.full_like(tmp_weights, edit_weight_c) #* (1 / enabled_editing_prompts)
                                if reverse_editing_direction_c:
                                    noise_guidance_edit_tmp = noise_guidance_edit_tmp * -1
                                concept_weights[c, :] = tmp_weights

                                noise_guidance_edit_tmp = noise_guidance_edit_tmp * edit_guidance_scale_c

                                if latents_dtype not in [torch.float, torch.double]:
                                    noise_guidance_edit_tmp = noise_guidance_edit_tmp.float()
                                tmp = torch.quantile(torch.abs(noise_guidance_edit_tmp).flatten(start_dim=2), edit_threshold_c, dim=2, keepdim=False)
                                #noise_guidance_edit_tmp = noise_guidance_edit_tmp.type(orig_dtype)                        
                                noise_guidance_edit_tmp = torch.where(
                                    torch.abs(noise_guidance_edit_tmp) >= tmp[:, :, None, None]
                                    , noise_guidance_edit_tmp
                                    , torch.zeros_like(noise_guidance_edit_tmp)
                                )

                                noise_guidance_edit[c, :, :, :, :] = noise_guidance_edit_tmp


                                # noise_guidance_edit = noise_guidance_edit + noise_guidance_edit_tmp

                            warmup_inds = torch.tensor(warmup_inds).to(self.device)
                            if len(noise_pred_edit_concepts) > warmup_inds.shape[0] > 0:
                                concept_weights = concept_weights.to("cpu")  # Offload to cpu
                                noise_guidance_edit = noise_guidance_edit.to("cpu")

                                concept_weights_tmp = torch.index_select(concept_weights.to(self.device), 0, warmup_inds)
                                concept_weights_tmp = torch.where(
                                    concept_weights_tmp < 0, torch.zeros_like(concept_weights_tmp), concept_weights_tmp
                                )
                                concept_weights_tmp = concept_weights_tmp / concept_weights_tmp.sum(dim=0)
                               # concept_weights_tmp = torch.nan_to_num(concept_weights_tmp)

                                noise_guidance_edit_tmp = torch.index_select(
                                    noise_guidance_edit.to(self.device), 0, warmup_inds
                                )
                                noise_guidance_edit_tmp = torch.einsum(
                                    "cb,cbijk->bijk", concept_weights_tmp, noise_guidance_edit_tmp
                                )
                                noise_guidance_edit_tmp = noise_guidance_edit_tmp
                                noise_guidance = noise_guidance + noise_guidance_edit_tmp

                                self.sem_guidance[i] = noise_guidance_edit_tmp.detach().cpu()

                                del noise_guidance_edit_tmp
                                del concept_weights_tmp
                                concept_weights = concept_weights.to(self.device)
                                noise_guidance_edit = noise_guidance_edit.to(self.device)

                            concept_weights = torch.where(
                                concept_weights < 0, torch.zeros_like(concept_weights), concept_weights
                            )

                            concept_weights = torch.nan_to_num(concept_weights)
                            noise_guidance_edit = torch.einsum("cb,cbijk->bijk", concept_weights, noise_guidance_edit)

                            noise_guidance_edit = noise_guidance_edit + edit_momentum_scale * edit_momentum

                            edit_momentum = edit_mom_beta * edit_momentum + (1 - edit_mom_beta) * noise_guidance_edit

                            if warmup_inds.shape[0] == len(noise_pred_edit_concepts):
                                noise_guidance = noise_guidance + noise_guidance_edit
                                self.sem_guidance[i] = noise_guidance_edit.detach().cpu()

                            noise_pred = noise_pred_uncond + noise_guidance

                            #convert back to float16 if required
                            if latents_dtype not in [torch.float, torch.double]:
                                noise_pred = noise_pred.to(torch.float16)

                        if sem_guidance is not None:
                            if not enable_edit_guidance:
                                noise_guidance = guidance_scale * (noise_pred_text - noise_pred_uncond)
                            
                            edit_guidance = sem_guidance[i].to(self.device)
                            noise_guidance = noise_guidance + edit_guidance

                            noise_pred = noise_pred_uncond + noise_guidance

                            #convert back to float16 if required
                            if latents_dtype not in [torch.float, torch.double]:
                                noise_pred = noise_pred.to(torch.float16)
                        ### End SEGA ###                            

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

        # 8. Post-processing
        ### Start VAE Chop and Reassemble ###
        """MIT License
        Copyright (c) 2023 thekitchenscientist

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE."""    
        if width*height > post_process_window_size*post_process_window_size:
            vae_image_window = int(post_process_window_size)
            image_offset = vae_image_window//2
            # decode latents in small squares using less passes than denoising
            image_parts = []
            views = get_views(height//8, width//8, window_size=vae_image_window//8, stride=image_offset//8)
            for h_start, h_end, w_start, w_end in views:
                latent_part = latents[:, :, h_start:h_end, w_start:w_end]
                latent_part = 1 / 0.18215 * latent_part
                image_part = self.vae.decode(latent_part).sample
                image_parts.append(image_part)
                
            if post_process_recombine_image:
                
                up_alpha =[]
                down_alpha=[]      
                start = 5
                stop = 5
                step_size = (image_offset)/(start+stop)

                # create masks with gradients
                up_sigmoid_ = sigmoid(np.arange(start=1*start, stop=-1*stop, step=-1/step_size))
                up_sigmoid_ =up_sigmoid_.reshape(-1,1)
                up_alpha = np.repeat(up_sigmoid_, repeats=vae_image_window, axis=1)

                down_sigmoid_ = sigmoid(np.arange(start=-1*start, stop=1*stop, step=1/step_size))
                down_sigmoid_ =down_sigmoid_.reshape(-1,1)
                down_alpha = np.repeat(down_sigmoid_, repeats=vae_image_window, axis=1)

                alpha_centre = np.concatenate((up_alpha, down_alpha), axis=0)
                alpha_centre=torch.from_numpy(alpha_centre).cuda() 
                alpha_edge = np.concatenate((down_alpha,up_alpha), axis=0)
                alpha_edge=torch.from_numpy(alpha_edge).cuda() 

                w_alpha_centre=torch.rot90(alpha_centre).cuda() 
                w_alpha_edge=torch.rot90(alpha_edge).cuda() 
                h_alpha_centre = alpha_centre
                h_alpha_edge = alpha_edge

                canvas = torch.zeros(num_images_per_prompt,3,height,width).cuda() 
                h_seams = torch.zeros(num_images_per_prompt,3,height,width).cuda()
                w_seams = torch.zeros(num_images_per_prompt,3,height,width).cuda()
                c_seams = torch.zeros(num_images_per_prompt,3,height,width).cuda()
                canvas_mask = torch.ones(num_images_per_prompt,3,height,width).cuda() 
                h_seams_mask = torch.ones(num_images_per_prompt,3,height,width).cuda()
                w_seams_mask = torch.ones(num_images_per_prompt,3,height,width).cuda()
                c_seams_mask = torch.ones(num_images_per_prompt,3,height,width).cuda()
                #work through tiles, mask seams and reassemble
                count = 0
                views = get_views(height, width, window_size=vae_image_window, stride=image_offset)
                for h_start, h_end, w_start, w_end in views:
                    if h_start % vae_image_window == 0 and w_start % vae_image_window == 0:
                        canvas[:, :, h_start:h_end, w_start:w_end] += image_parts[count]
                    elif h_start % image_offset == 0 and w_start % vae_image_window == 0:
                        h_seams[:, :, h_start:h_end, w_start:w_end] += image_parts[count]
                        w_seams_mask[:, :, h_start:h_end, w_start:w_end] *= h_alpha_centre
                        h_seams_mask[:, :, h_start:h_end, w_start:w_end] *= h_alpha_edge
                        canvas_mask[:, :, h_start:h_end, w_start:w_end] *= h_alpha_centre
                    elif h_start % vae_image_window == 0 and w_start % image_offset == 0:
                        w_seams[:, :, h_start:h_end, w_start:w_end] += image_parts[count]
                        h_seams_mask[:, :, h_start:h_end, w_start:w_end] *= w_alpha_centre 
                        w_seams_mask[:, :, h_start:h_end, w_start:w_end] *= w_alpha_edge
                        canvas_mask[:, :, h_start:h_end, w_start:w_end] *= w_alpha_centre
                    else:
                        c_seams[:, :, h_start:h_end, w_start:w_end] += image_parts[count]
                        c_seams_mask[:, :, h_start:h_end, w_start:w_end] *= h_alpha_edge
                        c_seams_mask[:, :, h_start:h_end, w_start:w_end] *= w_alpha_edge        
                    count+= 1

                recombined_image = canvas*canvas_mask+c_seams*c_seams_mask+h_seams*h_seams_mask+ w_seams*w_seams_mask
                image = (recombined_image / 2 + 0.5).clamp(0, 1)
                # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            else:
                image = []
                for images in image_parts:
                    image.append((images / 2 + 0.5).clamp(0, 1))
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        ### End VAE Chop and Reassemble ###
        
        else:
            image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
