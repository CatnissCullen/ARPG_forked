from diffusers import DiffusionPipeline
import torch
import random
import numpy as np
import importlib.util
import sys
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import os
from torchvision.utils import save_image, make_grid
from PIL import Image
from safetensors.torch import load_file
from .vq_model import VQ_models
from .arpg import ARPG_models

# inheriting from DiffusionPipeline for HF
class ARPGModel(DiffusionPipeline):

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        """
        This method downloads the model and VAE components,
        then executes the forward pass based on the user's input.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # init the mar model architecture
        model_type = kwargs.get("model_type", "ARPG-XXL")

        # download the pretrained model and set diffloss parameters
        if model_type == "ARPG-L":
            model_path = "arpg_300m.pt"
        elif model_type == "ARPG-XL":
            model_path = "arpg_700m.pt"
        elif model_type == "ARPG-XXL":
            model_path = "arpg_1b.pt"
        else:
            raise NotImplementedError

        # download and load the model weights (.safetensors or .pth)
        # model_checkpoint_path = hf_hub_download(
        #     repo_id=kwargs.get("repo_id", "hp-l33/ARPG"),
        #     filename=kwargs.get("model_filename", model_path)
        # )
        cur_dir = os.getcwd()
        local_files_dir = os.path.join(cur_dir, "hfg_weights")
        model_filename = kwargs.get("model_filename", model_path)
        model_checkpoint_path = os.path.join(local_files_dir, model_filename)

        print(f"local model path: {model_checkpoint_path}")

        model_fn = ARPG_models[model_type]
        model = model_fn(
          num_classes=1000,
          vocab_size=16384
        ).cuda()

        state_dict = torch.load(model_checkpoint_path)['state_dict']
        model.load_state_dict(state_dict)
        model.eval()

        # download and load the vae
        vae_checkpoint_path = hf_hub_download(
            repo_id=kwargs.get("repo_id", "FoundationVision/LlamaGen"),
            filename=kwargs.get("vae_filename", "vq_ds16_c2i.pt")
        )

        vae = VQ_models['VQ-16']()

        vae_state_dict = torch.load(vae_checkpoint_path)['model']
        vae.load_state_dict(vae_state_dict)
        vae = vae.to(device).eval()

        # set up user-specified or default values for generation
        seed = kwargs.get("seed", 6)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        num_steps = kwargs.get("num_steps", 64)
        cfg_scale = kwargs.get("cfg_scale", 4)
        cfg_schedule = kwargs.get("cfg_schedule", "constant")
        sample_schedule = kwargs.get("sample_schedule", "arccos")
        temperature = kwargs.get("temperature", 1.0)
        top_k = kwargs.get("top_k", 600)
        class_labels = kwargs.get("class_labels", [207, 360, 388, 113, 355, 980, 323, 979])

        # generate the tokens and images
        with torch.cuda.amp.autocast():
            sampled_tokens = model.generate(
               condition=torch.Tensor(class_labels).long().cuda(),
               num_iter=num_steps,
               guidance_scale=cfg_scale,
               cfg_schedule=cfg_schedule,
               sample_schedule=sample_schedule,
               temperature=temperature,
               top_k=top_k,
            )
            sampled_images = vae.decode_code(sampled_tokens, shape=(len(class_labels), 8, 16, 16))

        output_dir = kwargs.get("output_dir", "../")
        os.makedirs(output_dir, exist_ok=True)
    
        # save the images
        image_path = os.path.join(output_dir, "sampled_image.png")
        samples_per_row = kwargs.get("samples_per_row", 4)
    
        ndarr = make_grid(
          torch.clamp(127.5 * sampled_images + 128.0, 0, 255),
          nrow=int(samples_per_row)
        ).permute(1, 2, 0).to("cpu", dtype=torch.uint8).numpy()

        Image.fromarray(ndarr).save(image_path)

        # return as a pil image
        image = Image.open(image_path)
    
        return image
