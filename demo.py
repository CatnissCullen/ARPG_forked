import os

from diffusers import DiffusionPipeline
import torch
import sys
sys.path.append(".")

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pipeline = DiffusionPipeline.from_pretrained("./hfg_weights/", custom_pipeline="./hfg_weights/",
                                             cache_dir="./hfg_weights/",
                                             force_download=False, resume_download=False,
                                             local_files_only=True)

class_labels = [207, 360, 388, 113, 355, 980, 323, 979]

generated_image = pipeline(
    model_type="ARPG-XL",       # choose from 'ARPG-L', 'ARPG-XL', or 'ARPG-XXL'
    seed=0,                     # set a seed for reproducibility
    num_steps=64,               # number of autoregressive steps
    class_labels=class_labels,  # provide valid ImageNet class labels
    cfg_scale=4,                # classifier-free guidance scale
    output_dir="./images",      # directory to save generated images
    cfg_schedule="constant",    # choose between 'constant' (suggested) and 'linear'
    sample_schedule="arccos",   # choose between 'arccos' (suggested) and 'cosine'
)

generated_image.show()