---
license: mit
pipeline_tag: unconditional-image-generation
library_name: diffusers
---

# Autoregressive Image Generation with Randomized Parallel Decoding

[Haopeng Li](https://github.com/hp-l33)<sup>1</sup>, Jinyue Yang<sup>2</sup>, [Guoqi Li](https://casialiguoqi.github.io)<sup>2,📧</sup>, [Huan Wang](https://huanwang.tech)<sup>1,📧</sup>

<sup>1</sup> Westlake University,
<sup>2</sup> Institute of Automation, Chinese Academy of Sciences

## TL;DR
**ARPG** is a novel autoregressive image generation framework capable of performing **BERT-style masked modeling** with a **GPT-style causal architecture**.

``💪 FID 1.94`` ``🚀 Fast Speed`` ``♻️ Low Memory Usage`` ``🎲 Radnom Order`` ``💡 Zero-shot Inference``

## Usage:
You can easily load it through the Hugging Face DiffusionPipeline and optionally customize various parameters such as the model type, number of steps, and class labels.
```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("hp-l33/ARPG", custom_pipeline="hp-l33/ARPG")

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
```

## Citation
If this work is helpful for your research, please give it a star or cite it:
```bibtex
@article{li2025autoregressive,
    title={Autoregressive Image Generation with Randomized Parallel Decoding},
    author={Haopeng Li and Jinyue Yang and Guoqi Li and Huan Wang},
    journal={arXiv preprint arXiv:2503.10568},
    year={2025}
}
```

## Acknowledgement

Thanks to [LlamaGen](https://github.com/FoundationVision/LlamaGen) for its open-source codebase. Appreciate [RandAR](https://github.com/ziqipang/RandAR) and [RAR](https://github.com/bytedance/1d-tokenizer/blob/main/README_RAR.md) for inspiring this work, and also thank [ControlAR](https://github.com/hustvl/ControlAR).