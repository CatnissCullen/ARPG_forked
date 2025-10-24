# ARPG with RL Policy Optimization

This framework implements reinforcement learning-based policy optimization for ARPG (Autoregressive Image Generation with Randomized Parallel Decoding).

## Overview

The key idea is to use a policy network to optimize:
1. **Parallel Degree**: How many tokens to generate at each step
2. **Position Selection**: Which positions to generate

This is done through **Group Relative Policy Optimization (GRPO)**, which doesn't require human-annotated preference data.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ARPG Model (Frozen)                  │
│                                                         │
│  ┌──────────────┐                                      │
│  │ Token Embed  │                                      │
│  └──────┬───────┘                                      │
│         │                                              │
│  ┌──────▼────────┐                                     │
│  │ Transformer   │                                     │
│  │ Blocks        │                                     │
│  └──────┬────────┘                                     │
│         │                                              │
│  ┌──────▼────────┐                                     │
│  │ Token Decoder │                                     │
│  └───────────────┘                                     │
└─────────────────────────────────────────────────────────┘
                        │
                        │ Features
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Policy Network (Trainable)                 │
│                                                         │
│  ┌──────────────┐                                      │
│  │ State        │                                      │
│  │ Encoder      │                                      │
│  └──────┬───────┘                                      │
│         │                                              │
│    ┌────┴────┐                                         │
│    │         │                                         │
│  ┌─▼──────┐ ┌▼─────────┐                              │
│  │Parallel│ │Position  │                              │
│  │Degree  │ │Selection │                              │
│  │Head    │ │Head      │                              │
│  └────────┘ └──────────┘                              │
└─────────────────────────────────────────────────────────┘
                        │
                        │ Actions
                        ▼
┌─────────────────────────────────────────────────────────┐
│           Automatic Reward Calculator                   │
│                                                         │
│  • CLIP Score (text-image alignment)                   │
│  • Image Quality (sharpness, contrast, saturation)     │
│  • Efficiency (fewer steps is better)                  │
└─────────────────────────────────────────────────────────┘
```

## File Structure

```
arpg_rl/
├── arpg_rl.py    # Core RL framework
├── train_rl.py        # Training script
├── configs/
│   └── arpg_rl.yaml        # Configuration file
├── arpg.py                 # Original ARPG model
└── README.md               # This file
```

## Installation

```bash
# Install dependencies
pip install torch torchvision
pip install einops
pip install transformers
pip install clip  # For CLIP-based rewards
pip install pyyaml

# Optional: for better image quality metrics
pip install lpips
pip install pytorch-fid
```

## Quick Start

### 1. Prepare Configuration

Edit `configs/arpg_rl.yaml`:

```yaml
# Set your data path
data_dir: /path/to/your/imagenet

# Set your ARPG checkpoint path
arpg_checkpoint: /path/to/arpg_checkpoint.pt

# Adjust training parameters
training:
  num_epochs: 10
  batch_size: 1
  group_size: 4
```

### 2. Integrate with Your ARPG Model

In `train_arpg_rl.py`, modify the `load_arpg_model` function:

```python
def load_arpg_model(config):
    from arpg import Transformer, ModelArgs
    
    # Load model config
    model_args = ModelArgs(**config['model'])
    
    # Create model
    model = Transformer(model_args)
    
    # Load checkpoint
    checkpoint = torch.load(config['arpg_checkpoint'])
    model.load_state_dict(checkpoint['model'])
    
    model.eval()
    return model
```

### 3. Implement Token Generation

In `arpg_rl_framework.py`, implement `_generate_at_positions`:

```python
def _generate_at_positions(self, cond_idx, current_tokens, positions, ...):
    # Use ARPG's actual generation API
    # This depends on your ARPG implementation
    
    # Example (pseudo-code):
    logits = self.arpg.forward(current_tokens, cond_idx)
    
    # Sample tokens at specified positions
    for b in range(batch_size):
        for pos in positions[b]:
            if pos >= 0:
                token = sample_from_logits(logits[b, pos], temperature)
                current_tokens[b, pos] = token
    
    return current_tokens
```

### 4. Start Training

```bash
python train_arpg_rl.py --config configs/arpg_rl.yaml --device cuda
```

## Training Process

The training follows these steps:

1. **Sample Generation**: For each training sample, generate `group_size` (e.g., 4) candidates using the current policy

2. **Reward Calculation**: Compute automatic rewards for each candidate:
   - CLIP score (if text prompt available)
   - Image quality metrics
   - Efficiency (number of steps)

3. **Group Normalization**: Compute advantages within each group:
   ```
   advantage = reward - mean(group_rewards)
   ```

4. **Policy Update**: Update policy using PPO-style objective:
   ```
   ratio = exp(new_log_prob - old_log_prob)
   loss = -min(ratio * advantage, clip(ratio) * advantage)
   ```

5. **Repeat**: Continue for multiple epochs

## Key Features

### ✅ No Human Annotation Required

- Uses pre-trained CLIP for text-image alignment
- Uses statistical metrics for image quality
- Completely automatic reward calculation

### ✅ GRPO: Stable Training

- Group-relative advantages reduce variance
- No need for value function
- More stable than vanilla policy gradient

### ✅ Flexible Reward Design

- Easy to add new reward components
- Adjustable weights for different objectives
- Can incorporate domain knowledge

## Customization

### Adding New Reward Components

In `AutomaticRewardCalculator.compute_reward`:

```python
def compute_reward(self, image, prompt, num_steps):
    rewards = {}
    
    # Existing rewards
    rewards['clip'] = self._compute_clip_score(image, prompt)
    rewards['quality'] = self._compute_image_quality(image)
    rewards['efficiency'] = ...
    
    # Add your custom reward
    rewards['custom'] = self._compute_custom_reward(image)
    
    # Weighted combination
    total = (
        1.0 * rewards['clip'] +
        0.5 * rewards['quality'] +
        0.3 * rewards['efficiency'] +
        0.2 * rewards['custom']  # Your weight
    )
    
    return total
```

### Adjusting Policy Network

In `PolicyNetwork.__init__`:

```python
# Increase capacity
self.encoder = nn.Sequential(
    nn.Linear(state_dim, 512),  # Larger hidden dim
    nn.ReLU(),
    nn.LayerNorm(512),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.LayerNorm(512),
)
```

### Changing Generation Strategy

In config file:

```yaml
policy:
  max_parallel: 64  # Allow more aggressive parallelism

training:
  target_steps: 5  # Encourage fewer steps
```

## Monitoring Training

The training script prints:

```
Epoch 0, Batch 0: Mean Reward = 0.723, Loss = 0.145
Epoch 0, Batch 10: Mean Reward = 0.756, Loss = 0.132
...

Epoch 0 Summary:
  Average Reward: 0.745
  Average Loss: 0.138
  Reward Std: 0.089
```

Key metrics to watch:
- **Mean Reward**: Should increase over time
- **Reward Std**: Should decrease (more consistent)
- **Loss**: Should decrease initially, then stabilize

## Troubleshooting

### Issue: CLIP model fails to load

```python
# Install CLIP
pip install git+https://github.com/openai/CLIP.git
```

### Issue: Out of memory

```yaml
# Reduce batch size and group size
training:
  batch_size: 1
  group_size: 2
```

### Issue: Policy not improving

```yaml
# Increase learning rate
training:
  learning_rate: 5.0e-4
  
# Increase entropy coefficient (more exploration)
training:
  entropy_coef: 0.05
```

### Issue: Rewards are too similar (low variance)

```yaml
# Increase temperature for more diversity
training:
  temperature: 1.5
```

## Next Steps

1. **Phase 1**: Get the basic training loop working
   - Start with small dataset (100 samples)
   - Verify rewards are being computed correctly
   - Check that policy is updating

2. **Phase 2**: Optimize hyperparameters
   - Try different learning rates
   - Adjust reward weights
   - Experiment with group sizes

3. **Phase 3**: Scale up
   - Train on full dataset
   - Increase policy network capacity
   - Add more sophisticated rewards

4. **Phase 4**: Evaluation
   - Compare with baseline ARPG
   - Measure FID, IS, CLIP scores
   - Analyze generation efficiency

## Citation

If you use this code, please cite:

```bibtex
@article{your_paper,
  title={Reinforcement Learning for Optimizing Autoregressive Image Generation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].