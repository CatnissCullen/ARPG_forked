"""
ARPG with Reinforcement Learning Policy Optimization

This module implements:
1. Policy network for deciding parallel degree and position selection
2. Automatic reward calculator (no human annotation needed)
3. GRPO training loop
4. Modified ARPG generate function with policy guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from einops import rearrange


# ============================================================================
# 1. State Representation
# ============================================================================

@dataclass
class GenerationState:
    """Represents the current state during generation"""
    generated_mask: torch.Tensor  # [batch, seq_len], True if generated
    remaining_tokens: int
    step_count: int
    max_steps: int
    current_features: Optional[torch.Tensor] = None  # [batch, generated_len, dim]
    
    def to_vector(self, seq_len: int) -> torch.Tensor:
        """Convert state to a vector for policy network input"""
        batch_size = self.generated_mask.shape[0]
        
        # Basic statistics
        progress = self.step_count / self.max_steps
        coverage = self.generated_mask.float().mean(dim=-1)  # [batch]
        remaining_ratio = self.remaining_tokens / seq_len
        
        # Spatial features: divide sequence into regions and compute coverage
        num_regions = 8
        region_size = seq_len // num_regions
        region_coverage = []
        for i in range(num_regions):
            start = i * region_size
            end = start + region_size
            region_cov = self.generated_mask[:, start:end].float().mean(dim=-1)
            region_coverage.append(region_cov)
        region_coverage = torch.stack(region_coverage, dim=-1)  # [batch, num_regions]
        
        # Combine all features
        state_vec = torch.cat([
            coverage.unsqueeze(-1),  # [batch, 1]
            torch.tensor([progress], device=self.generated_mask.device).expand(batch_size, 1),
            torch.tensor([remaining_ratio], device=self.generated_mask.device).expand(batch_size, 1),
            region_coverage,  # [batch, num_regions]
        ], dim=-1)  # [batch, 1+1+1+num_regions]
        
        # Add feature statistics if available
        if self.current_features is not None:
            feat_mean = self.current_features.mean(dim=1)  # [batch, dim]
            feat_std = self.current_features.std(dim=1)  # [batch, dim]
            state_vec = torch.cat([state_vec, feat_mean, feat_std], dim=-1)
        
        return state_vec


# ============================================================================
# 2. Policy Network
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    Policy network that decides:
    1. How many tokens to generate (parallel degree)
    2. Which positions to generate (position selection)
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        max_parallel: int = 32,
        seq_len: int = 256,
    ):
        super().__init__()
        self.max_parallel = max_parallel
        self.seq_len = seq_len
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Parallel degree policy head
        self.parallel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_parallel)
        )
        
        # Position importance scorer
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len)
        )
        
    def forward(
        self, 
        state: torch.Tensor,  # [batch, state_dim]
        generated_mask: torch.Tensor,  # [batch, seq_len]
    ) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        """
        Returns:
            parallel_dist: Categorical distribution over parallel degrees
            position_scores: Scores for each position [batch, seq_len]
        """
        # Encode state
        features = self.encoder(state)  # [batch, hidden_dim]
        
        # Parallel degree distribution
        parallel_logits = self.parallel_head(features)  # [batch, max_parallel]
        
        # Mask out invalid parallel degrees (can't generate more than remaining)
        remaining = (~generated_mask).sum(dim=-1)  # [batch]
        for b in range(state.shape[0]):
            if remaining[b] < self.max_parallel:
                parallel_logits[b, remaining[b]:] = float('-inf')
        
        parallel_dist = torch.distributions.Categorical(logits=parallel_logits)
        
        # Position scores
        position_scores = self.position_head(features)  # [batch, seq_len]
        
        # Mask out already generated positions
        position_scores = position_scores.masked_fill(generated_mask, float('-inf'))
        
        return parallel_dist, position_scores
    
    def sample_positions(
        self,
        position_scores: torch.Tensor,  # [batch, seq_len]
        num_tokens: torch.Tensor,  # [batch]
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample positions to generate
        
        Returns:
            positions: [batch, max_num_tokens], padded with -1
            log_probs: [batch], log probability of the sampled positions
        """
        batch_size = position_scores.shape[0]
        max_num_tokens = num_tokens.max().item()
        
        positions = torch.full(
            (batch_size, max_num_tokens), 
            -1, 
            dtype=torch.long,
            device=position_scores.device
        )
        log_probs = torch.zeros(batch_size, device=position_scores.device)
        
        for b in range(batch_size):
            n = num_tokens[b].item()
            scores = position_scores[b]
            
            if training:
                # Sample with Gumbel-Softmax for exploration
                probs = F.softmax(scores, dim=-1)
                sampled_pos = torch.multinomial(probs, n, replacement=False)
                
                # Compute log prob
                selected_probs = probs[sampled_pos]
                log_probs[b] = torch.log(selected_probs + 1e-10).sum()
            else:
                # Greedy: select top-k
                sampled_pos = torch.topk(scores, n, dim=-1).indices
                
                probs = F.softmax(scores, dim=-1)
                selected_probs = probs[sampled_pos]
                log_probs[b] = torch.log(selected_probs + 1e-10).sum()
            
            positions[b, :n] = sampled_pos
        
        return positions, log_probs


# ============================================================================
# 3. Automatic Reward Calculator
# ============================================================================

class AutomaticRewardCalculator:
    """
    Computes rewards automatically without human annotation
    Uses pre-trained models: CLIP, aesthetic predictor, etc.
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.clip_model = None
        self.clip_preprocess = None
        self._init_clip()
    
    def _init_clip(self):
        """Initialize CLIP model (lazy loading)"""
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load(
                "ViT-B/32", 
                device=self.device
            )
            self.clip_model.eval()
            print("CLIP model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load CLIP model: {e}")
            print("Reward calculation will use simplified metrics")
    
    def compute_reward(
        self,
        image: torch.Tensor,  # [C, H, W] or [B, C, H, W]
        prompt: Optional[str] = None,
        num_steps: int = 1,
        target_steps: int = 10,
    ) -> float:
        """
        Compute automatic reward
        
        Components:
        1. CLIP score (if prompt provided)
        2. Image quality (sharpness, contrast, saturation)
        3. Efficiency (fewer steps is better)
        """
        with torch.no_grad():
            rewards = {}
            
            # Ensure batch dimension
            if image.ndim == 3:
                image = image.unsqueeze(0)
            
            # 1. CLIP Score
            if prompt is not None and self.clip_model is not None:
                try:
                    clip_score = self._compute_clip_score(image, prompt)
                    rewards['clip'] = clip_score
                except Exception as e:
                    print(f"CLIP score computation failed: {e}")
                    rewards['clip'] = 0.5  # neutral score
            else:
                rewards['clip'] = 0.5
            
            # 2. Image Quality
            quality = self._compute_image_quality(image)
            rewards['quality'] = quality
            
            # 3. Efficiency
            efficiency = max(0, (target_steps - num_steps) / target_steps)
            rewards['efficiency'] = efficiency
            
            # Weighted combination
            total_reward = (
                1.0 * rewards['clip'] +
                0.5 * rewards['quality'] +
                0.3 * rewards['efficiency']
            )
            
            return total_reward
    
    def _compute_clip_score(self, image: torch.Tensor, prompt: str) -> float:
        """Compute CLIP similarity between image and text"""
        import clip
        
        # Preprocess image
        if image.shape[-1] != 224:  # CLIP expects 224x224
            image = F.interpolate(image, size=(224, 224), mode='bilinear')
        
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0
        
        # Encode image and text
        image_features = self.clip_model.encode_image(image)
        text_tokens = clip.tokenize([prompt]).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        similarity = (image_features @ text_features.T).mean().item()
        
        # Scale to [0, 1]
        similarity = (similarity + 1) / 2
        
        return similarity
    
    def _compute_image_quality(self, image: torch.Tensor) -> float:
        """
        Compute image quality based on statistical properties
        No training required
        """
        # Ensure image is in [0, 1] range
        if image.max() > 1.0:
            image = image / 255.0
        
        # 1. Sharpness (Laplacian variance)
        sharpness = self._compute_sharpness(image)
        
        # 2. Contrast
        contrast = self._compute_contrast(image)
        
        # 3. Saturation
        saturation = self._compute_saturation(image)
        
        # Combine
        quality = (sharpness + contrast + saturation) / 3
        
        return min(quality, 1.0)
    
    def _compute_sharpness(self, image: torch.Tensor) -> float:
        """Compute sharpness using Laplacian variance"""
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
        
        # Convert to grayscale
        if image.shape[1] == 3:
            gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        else:
            gray = image
        
        # Apply Laplacian
        laplacian = F.conv2d(gray, laplacian_kernel, padding=1)
        variance = laplacian.var().item()
        
        # Normalize
        sharpness = min(variance * 100, 1.0)
        
        return sharpness
    
    def _compute_contrast(self, image: torch.Tensor) -> float:
        """Compute contrast"""
        contrast = (image.max() - image.min()).item()
        return contrast
    
    def _compute_saturation(self, image: torch.Tensor) -> float:
        """Compute color saturation"""
        if image.shape[1] == 3:
            saturation = image.std(dim=1).mean().item()
        else:
            saturation = 0.5  # grayscale
        
        return min(saturation * 2, 1.0)


# ============================================================================
# 4. ARPG with Policy Integration
# ============================================================================

class ARPGWithPolicy:
    """
    Wrapper around ARPG model that integrates policy network
    """
    def __init__(
        self,
        arpg_model,
        vae_model,
        policy_net: PolicyNetwork,
        reward_calculator: AutomaticRewardCalculator,
    ):
        self.arpg = arpg_model
        self.vqvae = vae_model
        self.policy = policy_net
        self.reward_calc = reward_calculator
    
    def generate_with_policy(
        self,
        cond_idx: torch.Tensor,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
        max_steps: int = 20,
        prompt: Optional[str] = None,
    ) -> Dict:
        """
        Generate images using policy network guidance
        
        Returns:
            Dictionary containing:
            - image: generated image
            - trajectory: list of (state, action, reward) tuples
            - num_steps: total steps taken
            - final_reward: reward of the final image
        """
        batch_size = cond_idx.shape[0]
        seq_len = self.arpg.block_size
        device = cond_idx.device
        
        # Initialize generation state
        generated_tokens = torch.full(
            (batch_size, seq_len), 
            self.arpg.vocab_size,  # use vocab_size as padding
            dtype=torch.long,
            device=device
        )
        generated_mask = torch.zeros(
            batch_size, seq_len, 
            dtype=torch.bool, 
            device=device
        )
        
        state = GenerationState(
            generated_mask=generated_mask,
            remaining_tokens=seq_len,
            step_count=0,
            max_steps=max_steps,
            current_features=None,
        )
        
        trajectory = []
        
        # Generation loop
        for step in range(max_steps):
            # 1. Get state representation
            state_vec = state.to_vector(seq_len)
            
            # 2. Policy decides parallel degree
            parallel_dist, position_scores = self.policy(state_vec, generated_mask)
            num_tokens = parallel_dist.sample()  # [batch]
            parallel_logprob = parallel_dist.log_prob(num_tokens)
            
            # 3. Policy selects positions
            positions, position_logprob = self.policy.sample_positions(
                position_scores,
                num_tokens,
                training=self.policy.training
            )
            
            # 4. Generate tokens at selected positions
            # (This is a simplified version - you need to adapt to actual ARPG API)
            new_tokens = self._generate_at_positions(
                cond_idx,
                generated_tokens,
                positions,
                num_tokens,
                temperature,
                cfg_scale,
            )
            
            # 5. Update generated tokens and mask
            for b in range(batch_size):
                n = num_tokens[b].item()
                valid_positions = positions[b, :n]
                valid_positions = valid_positions[valid_positions >= 0]
                
                generated_tokens[b, valid_positions] = new_tokens[b, valid_positions]
                generated_mask[b, valid_positions] = True
            
            # 6. Update state
            state.step_count = step + 1
            state.remaining_tokens = (~generated_mask).sum().item()
            
            # 7. Record trajectory
            trajectory.append({
                'state': state_vec.detach(),
                'num_tokens': num_tokens.detach(),
                'positions': positions.detach(),
                'parallel_logprob': parallel_logprob.detach(),
                'position_logprob': position_logprob.detach(),
                'generated_mask': generated_mask.clone(),
            })
            
            # 8. Check if done
            if generated_mask.all():
                break
        
        # Decode tokens to image
        # (Adapt this to your actual ARPG decode function)
        final_image = self._decode_tokens(generated_tokens)
        
        # Compute final reward
        final_reward = self.reward_calc.compute_reward(
            final_image[0] if batch_size == 1 else final_image,
            prompt=prompt,
            num_steps=step + 1,
        )
        
        return {
            'image': final_image,
            'trajectory': trajectory,
            'num_steps': step + 1,
            'final_reward': final_reward,
            'tokens': generated_tokens,
        }

    def _generate_at_positions_greedy(
            self,
            cond_idx: torch.Tensor,
            current_tokens: torch.Tensor,
            positions: torch.Tensor,
            num_tokens: torch.Tensor,
            cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate tokens using greedy decoding (argmax)
        Useful for evaluation/inference
        """
        batch_size, seq_len = current_tokens.shape

        new_tokens = current_tokens.clone()
        input_tokens = current_tokens.clone()

        # Mark positions to generate
        mask_token_id = self.arpg.vocab_size
        for b in range(batch_size):
            n = num_tokens[b].item()
            valid_positions = positions[b, :n]
            valid_positions = valid_positions[valid_positions >= 0]
            input_tokens[b, valid_positions] = mask_token_id

        with torch.no_grad():
            # CFG
            if cfg_scale > 1.0:
                input_tokens_cfg = torch.cat([input_tokens, input_tokens], dim=0)
                cond_idx_cfg = torch.cat([
                    cond_idx,
                    torch.full_like(cond_idx, self.arpg.num_classes)
                ], dim=0)
            else:
                input_tokens_cfg = input_tokens
                cond_idx_cfg = cond_idx

            # Forward
            logits = self.arpg(input_tokens_cfg, cond_idx_cfg)

            # CFG
            if cfg_scale > 1.0:
                logits_cond, logits_uncond = logits.chunk(2, dim=0)
                logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)

            # Greedy sampling
            for b in range(batch_size):
                n = num_tokens[b].item()
                valid_positions = positions[b, :n]
                valid_positions = valid_positions[valid_positions >= 0]

                if len(valid_positions) == 0:
                    continue

                position_logits = logits[b, valid_positions, :self.arpg.vocab_size]
                sampled_tokens = position_logits.argmax(dim=-1)

                new_tokens[b, valid_positions] = sampled_tokens

        return new_tokens

    # ============================================================================
    # Helper: Decode tokens to image
    # ============================================================================

    def _decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode tokens to image using VQ-VAE decoder

        Args:
            tokens: [batch, seq_len] token indices

        Returns:
            images: [batch, 3, H, W] decoded images
        """
        # This requires the VQ-VAE model used to train ARPG
        # You need to load the corresponding VQ-VAE checkpoint

        # Assuming you have vqvae model loaded as self.vqvae
        if not hasattr(self, 'vqvae'):
            raise RuntimeError(
                "VQ-VAE model not loaded. Please load the VQ-VAE model first:\n"
                "  self.vqvae = load_vqvae_model(checkpoint_path)\n"
                "  self.vqvae.eval()"
            )

        with torch.no_grad():
            # Reshape tokens to 2D grid (assuming square layout)
            batch_size, seq_len = tokens.shape
            grid_size = int(seq_len ** 0.5)
            assert grid_size * grid_size == seq_len, f"seq_len {seq_len} must be a perfect square"

            # Reshape to [batch, h, w]
            token_grid = tokens.view(batch_size, grid_size, grid_size)

            # Decode using VQ-VAE
            # This depends on your VQ-VAE implementation
            # Common interface: vqvae.decode_code(token_grid)
            images = self.vqvae.decode_code(token_grid)

            # Ensure output is in [0, 1] range
            if images.min() < 0:
                images = (images + 1) / 2  # from [-1, 1] to [0, 1]

            return images


# ============================================================================
# 5. GRPO Training Loop
# ============================================================================

def train_policy_grpo(
    arpg_with_policy: ARPGWithPolicy,
    dataloader,
    num_epochs: int = 10,
    group_size: int = 4,
    lr: float = 1e-4,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    device: str = 'cuda',
):
    """
    Train policy network using Group Relative Policy Optimization (GRPO)
    
    Args:
        arpg_with_policy: ARPG model with policy network
        dataloader: DataLoader providing (condition, prompt) pairs
        num_epochs: Number of training epochs
        group_size: Number of samples per group for GRPO
        lr: Learning rate
        clip_epsilon: PPO clipping parameter
        entropy_coef: Entropy regularization coefficient
    """
    optimizer = torch.optim.Adam(arpg_with_policy.policy.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        epoch_rewards = []
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            cond_idx = batch['condition'].to(device)
            prompts = batch.get('prompt', [None] * cond_idx.shape[0])
            
            # Generate multiple candidates per sample (GRPO group)
            group_outputs = []
            
            for group_idx in range(group_size):
                output = arpg_with_policy.generate_with_policy(
                    cond_idx=cond_idx,
                    temperature=1.0,
                    cfg_scale=1.0,
                    prompt=prompts[0] if len(prompts) > 0 else None,
                )
                group_outputs.append(output)
            
            # Compute group rewards and advantages
            rewards = [o['final_reward'] for o in group_outputs]
            mean_reward = np.mean(rewards)
            advantages = [r - mean_reward for r in rewards]
            
            # Update policy using GRPO
            loss = update_policy_step(
                arpg_with_policy.policy,
                group_outputs,
                advantages,
                optimizer,
                clip_epsilon,
                entropy_coef,
            )
            
            epoch_rewards.extend(rewards)
            epoch_losses.append(loss)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"Mean Reward = {mean_reward:.3f}, "
                      f"Loss = {loss:.3f}")
        
        avg_reward = np.mean(epoch_rewards)
        avg_loss = np.mean(epoch_losses)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Average Reward: {avg_reward:.3f}")
        print(f"  Average Loss: {avg_loss:.3f}")
        print(f"  Reward Std: {np.std(epoch_rewards):.3f}\n")


def update_policy_step(
    policy_net: PolicyNetwork,
    trajectories: List[Dict],
    advantages: List[float],
    optimizer: torch.optim.Optimizer,
    clip_epsilon: float,
    entropy_coef: float,
) -> float:
    """
    Single policy update step using GRPO
    
    Returns:
        loss value
    """
    total_loss = 0
    total_entropy = 0
    num_steps = 0
    
    for traj_data, advantage in zip(trajectories, advantages):
        trajectory = traj_data['trajectory']
        
        for step_data in trajectory:
            # Get old log probs
            old_parallel_logprob = step_data['parallel_logprob']
            old_position_logprob = step_data['position_logprob']
            old_logprob = old_parallel_logprob + old_position_logprob
            
            # Recompute log probs with current policy
            state = step_data['state']
            num_tokens = step_data['num_tokens']
            positions = step_data['positions']
            generated_mask = step_data['generated_mask']
            
            parallel_dist, position_scores = policy_net(state, generated_mask)
            
            new_parallel_logprob = parallel_dist.log_prob(num_tokens)
            
            # Recompute position log prob
            new_position_logprob = torch.zeros_like(new_parallel_logprob)
            for b in range(state.shape[0]):
                n = num_tokens[b].item()
                valid_positions = positions[b, :n]
                valid_positions = valid_positions[valid_positions >= 0]
                
                if len(valid_positions) > 0:
                    probs = F.softmax(position_scores[b], dim=-1)
                    selected_probs = probs[valid_positions]
                    new_position_logprob[b] = torch.log(selected_probs + 1e-10).sum()
            
            new_logprob = new_parallel_logprob + new_position_logprob
            
            # Compute importance ratio
            ratio = torch.exp(new_logprob - old_logprob)
            
            # Clipped surrogate objective
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy bonus
            entropy = parallel_dist.entropy().mean()
            
            total_loss += policy_loss
            total_entropy += entropy
            num_steps += 1
    
    # Average over all steps
    avg_loss = total_loss / num_steps
    avg_entropy = total_entropy / num_steps
    
    # Total loss with entropy regularization
    loss = avg_loss - entropy_coef * avg_entropy
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()


# ============================================================================
# 6. Example Usage
# ============================================================================

if __name__ == "__main__":
    # This is a minimal example showing how to use the framework
    
    # 1. Load ARPG model (you need to implement this)
    # arpg_model = load_arpg_model()
    
    # 2. Create policy network
    state_feat_chan = 11 + 8  # basic stats + region coverage (without features)
    policy = PolicyNetwork(
        state_dim=state_feat_chan,
        hidden_dim=256,
        max_parallel=32,
        seq_len=256,
    )
    
    # 3. Create reward calculator
    reward_calc = AutomaticRewardCalculator(device='cuda')
    
    # 4. Create integrated model
    # arpg_with_policy = ARPGWithPolicy(arpg_model, policy, reward_calc)
    
    # 5. Prepare dataloader (you need to implement this)
    # dataloader = create_dataloader()
    
    # 6. Train
    # train_policy_grpo(
    #     arpg_with_policy,
    #     dataloader,
    #     num_epochs=10,
    #     group_size=4,
    #     lr=1e-4,
    # )
    
    print("ARPG RL Training Framework initialized successfully!")
    print("Please integrate with your ARPG model and dataloader to start training.")