"""
Training script for ARPG with RL Policy Optimization

Usage:
    python train_arpg_rl.py --config configs/arpg_rl.yaml
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse
import yaml
import os
from pathlib import Path

# Import from the RL framework
from arpg_rl import (
    PolicyNetwork,
    AutomaticRewardCalculator,
    ARPGWithPolicy,
    train_policy_grpo,
)

# Import ARPG model (you need to adapt this)
# from arpg import Transformer as ARPGModel


# ============================================================================
# Dataset
# ============================================================================

class ImageNetDataset(Dataset):
    """
    Simple ImageNet dataset for class-conditional generation
    
    You should replace this with your actual dataset
    """
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Load class labels
        self.samples = self._load_samples()
    
    def _load_samples(self):
        """Load dataset samples"""
        # TODO: Implement actual data loading
        # For now, return dummy data
        return [
            {'class_idx': i % 1000, 'prompt': f'class_{i % 1000}'}
            for i in range(1000)
        ]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        return {
            'condition': torch.tensor([sample['class_idx']], dtype=torch.long),
            'prompt': sample['prompt'],
        }


# ============================================================================
# Model Loading
# ============================================================================

def load_arpg_model(config):
    """
    Load pre-trained ARPG model
    
    You need to implement this based on your ARPG checkpoint
    """
    # DONE-TODO: Load actual ARPG model
    from hfg_weights.arpg.arpg import Transformer, ModelArgs
    model_args = ModelArgs(**config['model'])
    model = Transformer(model_args)
    checkpoint = torch.load(config['arpg_checkpoint'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


# ============================================================================
# Main Training Function
# ============================================================================

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("ARPG with RL Policy Optimization - Training")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # ========================================================================
    # 1. Load ARPG Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("Loading ARPG Model...")
    print("=" * 80)
    
    arpg_model = load_arpg_model(config)
    arpg_model.to(device)
    arpg_model.eval()  # ARPG stays frozen, only policy is trained
    
    print("ARPG model loaded successfully")
    
    # ========================================================================
    # 2. Create Policy Network
    # ========================================================================
    print("\n" + "=" * 80)
    print("Creating Policy Network...")
    print("=" * 80)
    
    policy_config = config['policy']
    policy_net = PolicyNetwork(
        state_dim=policy_config['state_dim'],
        hidden_dim=policy_config['hidden_dim'],
        max_parallel=policy_config['max_parallel'],
        seq_len=arpg_model.block_size,
    )
    policy_net.to(device)
    policy_net.train()
    
    num_params = sum(p.numel() for p in policy_net.parameters())
    print(f"Policy network created with {num_params:,} parameters")
    
    # ========================================================================
    # 3. Create Reward Calculator
    # ========================================================================
    print("\n" + "=" * 80)
    print("Creating Reward Calculator...")
    print("=" * 80)
    
    reward_calc = AutomaticRewardCalculator(device=device)
    print("Reward calculator initialized")
    
    # ========================================================================
    # 4. Create Integrated Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("Creating Integrated Model...")
    print("=" * 80)
    
    arpg_with_policy = ARPGWithPolicy(
        arpg_model=arpg_model,
        policy_net=policy_net,
        reward_calculator=reward_calc,
    )
    print("Integrated model created")
    
    # ========================================================================
    # 5. Create Dataset and DataLoader
    # ========================================================================
    print("\n" + "=" * 80)
    print("Loading Dataset...")
    print("=" * 80)
    
    dataset = ImageNetDataset(
        data_dir=config['data_dir'],
        split='train',
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"DataLoader created: {len(dataloader)} batches")
    
    # ========================================================================
    # 6. Training
    # ========================================================================
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)
    print()
    
    train_config = config['training']
    
    train_policy_grpo(
        arpg_with_policy=arpg_with_policy,
        dataloader=dataloader,
        num_epochs=train_config['num_epochs'],
        group_size=train_config['group_size'],
        lr=train_config['learning_rate'],
        clip_epsilon=train_config['clip_epsilon'],
        entropy_coef=train_config['entropy_coef'],
        device=device,
    )
    
    # ========================================================================
    # 7. Save Policy Network
    # ========================================================================
    print("\n" + "=" * 80)
    print("Saving Policy Network...")
    print("=" * 80)
    
    checkpoint_path = output_dir / 'policy_final.pt'
    torch.save({
        'policy_state_dict': policy_net.state_dict(),
        'config': config,
    }, checkpoint_path)
    
    print(f"Policy network saved to: {checkpoint_path}")
    
    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train ARPG with RL Policy Optimization'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/arpg_rl.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    main(args)