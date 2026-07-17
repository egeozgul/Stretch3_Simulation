#!/usr/bin/env python3
"""
Test script to verify the instruction concatenation fix
"""

import torch
import sys
sys.path.insert(0, '/home/willy/macro_marl_ppo/src')

from macro_marl.cores.pg_based.mac_cac.models import Actor, Critic

def test_concat_with_no_instruction():
    """Test that concat fusion works even when no instruction is provided"""
    print("="*60)
    print("Testing concat fusion with no instruction provided")
    print("="*60)
    
    obs_dim = 56  # From error message
    action_dim = 12  # Example
    rnn_layer_size = 32
    
    # Create actor with concat fusion and instructions enabled
    actor = Actor(
        input_dim=obs_dim,
        output_dim=action_dim,
        mlp_layer_size=[32, 32],
        rnn_layer_size=rnn_layer_size,
        use_instructions=True,
        instruction_fusion='concat',
        freeze_bert=True
    )
    
    # Test with no instruction (simulating the error condition)
    x = torch.randn(1, 1, obs_dim)  # Shape from controller: view(1,1,-1)
    h = torch.zeros(1, 1, rnn_layer_size)
    
    print(f"Input shape: {x.shape}")
    print(f"Hidden state shape: {h.shape}")
    print(f"Actor fc1 expects input size: {actor.fc1.in_features}")
    
    try:
        # This should NOT fail anymore
        action_logits, new_h = actor(x, h, instruction_emb=None, test_mode=True)
        print(f"✓ Success! Action logits shape: {action_logits.shape}")
        print(f"✓ Expected: (1, 1, {action_dim}), got: {action_logits.shape}")
        return True
    except RuntimeError as e:
        print(f"✗ Failed with error: {e}")
        return False

def test_concat_with_instruction():
    """Test that concat fusion still works when instruction IS provided"""
    print("\n" + "="*60)
    print("Testing concat fusion WITH instruction provided")
    print("="*60)
    
    obs_dim = 56
    action_dim = 12
    rnn_layer_size = 32
    
    actor = Actor(
        input_dim=obs_dim,
        output_dim=action_dim,
        mlp_layer_size=[32, 32],
        rnn_layer_size=rnn_layer_size,
        use_instructions=True,
        instruction_fusion='concat',
        freeze_bert=True
    )
    
    x = torch.randn(1, 1, obs_dim)
    h = torch.zeros(1, 1, rnn_layer_size)
    
    # Create a dummy instruction embedding
    instruction_emb = torch.randn(1, rnn_layer_size)
    
    print(f"Input shape: {x.shape}")
    print(f"Instruction embedding shape: {instruction_emb.shape}")
    
    try:
        action_logits, new_h = actor(x, h, instruction_emb=instruction_emb, test_mode=True)
        print(f"✓ Success! Action logits shape: {action_logits.shape}")
        return True
    except RuntimeError as e:
        print(f"✗ Failed with error: {e}")
        return False

def test_critic():
    """Test that Critic also works correctly"""
    print("\n" + "="*60)
    print("Testing Critic with concat fusion and no instruction")
    print("="*60)
    
    obs_dim = 56
    rnn_layer_size = 32
    
    critic = Critic(
        input_dim=obs_dim,
        output_dim=1,
        mlp_layer_size=[32, 32],
        rnn_layer_size=rnn_layer_size,
        use_instructions=True,
        instruction_fusion='concat',
        freeze_bert=True
    )
    
    x = torch.randn(1, 1, obs_dim)
    h = torch.zeros(1, 1, rnn_layer_size)
    
    print(f"Input shape: {x.shape}")
    print(f"Critic fc1 expects input size: {critic.fc1.in_features}")
    
    try:
        values, new_h = critic(x, h, instruction_emb=None)
        print(f"✓ Success! Values shape: {values.shape}")
        return True
    except RuntimeError as e:
        print(f"✗ Failed with error: {e}")
        return False

if __name__ == '__main__':
    results = []
    
    results.append(test_concat_with_no_instruction())
    results.append(test_concat_with_instruction())
    results.append(test_critic())
    
    print("\n" + "="*60)
    if all(results):
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
