#!/usr/bin/env python3
"""
Test script to check if a policy model supports instructions.

This script loads a policy and tests:
1. Whether it has instruction support enabled (use_instructions attribute)
2. Whether the model architecture includes instruction-related components
3. Whether passing an instruction changes the model's behavior
"""

import torch
import os
import sys
import numpy as np

# Add paths for custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import the Actor model class so torch.load can reconstruct it
try:
    from macro_marl.cores.pg_based.mac_iac.models import Actor, Critic
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    MODELS_AVAILABLE = False

def test_policy_instruction_support(policy_path):
    """
    Test if a policy supports instructions.
    
    Args:
        policy_path: Path to the policy .pt file
    
    Returns:
        dict with test results
    """
    print(f"\n{'='*80}")
    print(f"Testing Policy: {policy_path}")
    print(f"{'='*80}\n")
    
    if not MODELS_AVAILABLE:
        return {
            'error': "Model classes not available. Cannot load policy.",
            'supports_instructions': False
        }
    
    # Load the policy
    try:
        policy = torch.load(policy_path, map_location='cpu', weights_only=False)
    except Exception as e:
        return {
            'error': f"Failed to load policy: {str(e)}",
            'supports_instructions': False
        }
    
    policy.eval()
    
    results = {
        'policy_path': policy_path,
        'supports_instructions': False,
        'has_use_instructions_attr': False,
        'use_instructions_value': None,
        'has_instruction_encoder': False,
        'has_instruction_projection': False,
        'input_dim': None,
        'output_dim': None,
        'instruction_fusion_method': None,
        'behavior_changes_with_instruction': None,
    }
    
    # Test 1: Check if the policy has use_instructions attribute
    print("Test 1: Checking for 'use_instructions' attribute...")
    if hasattr(policy, 'use_instructions'):
        results['has_use_instructions_attr'] = True
        results['use_instructions_value'] = policy.use_instructions
        results['supports_instructions'] = policy.use_instructions
        print(f"  ✓ Found 'use_instructions' = {policy.use_instructions}")
    else:
        print(f"  ✗ No 'use_instructions' attribute found")
    
    # Test 2: Check for instruction-related components in the model
    print("\nTest 2: Checking for instruction-related components...")
    
    if hasattr(policy, 'instruction_encoder'):
        results['has_instruction_encoder'] = True
        print(f"  ✓ Found 'instruction_encoder' (BERT model)")
    else:
        print(f"  ✗ No 'instruction_encoder' found")
    
    if hasattr(policy, 'instruction_projection'):
        results['has_instruction_projection'] = True
        print(f"  ✓ Found 'instruction_projection' layer")
    else:
        print(f"  ✗ No 'instruction_projection' found")
    
    if hasattr(policy, 'instruction_fusion'):
        results['instruction_fusion_method'] = policy.instruction_fusion
        print(f"  ✓ Instruction fusion method: {policy.instruction_fusion}")
    else:
        print(f"  ✗ No 'instruction_fusion' attribute found")
    
    # Test 3: Check model dimensions
    print("\nTest 3: Checking model dimensions...")
    if hasattr(policy, 'fc1'):
        results['input_dim'] = policy.fc1.in_features
        print(f"  ✓ Input dimension: {results['input_dim']}")
    
    if hasattr(policy, 'fc4'):
        results['output_dim'] = policy.fc4.out_features
        print(f"  ✓ Output dimension (actions): {results['output_dim']}")
    
    # Test 4: Test forward pass with and without instruction
    print("\nTest 4: Testing forward pass behavior...")
    
    try:
        # Get input dimension
        if results['input_dim'] is None:
            state_dict = policy.state_dict()
            input_dim = state_dict['fc1.weight'].shape[1]
        else:
            input_dim = results['input_dim']
        
        # Create dummy observation
        dummy_obs = torch.randn(1, 1, input_dim)
        
        # Forward pass without instruction
        print("  Testing without instruction...")
        with torch.no_grad():
            output_no_inst, h1 = policy(dummy_obs)
        print(f"    ✓ Forward pass successful (output shape: {output_no_inst.shape})")
        
        # Forward pass with instruction (if supported)
        if results['supports_instructions']:
            print("  Testing with instruction...")
            try:
                with torch.no_grad():
                    output_with_inst, h2 = policy(dummy_obs, instruction="get tomato")
                print(f"    ✓ Forward pass with instruction successful")
                
                # Check if outputs differ
                diff = torch.abs(output_no_inst - output_with_inst).max().item()
                if diff > 1e-6:
                    results['behavior_changes_with_instruction'] = True
                    print(f"    ✓ Output changes with instruction (max diff: {diff:.6f})")
                else:
                    results['behavior_changes_with_instruction'] = False
                    print(f"    ⚠ Output SAME with instruction (max diff: {diff:.6f})")
                    print(f"      This might indicate the instruction is being ignored!")
            except Exception as e:
                print(f"    ✗ Forward pass with instruction failed: {str(e)}")
                results['behavior_changes_with_instruction'] = False
        else:
            print("  ⊘ Skipping instruction test (model doesn't support instructions)")
            
    except Exception as e:
        print(f"  ✗ Forward pass test failed: {str(e)}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test if a policy supports instructions')
    parser.add_argument('--policy_path', type=str, default=None,
                        help='Path to policy .pt file. If not provided, tests default policies.')
    parser.add_argument('--mapType', type=str, default='A', choices=['A', 'B', 'C'],
                        help='Map type for default policy search')
    parser.add_argument('--agent_idx', type=int, default=0,
                        help='Agent index for default policy search')
    parser.add_argument('--policy_prefix', type=str, default='inst3',
                        help='Policy file prefix (e.g., inst2, inst3, inst4)')
    
    args = parser.parse_args()
    
    if args.policy_path:
        policy_paths = [args.policy_path]
    else:
        # Test default policies
        base_path = os.path.join(os.path.dirname(__file__), "visualization", "policy_nns", "Overcooked", f"map{args.mapType}")
        policy_paths = []
        
        # Try different policy prefixes
        for prefix in ['inst2', 'inst3', 'inst4', '4']:
            policy_path = os.path.join(base_path, f"{prefix}_agent_{args.agent_idx}.pt")
            if os.path.exists(policy_path):
                policy_paths.append(policy_path)
        
        if not policy_paths:
            print(f"No policies found in {base_path}")
            print(f"Tried prefixes: inst2, inst3, inst4, 4")
            return
    
    all_results = []
    for policy_path in policy_paths:
        if not os.path.exists(policy_path):
            print(f"\n⚠ Policy not found: {policy_path}")
            continue
        
        results = test_policy_instruction_support(policy_path)
        all_results.append(results)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    for i, results in enumerate(all_results, 1):
        if 'error' in results:
            print(f"{i}. {os.path.basename(results.get('policy_path', 'unknown'))}: ERROR - {results['error']}")
        else:
            policy_name = os.path.basename(results['policy_path'])
            support = "✓ YES" if results['supports_instructions'] else "✗ NO"
            print(f"{i}. {policy_name}:")
            print(f"   Supports instructions: {support}")
            if results['supports_instructions']:
                print(f"   Fusion method: {results['instruction_fusion_method']}")
                if results['behavior_changes_with_instruction'] is not None:
                    behavior = "✓ YES" if results['behavior_changes_with_instruction'] else "⚠ NO (might be ignored)"
                    print(f"   Behavior changes: {behavior}")
            print(f"   Input dim: {results['input_dim']}, Output dim: {results['output_dim']}")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()

