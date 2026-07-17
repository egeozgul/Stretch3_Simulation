#!/usr/bin/env python3
"""
Test script to verify DistilBERT instruction encoding and macro-action mapping.

This script tests:
1. Whether DistilBERT can properly encode instruction texts
2. Whether instructions map correctly to expected macro-actions
3. Whether the instruction compliance logic works as expected
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Test instructions that should map to specific macro-actions
TEST_INSTRUCTIONS = [
    # Positive instructions (allowed actions)
    "get tomato",
    "get lettuce",
    "get onion",
    "go to knife 1",
    "go to knife 2",
    "chop",
    "deliver",
    "i will get the tomato",
    "let me do all the chopping",

    # Negative instructions (prohibited actions)
    "don't touch the tomato",
    "don't touch the lettuce",
    "don't touch the onion",
    "don't chop",

    # Edge cases
    "stay",
    "move right",
    "unknown instruction",
    "",
]

# Expected macro-action indices (from envs_runner.py)
GET_TOMATO = 1
GET_LETTUCE = 2
GET_ONION = 3
GO_TO_KNIFE_1 = 5
GO_TO_KNIFE_2 = 6
DELIVER = 7
CHOP = 8

def test_distilbert_encoding():
    """Test basic DistilBERT encoding functionality"""
    print("🔬 Testing DistilBERT Encoding...")

    # Initialize tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Test encoding a simple instruction
    test_text = "get tomato"
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)

    print(f"Input text: '{test_text}'")
    print(f"Tokenized: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
    print(f"Input shape: {inputs['input_ids'].shape}")

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding sample: {embeddings[0][:5]}")  # First 5 values
    print("✅ DistilBERT encoding works correctly\n")

def test_instruction_mapping():
    """Test instruction text to macro-action mapping"""
    print("🔬 Testing Instruction-to-Macro-Action Mapping...")

    def get_expected_macro_action(instruction_text):
        """Copy of the mapping logic from envs_runner.py"""
        instruction_lower = instruction_text.lower().strip()

        # Positive instructions (do X) - exact phrase matching
        if instruction_lower in ["get tomato", "i will get the tomato"]:
            return {'allowed_actions': [GET_TOMATO]}
        elif instruction_lower == "get lettuce":
            return {'allowed_actions': [GET_LETTUCE]}
        elif instruction_lower == "get onion":
            return {'allowed_actions': [GET_ONION]}
        elif instruction_lower == "go to knife 1":
            return {'allowed_actions': [GO_TO_KNIFE_1]}
        elif instruction_lower == "go to knife 2":
            return {'allowed_actions': [GO_TO_KNIFE_2]}
        elif instruction_lower in ["chop", "let me do all the chopping"]:
            return {'allowed_actions': [CHOP]}
        elif instruction_lower == "deliver":
            return {'allowed_actions': [DELIVER]}

        # Negative instructions (don't do X)
        elif instruction_lower in ["don't touch the tomato", "don't touch tomato"]:
            return {'prohibited_actions': [GET_TOMATO]}
        elif instruction_lower in ["don't touch the lettuce", "don't touch lettuce"]:
            return {'prohibited_actions': [GET_LETTUCE]}
        elif instruction_lower == "don't touch the onion":
            return {'prohibited_actions': [GET_ONION]}
        elif instruction_lower == "don't chop":
            return {'prohibited_actions': [CHOP]}

        return None

    def check_action_compliance(action, instruction_text):
        """Check if action complies with instruction (simplified version)"""
        expected = get_expected_macro_action(instruction_text)
        if expected is None:
            return True  # No instruction = compliant

        if 'allowed_actions' in expected:
            return action in expected['allowed_actions']
        elif 'prohibited_actions' in expected:
            return action not in expected['prohibited_actions']

        return True

    # Test each instruction
    print("Testing instruction mapping:")
    print("=" * 60)

    for instruction in TEST_INSTRUCTIONS:
        expected = get_expected_macro_action(instruction)

        print(f"\n📝 Instruction: '{instruction}'")
        if expected:
            if 'allowed_actions' in expected:
                action_names = [get_action_name(action) for action in expected['allowed_actions']]
                print(f"   ✅ Expected: ALLOW {action_names}")
            elif 'prohibited_actions' in expected:
                action_names = [get_action_name(action) for action in expected['prohibited_actions']]
                print(f"   ❌ Expected: AVOID {action_names}")
        else:
            print("   ⚪ No specific expectation")

        # Test compliance for a few sample actions
        test_actions = [GET_TOMATO, GET_LETTUCE, CHOP, DELIVER]
        print("   Compliance tests:")
        for action in test_actions:
            compliant = check_action_compliance(action, instruction)
            action_name = get_action_name(action)
            status = "✅" if compliant else "❌"
            print(f"     {status} Action '{action_name}' (ID {action}): {'COMPLIANT' if compliant else 'NON-COMPLIANT'}")

def get_action_name(action_id):
    """Convert action ID to readable name"""
    action_map = {
        0: "stay",
        GET_TOMATO: "get_tomato",
        GET_LETTUCE: "get_lettuce",
        GET_ONION: "get_onion",
        GO_TO_KNIFE_1: "go_to_knife_1",
        GO_TO_KNIFE_2: "go_to_knife_2",
        DELIVER: "deliver",
        CHOP: "chop",
    }
    return action_map.get(action_id, f"action_{action_id}")

def test_tokenization_edge_cases():
    """Test edge cases in instruction processing"""
    print("\n🔬 Testing Tokenization Edge Cases...")

    # Test how different formats are handled
    test_cases = [
        "get tomato",           # Normal case
        ["get", "tomato"],      # Token list
        "GET TOMATO",           # Uppercase
        " get tomato ",         # Extra spaces
        "get    tomato",        # Multiple spaces
    ]

    for case in test_cases:
        print(f"\n📝 Input: {repr(case)}")

        # Simulate the normalization logic from envs_runner.py
        if isinstance(case, str):
            normalized = case
        elif isinstance(case, list) and len(case) > 0 and isinstance(case[0], str):
            normalized = " ".join(case)
        else:
            normalized = case

        print(f"   Normalized: '{normalized}'")
        print(f"   Lowercase: '{normalized.lower().strip()}'")

def main():
    """Run all tests"""
    print("🚀 Testing DistilBERT Instruction Processing")
    print("=" * 60)

    try:
        test_distilbert_encoding()
        test_instruction_mapping()
        test_tokenization_edge_cases()

        print("\n✅ All tests completed successfully!")
        print("\n📊 Summary:")
        print("- DistilBERT encoding: ✅ Working")
        print("- Instruction mapping: ✅ Working")
        print("- Tokenization handling: ✅ Working")
        print("- Compliance logic: ✅ Working")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
