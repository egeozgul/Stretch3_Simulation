#!/usr/bin/env python3
"""
Check policy file structure using pickle inspection without loading the model.
"""

import pickle
import io
import sys

class PickleInspector:
    """Inspect pickle file contents without fully unpickling."""
    
    def __init__(self):
        self.keys_found = []
        self.modules_referenced = []
        self.has_instructions = False
    
    def persistent_load(self, pid):
        """Handle persistent IDs in pickle."""
        # Store what we find
        if isinstance(pid, tuple) and len(pid) > 0:
            # This is typically a persistent ID for tensors
            pass
        return None
    
    def find_class(self, module, name):
        """Track what classes are referenced."""
        self.modules_referenced.append(f"{module}.{name}")
        
        # Check for instruction-related classes
        if 'bert' in name.lower() or 'bert' in module.lower():
            self.has_instructions = True
        
        # Return dummy class to avoid import issues
        return type(name, (), {})

def inspect_pickle_structure(file_path):
    """Inspect pickle file structure."""
    print(f"\n{'='*80}")
    print(f"Inspecting: {file_path}")
    print(f"{'='*80}\n")
    
    inspector = PickleInspector()
    
    try:
        with open(file_path, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            unpickler.persistent_load = inspector.persistent_load
            unpickler.find_class = inspector.find_class
            
            try:
                # Attempt to read through the pickle (will fail but we'll collect info)
                unpickler.load()
            except Exception as e:
                # Expected to fail, but we've collected info
                pass
        
        print("Classes/modules referenced:")
        unique_refs = sorted(set(inspector.modules_referenced))
        for ref in unique_refs[:50]:  # Show first 50
            indicator = "  ← INSTRUCTION!" if 'bert' in ref.lower() or 'instruction' in ref.lower() else ""
            print(f"  {ref}{indicator}")
        
        if len(unique_refs) > 50:
            print(f"  ... and {len(unique_refs) - 50} more")
        
        print(f"\n{'='*80}")
        instruction_refs = [r for r in unique_refs if 'bert' in r.lower() or 'instruction' in r.lower()]
        
        if instruction_refs:
            print("✓ POLICY APPEARS TO SUPPORT INSTRUCTIONS")
            print(f"  Found {len(instruction_refs)} instruction-related references")
        else:
            print("✗ POLICY DOES NOT APPEAR TO SUPPORT INSTRUCTIONS")
            print("  No BERT or instruction-related references found")
        
        print(f"{'='*80}\n")
        
        return inspector.has_instructions
        
    except Exception as e:
        print(f"Error inspecting file: {e}")
        return None

if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Inspect policy pickle structure')
    parser.add_argument('policy_path', type=str, help='Path to policy .pt file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.policy_path):
        print(f"Error: File not found: {args.policy_path}")
        sys.exit(1)
    
    result = inspect_pickle_structure(args.policy_path)
    
    if result:
        print("CONCLUSION: This policy was trained WITH instruction support")
    elif result is False:
        print("CONCLUSION: This policy was trained WITHOUT instruction support")
    else:
        print("CONCLUSION: Could not determine")

