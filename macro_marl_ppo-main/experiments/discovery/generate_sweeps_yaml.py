import argparse
import pathlib
from copy import deepcopy
from distutils.util import strtobool

from ruamel.yaml import YAML

# --- Constants and Directory Setup ---
# Resolve to absolute path so the script works regardless of whether __file__
# comes in relative (e.g. when invoked as `python generate_sweeps_yaml.py`)
# or absolute. Previous `.parent.parent / 'discovery'` form broke on Discovery
# because it pointed to `./discovery/...` relative to the CWD.
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent   # experiments/discovery/
SWEEPS_DIR = SCRIPT_DIR / 'sweep_yaml'
GEN_DIR = SCRIPT_DIR / 'gen_commands'
GEN_DIR.mkdir(parents=True, exist_ok=True)
BASIC_FILENAME = 'commands'

def format_param(param_name, value):
    """Format a parameter for command line, handling flags and list values"""
    # Handle boolean flags (empty string value means it's a flag)
    if value == '' or value is True:
        return f'--{param_name}'
    # Handle False values - skip them entirely
    if value is False:
        return None
    # Handle all other values (including strings with spaces for list args like "7 7")
    return f'--{param_name} {value}'


def wandb_sweep_to_textline(sweep):
    """Convert wandb sweep to flat list of commands with all arguments"""
    from itertools import product
    import os
    
    print("  [INFO] Inside wandb_sweep_to_textline: Parsing sweep configuration...")
    conf = deepcopy(sweep)
    
    script = conf.pop('program')
    
    params = conf['parameters']
    
    # Check if script is a console script (no path separator) or a Python file path
    if os.sep not in script and not script.startswith('.'):
        # Console script - call directly without 'python' prefix
        base_line = [script]
        print(f"    - Using console script: {script}")
    else:
        # Python file path - use 'python' prefix
        base_line = ['python', script]
        print(f"    - Using Python script: {script}")
    
    # Identify single-value parameters (parameters with one 'value' or a 'values' list of length 1)
    single_value_params = {k: v['value'] for k, v in params.items() if 'value' in v}
    single_value_params.update({k: v['values'][0] for k, v in params.items() if 'values' in v and len(v['values']) == 1})
    
    # Extract save_dir for special handling (will be made unique per sweep combination)
    base_save_dir = single_value_params.pop('save_dir', None)
    
    # Format single-value params, filtering out None (skipped params)
    formatted_single = [format_param(p, v) for p, v in single_value_params.items()]
    base_line.extend([f for f in formatted_single if f is not None])
    print(f"    - Found {len(single_value_params)} single-value parameters.")
    if base_save_dir:
        print(f"    - Base save_dir: '{base_save_dir}' (will be made unique per sweep combination)")
    
    # Identify multiple-value parameters to create permutations from
    multiple_value_params = {k: v['values'] for k, v in params.items() if k not in single_value_params.keys() and 'values' in v and len(v['values']) > 1}
    print(f"    - Found {len(multiple_value_params)} multi-value parameters to sweep over.")
    
    # Generate all combinations (Cartesian product) of multiple-value parameters
    if multiple_value_params:
        keys, values = zip(*multiple_value_params.items())
        extensions = [dict(zip(keys, v)) for v in product(*values)]
    else:
        extensions = [{}] # If no multi-value params, run the base command once

    # Construct the full command strings with proper formatting
    full_scripts = []
    
    for d in extensions:
        cmd_parts = base_line.copy()
        
        # Create a unique save_dir by appending sweep values.
        # We include ALL multi-value sweep params (except run_id, which is already encoded
        # elsewhere by the training script) so different hyperparameter combinations don't
        # clobber each other's performance/ and policy_nns/ directories.
        if base_save_dir is not None:
            # Priority params go first for readability
            priority = ['alg', 'map_type', 'task']
            suffix_parts = []
            for param in priority:
                if param in d:
                    suffix_parts.append(f"{param}-{d[param]}")
            # Then append every other sweep param in deterministic order
            for param in sorted(k for k in d.keys() if k not in priority and k != 'run_id'):
                val = str(d[param]).replace(' ', '-')
                suffix_parts.append(f"{param}-{val}")
            if suffix_parts:
                unique_save_dir = f"{base_save_dir}__{'_'.join(suffix_parts)}"
            else:
                unique_save_dir = base_save_dir
            cmd_parts.append(format_param('save_dir', unique_save_dir))
        
        formatted_multi = [format_param(p, v) for p, v in d.items()]
        formatted_multi = [f for f in formatted_multi if f is not None]
        full_scripts.append(' '.join(cmd_parts + formatted_multi) + '\n')
    print(f"    - Generated {len(full_scripts)} unique command(s).")
    
    return full_scripts

def generate_combined_text_sweeps(c_confs, args):
    """Generate full text strings that can be run directly"""
    out_name = getattr(args, 'out_name', None) or BASIC_FILENAME
    filename = out_name + '_w_args.txt'
    output_path = GEN_DIR / filename
    mode = 'wt' if not args.append else 'at'
    
    print(f"\n  [INFO] Preparing to write commands to: {output_path}")
    print(f"    - File mode: {'append' if mode == 'at' else 'write (overwrite)'}")

    # Convert sweep config to a list of full-text commands
    lines = wandb_sweep_to_textline(c_confs)
    
    try:
        with output_path.open(mode) as f:
            f.writelines(lines)
        print(f"  [SUCCESS] Successfully wrote {len(lines)} lines to {filename}.")
    except IOError as e:
        print(f"  [ERROR] Failed to write to file: {e}")

if __name__ == "__main__":
    print("--- Script Start: Generating Commands from Sweep Files ---")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, default='mappo_sweep',
                          help="Name of sweep file (without .yaml extension).")
    parser.add_argument('--append', type=lambda x: bool(strtobool(x)), default=True,
                          help="Append to existing file (True) or overwrite it (False).")
    parser.add_argument('--out-name', type=str, default=None,
                          help="Output cmd-file basename (default 'commands' -> commands_w_args.txt). "
                               "Use a unique name per variant to avoid sharing one queue.")
    
    args, _ = parser.parse_known_args()
    print(f"[CONFIG] Running with arguments: config-file='{args.config_file}', append={args.append}")
    
    base_parser = YAML(typ='safe')
    
    # Find all matching YAML files
    search_pattern = args.config_file + '.yaml'
    print(f"[INFO] Searching for sweep files in directory: {SWEEPS_DIR}")
    print(f"[INFO] Using search pattern: {search_pattern}")
    sweep_files = list(SWEEPS_DIR.glob(search_pattern))
    
    if not sweep_files:
        print(f"[WARNING] No YAML files found matching '{search_pattern}'. Please check the file name and location.")
    else:
        print(f"[INFO] Found {len(sweep_files)} matching file(s) to process.")
        for i, path in enumerate(sweep_files):
            print(f"\n--- Processing file {i+1}/{len(sweep_files)}: {path.name} ---")
            try:
                with path.open('r') as f:
                    conf = base_parser.load(f)
                generate_combined_text_sweeps(conf, args)
            except Exception as e:
                print(f"[ERROR] Could not process file {path.name}. Reason: {e}")

    print("\n--- Script End ---")