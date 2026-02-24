#!/usr/bin/env python
"""
Setup verification script.
Run this to verify that your environment is set up correctly.
"""
import sys
import os

def check_conda_env():
    """Check if conda environment is activated."""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env == 'simenv':
        print("✓ Conda environment 'simenv' is activated")
        return True
    else:
        print(f"✗ Conda environment 'simenv' is not activated (current: {conda_env})")
        print("  Run: conda activate simenv")
        return False

def check_imports():
    """Check if all required packages can be imported."""
    required_packages = [
        'mujoco',
        'mujoco.viewer',
        'numpy',
        'pynput',
        'click',
        'trimesh',
    ]
    
    failed = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package} - {e}")
            failed.append(package)
    
    return len(failed) == 0

def check_files():
    """Check if required files exist."""
    required_files = [
        'table_world.xml',
        'stretch.xml',
        'meshes/table_fixed.obj',
        'meshes/tomato.obj',
        'meshes/onion.obj',
        'meshes/plate.obj',
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    missing = []
    
    for file in required_files:
        path = os.path.join(script_dir, file)
        if os.path.exists(path):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - NOT FOUND")
            missing.append(file)
    
    return len(missing) == 0

def check_model_loading():
    """Check if the MuJoCo model can be loaded."""
    try:
        import mujoco
        script_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(script_dir, 'table_world.xml')
        model = mujoco.MjModel.from_xml_path(xml_path)
        print(f"✓ MuJoCo model loads successfully ({model.nq} DOF, {model.nu} actuators)")
        return True
    except Exception as e:
        print(f"✗ Failed to load MuJoCo model: {e}")
        return False

def main():
    print("=" * 60)
    print("Stretch 3 Simulation Environment - Setup Verification")
    print("=" * 60)
    print()
    
    results = []
    
    print("1. Checking conda environment...")
    results.append(check_conda_env())
    print()
    
    print("2. Checking Python packages...")
    results.append(check_imports())
    print()
    
    print("3. Checking required files...")
    results.append(check_files())
    print()
    
    print("4. Checking MuJoCo model loading...")
    results.append(check_model_loading())
    print()
    
    print("=" * 60)
    if all(results):
        print("✅ All checks passed! Your environment is ready.")
        print("   You can now run: python stretch_ros2_sim.py")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)
    print("=" * 60)

if __name__ == '__main__':
    main()

