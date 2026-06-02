
import os

def setup():
    # Paths are relative to repo root
    source = os.path.abspath('dprc_stress.py')
    target = os.path.abspath('../stress_tests/dprc_stress.py')
    
    if os.path.exists(target):
        print(f"Removing existing link at {target}")
        os.remove(target)
        
    print(f"Creating symlink: {source} -> {target}")
    os.symlink(source, target)
    
    # Also ensure codes is in the python path for model_1
    # We do this by adding a small .pth file or just relying on the script's sys.path.append
    print("Bridge setup complete.")

if __name__ == "__main__":
    setup()
