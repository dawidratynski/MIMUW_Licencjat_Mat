import os
import subprocess
import sys

# BASE_DIR is uvcgan_training/
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Data and Output directories
DATAROOT = os.path.join(BASE_DIR, "data")
OUTDIR = os.path.join(BASE_DIR, "experiments_output")

# Path to your NEW custom training script
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train_crystals.py")

def run_training():
    print("Starting UVCGAN2 training setup...")
    
    # Check if your custom script exists
    if not os.path.exists(TRAIN_SCRIPT):
        print(f"Error: Custom training script not found at '{TRAIN_SCRIPT}'.")
        print("Please create train_crystals.py based on the CelebA template first.")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(OUTDIR, exist_ok=True)

    # Setup Environment Variables for UVCGAN2
    # The UVCGAN2 framework automatically reads these to find data and save models
    env = os.environ.copy()
    env["UVCGAN2_DATA"] = DATAROOT
    env["UVCGAN2_OUTDIR"] = OUTDIR

    print("Environment Variables set:")
    print(f"  UVCGAN2_DATA = {DATAROOT}")
    print(f"  UVCGAN2_OUTDIR = {OUTDIR}")

    # Build the command. 
    # Notice there are no flags here! All settings are inside train_crystals.py
    command = [
        "python", TRAIN_SCRIPT
    ]
    
    command_str = " ".join(command)
    print(f"\nExecuting command:\n{command_str}\n")
    
    try:
        # Run training
        subprocess.run(command, env=env, check=True)
        print("\nTraining finished successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error: {e}")

if __name__ == "__main__":
    run_training()