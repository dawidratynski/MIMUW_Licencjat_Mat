import os
import subprocess
import sys

# --- Configuration ---
REPO_URL = "https://github.com/LS4GAN/uvcgan2.git" 
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "UVCGAN", "source")

def setup_repository():
    """Clones the UVCGAN2 repository and installs it via pip."""
    if not os.path.exists(SOURCE_DIR):
        print(f"Starting to clone UVCGAN2 repository from {REPO_URL}...")
        try:
            # 1. Clone the repository
            subprocess.run(["git", "clone", REPO_URL, SOURCE_DIR], check=True)
            print("Repository cloned successfully!\n")
            
            # 2. Install the package in editable mode (-e)
            print("Installing uvcgan2 and its dependencies via pip...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", SOURCE_DIR], check=True)
            print("\nInstallation complete!")
            
        except subprocess.CalledProcessError as e:
            print(f"Error during cloning or installation: {e}")
            sys.exit(1)
    else:
        print(f"Repository already exists in '{SOURCE_DIR}'.")
        print("Running pip install just in case to ensure dependencies are met...\n")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", SOURCE_DIR], check=True)

if __name__ == "__main__":
    setup_repository()