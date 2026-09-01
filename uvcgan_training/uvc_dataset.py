import os
import shutil

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SOURCE_DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")
TARGET_UVCGAN_DIR = os.path.join(BASE_DIR, "data", "crystal_dataset")

MAPPING = {
    os.path.join("synth", "train"): os.path.join("trainSYNTH"),
    os.path.join("real", "train"): os.path.join("trainREAL"),
    os.path.join("synth", "val"): os.path.join("testSYNTH"),
    os.path.join("real", "val"): os.path.join("testREAL")
}

def prepare_uvcgan_dataset():
    print("Preparing simplified structure for UVCGAN2...")
    
    if os.path.exists(TARGET_UVCGAN_DIR):
        shutil.rmtree(TARGET_UVCGAN_DIR)
    
    os.makedirs(TARGET_UVCGAN_DIR, exist_ok=True)

    for src_subpath, target_subpath in MAPPING.items():
        src_dir = os.path.join(SOURCE_DATASET_DIR, src_subpath)
        target_dir = os.path.join(TARGET_UVCGAN_DIR, target_subpath)

        if not os.path.exists(src_dir):
            print(f"Warning: {src_dir} not found.")
            continue

        os.makedirs(target_dir, exist_ok=True)
        files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        
        for filename in files:
            shutil.copy2(os.path.join(src_dir, filename), os.path.join(target_dir, filename))
            
    print("Done. Dataset structure is now: data/crystal_dataset/[synth, real]")

if __name__ == "__main__":
    prepare_uvcgan_dataset()