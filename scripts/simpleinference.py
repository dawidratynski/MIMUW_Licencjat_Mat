import os
import json
import torch
from PIL import Image
from torchvision import transforms
from typing import Union, Optional, Tuple

class AttrDict(dict):
    def __getattr__(self, key):
        try:
            val = self[key]
            if isinstance(val, dict):
                return AttrDict(val)
            return val
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


def to_attr_dict(obj):
    if isinstance(obj, dict):
        return AttrDict({k: to_attr_dict(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [to_attr_dict(elem) for elem in obj]
    return obj


try:
    from uvcgan2.models.generator import construct_generator
except ImportError:
    try:
        from uvcgan2.models import construct_generator
    except ImportError:
        from uvcgan2.models import construct_model as construct_generator


def infer_uvcgan(
    image_input: Union[str, Image.Image],
    model_dir: str,
    direction: str = "ab",
    output_path: Optional[str] = None,
    img_size: Optional[Tuple[int, int]] = None,
    device: Optional[str] = None
) -> Image.Image:
    """
    Performs inference using a trained UVCGAN2 generator model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load config
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # 2. Extract image size from config if not explicitly provided
    if img_size is None:
        dataset_cfg = config.get("dataset", {}) if isinstance(config, dict) else {}
        spatial_size = dataset_cfg.get("spatial_size") or dataset_cfg.get("size")
        if isinstance(spatial_size, int):
            img_size = (spatial_size, spatial_size)
        elif isinstance(spatial_size, (list, tuple)) and len(spatial_size) >= 2:
            img_size = (spatial_size[0], spatial_size[1])
        else:
            # Default to 160x160 matching the x_const shape of 100
            img_size = (160, 160)

    # 3. Extract generator configuration safely
    gen_config = None
    if isinstance(config, dict):
        if "generator" in config:
            gen_config = config["generator"]
        elif isinstance(config.get("model"), dict):
            gen_config = config["model"].get("generator")
        elif isinstance(config.get("model_args"), dict):
            gen_config = config["model_args"].get("generator")

    if gen_config is None:
        raise KeyError(f"Could not find generator configuration in {config_path}.")

    gen_config_obj = to_attr_dict(gen_config)

    # 4. Define tensor shapes (3, H, W)
    input_shape = (3, img_size[0], img_size[1])
    output_shape = (3, img_size[0], img_size[1])

    # 5. Instantiate Generator and load weights
    weights_filename = f"net_gen_{direction}.pth"
    weights_path = os.path.join(model_dir, weights_filename)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    generator = construct_generator(gen_config_obj, input_shape, output_shape, device)
    
    state_dict = torch.load(weights_path, map_location=device)
    generator.load_state_dict(state_dict)
    generator.to(device)
    generator.eval()

    # 6. Preprocess Input Image
    transform_in = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if isinstance(image_input, str):
        raw_img = Image.open(image_input).convert("RGB")
    else:
        raw_img = image_input.convert("RGB")

    input_tensor = transform_in(raw_img).unsqueeze(0).to(device)

    # 7. Run Model Forward Pass
    with torch.no_grad():
        output_tensor = generator(input_tensor)

    # 8. Postprocess Output
    output_tensor = output_tensor.squeeze(0).cpu()
    output_tensor = (output_tensor * 0.5 + 0.5).clamp(0, 1)

    transform_out = transforms.ToPILImage()
    result_img = transform_out(output_tensor)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        result_img.save(output_path)
        print(f"Saved translated image to: {output_path}")

    return result_img


if __name__ == "__main__":
    MODEL_DIRECTORY = "outM/crystal_experiments/translation/model_m(uvcgan2)_d(basic)_g(vit-modnet)_uvcgan2-bsd_(pre_True_cyc_10.0_gp_0.01_lr_0.0001)"
    INPUT_IMAGE = "crystal_data/data/crystal_dataset/train/synth/Kopia images_0.npy_@_0_sample_0.png"
    OUTPUT_IMAGE = "/lu/tetyda/home/jk426853/MIMUW_Licencjat_Mat/translated_sample.png"

    translated_image = infer_uvcgan(
        image_input=INPUT_IMAGE,
        model_dir=MODEL_DIRECTORY,
        direction="ab",
        output_path=OUTPUT_IMAGE,
        img_size=(160, 160)  # Matches 100-dim positional embedding checkpoint
    )