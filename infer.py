import os
import sys
current_cwd = os.getcwd()
new_cwd = '/'.join(current_cwd.split('\\')[:2])
sys.path.append(f'{new_cwd}/utils')
sys.path.append(f'{new_cwd}/src')

import torch
import cv2
import numpy as np
from model import unet_model  
from mask2rgb import mask2rgb  
from log import * 
from dataset import * 
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensorV2

OUTPUT_PATH = f"{new_cwd}/infer/"  # Directory to save predictions
os.makedirs(OUTPUT_PATH, exist_ok=True)

def infer(image_path):
    # Load the model
    model = unet_model
    checkpoint = torch.load(SAVE_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Read and preprocess the image
    ori_img = cv2.imread(image_path)
    if ori_img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_h, ori_w = ori_img.shape[:2]
    resized_img = cv2.resize(ori_img, (256, 256))
    transformed = val_transformation(image=resized_img)
    input_img = transformed["image"].unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output_mask = model(input_img).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    mask = cv2.resize(output_mask, (ori_w, ori_h))
    mask = np.argmax(mask, axis=2)

    # Convert mask to RGB
    mask_rgb = mask2rgb(mask, COLOR_DICT)
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)

    # Save the prediction
    output_path = os.path.join(OUTPUT_PATH, os.path.basename(image_path))
    cv2.imwrite(output_path, mask_rgb)
    print(f"Segmented mask saved at: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UNet Inference Script")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    args = parser.parse_args()

    infer(args.image_path)
