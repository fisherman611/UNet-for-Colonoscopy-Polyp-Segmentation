import segmentation_models_pytorch as smp
import os
import sys

# Add the 'utils' directory to the system path for importing custom modules
sys.path.append('/'.join(os.getcwd().split('\\')[:2]) + '/utils')

# Import logging utilities (custom module)
from log import *

# Initialize the UNet model from the segmentation_models_pytorch library.
# This library is built on top of the PyTorch framework for semantic segmentation tasks.
unet_model = smp.Unet(
    encoder_name=ENCODER_NAME,        # Name of the encoder (e.g., 'resnet34', 'efficientnet-b0')
    encoder_weights=ENCODER_WEIGHTS, # Pre-trained weights for the encoder (e.g., 'imagenet' or None)
    in_channels=IN_CHANNELS,         # Number of input channels (e.g., 3 for RGB images)
    classes=N_CLASSES                # Number of output segmentation classes
)
