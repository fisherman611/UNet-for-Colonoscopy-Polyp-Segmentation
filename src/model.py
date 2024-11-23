import segmentation_models_pytorch as smp 
import os 
import sys 

sys.path.append('/'.join(os.getcwd().split('\\')[:2]) + '/utils')

from log import * 

# I will use the UNet model using the segmentation_models_pytorch library built on Pytorch framework
unet_model = smp.Unet(
    encoder_name=ENCODER_NAME,        
    encoder_weights=ENCODER_WEIGHTS,     
    in_channels=IN_CHANNELS,                  
    classes=N_CLASSES 
    )