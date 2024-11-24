import os 
current_cwd = os.getcwd()
new_cwd = '/'.join(current_cwd.split('\\')[:2])

'''Some paths''' 
TRAIN_IMAGES_PATH = f'{new_cwd}/data/bkai-igh-neopolyp/train/train/'
TRAIN_IMAGES_DIR = f'{new_cwd}/data/bkai-igh-neopolyp/train/train'
TRAIN_MASKS_PATH = f'{new_cwd}/data/bkai-igh-neopolyp/train_gt/train_gt/'
TRAIN_MASKS_DIR = f'{new_cwd}/data/bkai-igh-neopolyp/train_gt/train_gt'
TEST_IMAGES_PATH = f'{new_cwd}/data/bkai-igh-neopolyp/test/test/'
TEST_IMAGES_DIR = f'{new_cwd}/data/bkai-igh-neopolyp/test/test'


'''Encoder name '''

# ENCODER_NAME="resnet18"
# ENCODER_NAME="resnet34"
# ENCODER_NAME="resnet50"
# ENCODER_NAME="resnet101"
ENCODER_NAME="resnet152"


'''Check point path '''

SAVE_PATH=f'{new_cwd}/checkpoint/unet_{ENCODER_NAME}.pth'


'''Prediction path'''
PREDICTION_PATH=f'{new_cwd}/unet_{ENCODER_NAME}_prediction/'


'''Mask dir path'''
MASK_DIR_PATH = f'{new_cwd}/unet_{ENCODER_NAME}_prediction'


'''Submission path'''
SUBMISSION_PATH=f'{new_cwd}/submissions/unet_{ENCODER_NAME}.csv'


'''Encoder weights '''       
ENCODER_WEIGHTS="imagenet"


'''Number of in channels'''
IN_CHANNELS = 3


'''Number of classes '''
N_CLASSES = 3


'''Number of epochs '''
EPOCHS = 200


'''Hyperparameters '''
BATCHSIZE=8
LEARNING_RATE = 0.0001


'''Train and validation ratio '''
TRAIN_RATIO = 0.9
VAL_RATIO = 0.1


'''Color dictionary '''
COLOR_DICT = {0: (0, 0, 0),
             1: (255, 0, 0),
             2: (0, 255, 0)}