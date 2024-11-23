import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class NeoPolypDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading images and their corresponding masks.

    Args:
        img_dir (str): Path to the directory containing input images.
        label_dir (str): Path to the directory containing label masks.
        resize (tuple): Desired size (width, height) to resize the images and masks.
        transform (callable, optional): Optional transformation to be applied on images.

    Attributes:
        img_dir (str): Directory path for input images.
        label_dir (str): Directory path for label masks.
        resize (tuple): Tuple containing target size for resizing (width, height).
        transform (callable): Transform function applied to images.
        images (list): List of filenames for images in `img_dir`.
    """
    
    def __init__(self, img_dir, label_dir, resize=None, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.resize = resize
        self.transform = transform
        self.images = os.listdir(self.img_dir)  # List all image filenames in the directory

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Number of images.
        """
        return len(self.images)
    
    def read_mask(self, mask_path):
        """
        Reads and processes a mask image from the given path.

        Args:
            mask_path (str): Path to the mask image.

        Returns:
            np.ndarray: Processed mask image with red and green regions labeled distinctly.
        """
        # Read the mask image
        image = cv2.imread(mask_path)
        
        # Resize the mask to the target size
        image = cv2.resize(image, self.resize)
        
        # Convert the mask image to HSV color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color thresholds for red in HSV
        lower_red1 = np.array([0, 100, 20])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 20])
        upper_red2 = np.array([179, 255, 255])

        # Create masks for red regions (split into two ranges for better coverage)
        lower_mask_red = cv2.inRange(image, lower_red1, upper_red1)
        upper_mask_red = cv2.inRange(image, lower_red2, upper_red2)
        
        # Combine the two red masks
        red_mask = lower_mask_red + upper_mask_red
        red_mask[red_mask != 0] = 1  # Label red regions as 1

        # Define color thresholds for green in HSV
        green_mask = cv2.inRange(image, (36, 25, 25), (70, 255, 255))
        green_mask[green_mask != 0] = 2  # Label green regions as 2

        # Combine red and green masks into a single mask
        full_mask = cv2.bitwise_or(red_mask, green_mask)
        
        # Add an additional dimension to make it compatible with model input
        full_mask = np.expand_dims(full_mask, axis=-1)
        full_mask = full_mask.astype(np.uint8)
        
        return full_mask

    def __getitem__(self, idx):
        """
        Fetches the image and its corresponding mask by index.

        Args:
            idx (int): Index of the image-mask pair to fetch.

        Returns:
            tuple: A tuple containing the processed image and its corresponding mask.
        """
        # Construct the paths to the image and label files
        img_path = os.path.join(self.img_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx])
        
        # Load the image and convert from BGR to RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load and process the mask
        label = self.read_mask(label_path)
        
        # Resize the image to the target size
        image = cv2.resize(image, self.resize)
        
        # Apply any transformations, if provided
        if self.transform:
            image = self.transform(image)
            
        return image, label

class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset for handling paired image and label data, 
    with optional transformations applied.

    Args:
        data (list or np.ndarray): List or array of input images.
        targets (list or np.ndarray): List or array of corresponding labels/masks.
        transform (callable, optional): A transformation function (e.g., Albumentations) 
                                        that takes an image and mask as input and returns transformed results.

    Attributes:
        data (list or np.ndarray): The input images.
        targets (list or np.ndarray): The corresponding labels or masks.
        transform (callable): Transformation function applied to images and labels.
    """
    
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        """
        Fetches an image-label pair by index and applies optional transformations.

        Args:
            index (int): Index of the data point to fetch.

        Returns:
            tuple: A tuple containing the transformed image and label/mask.
        """
        # Get the image and corresponding label at the specified index
        image = self.data[index]
        label = self.targets[index]
        
        # Ensure the spatial dimensions of the image and label match
        assert image.shape[:2] == label.shape[:2], \
            "Image and label dimensions do not match!"
        
        # Apply transformations, if provided
        if self.transform:
            # Transform the image and label using the provided function
            transformed = self.transform(image=image, mask=label)
            image = transformed['image'].float()  # Convert image to float tensor
            label = transformed['mask'].float()  # Convert label to float tensor
            
            # Rearrange label dimensions to match model input (C, H, W)
            label = label.permute(2, 0, 1)
        
        return image, label

    def __len__(self):
        """
        Returns the total number of data points in the dataset.

        Returns:
            int: The number of image-label pairs.
        """
        return len(self.data)


# Transformations for the training dataset
train_transformation = A.Compose([
    # Apply a horizontal flip with a 40% probability
    A.HorizontalFlip(p=0.4),
    
    # Apply a vertical flip with a 40% probability
    A.VerticalFlip(p=0.4),
    
    # Randomly adjust the gamma value to modify image brightness
    # Gamma is chosen from the range (70, 130), applied with 20% probability
    A.RandomGamma(gamma_limit=(70, 130), p=0.2),
    
    # Randomly shift the RGB color channels by a small amount
    # Shifts are limited to ±10 for each channel (R, G, B), applied with 30% probability
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
    
    # Normalize the image using the ImageNet mean and standard deviation
    # mean: (0.485, 0.456, 0.406)
    # std: (0.229, 0.224, 0.225)
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    
    # Convert the image and mask to PyTorch tensors
    ToTensorV2(),
])

# Transformations for the validation dataset
val_transformation = A.Compose([
    # Normalize the image using the ImageNet mean and standard deviation
    # mean: (0.485, 0.456, 0.406)
    # std: (0.229, 0.224, 0.225)
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    
    # Convert the image and mask to PyTorch tensors
    ToTensorV2(),
])