import numpy as np

def mask2rgb(mask, color_dict):
    """
    Converts a segmentation mask into an RGB image using a color dictionary.

    Args:
        mask (np.ndarray): A 2D array where each pixel value represents a class label.
        color_dict (dict): A dictionary mapping class labels to RGB color values.
                           Example: {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0]}

    Returns:
        np.ndarray: A 3D array (H, W, 3) representing the RGB image of the mask.
    """
    # Initialize an empty RGB image with the same height and width as the mask
    output = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)

    # Map each class label in the mask to its corresponding RGB color
    for k in color_dict.keys():
        # Set all pixels belonging to class `k` to the corresponding color
        output[mask == k] = color_dict[k]

    # Convert the output array to unsigned 8-bit integers for image format compatibility
    return np.uint8(output)
