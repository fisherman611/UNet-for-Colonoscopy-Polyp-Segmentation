import os
import numpy as np
import cv2

def rle_to_string(runs):
    """
    Converts a list of RLE (Run-Length Encoding) values into a space-separated string.

    Args:
        runs (list): A list of integers representing RLE values.

    Returns:
        str: A space-separated string of RLE values.
    """
    return ' '.join(str(x) for x in runs)

def rle_encode_one_mask(mask):
    """
    Encodes a single binary mask into Run-Length Encoding (RLE).

    Args:
        mask (np.ndarray): A 2D array representing the binary mask (0 or 255).

    Returns:
        str: The RLE string representation of the mask.
    """
    # Flatten the mask into a 1D array
    pixels = mask.flatten()
    
    # Threshold the pixel values to binary (0 or 255)
    pixels[pixels > 225] = 255
    pixels[pixels <= 225] = 0

    # Check for padding at the edges if the mask starts or ends with a filled pixel
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros(len(pixels) + 2, dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded

    # Identify start and lengths of consecutive filled pixels
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]

    # Convert RLE to string format
    return rle_to_string(rle)

def rle2mask(mask_rle, shape=(3, 3)):
    """
    Decodes a Run-Length Encoding (RLE) string back into a binary mask.

    Args:
        mask_rle (str): RLE string representation of the mask.
        shape (tuple): Shape of the output mask (height, width).

    Returns:
        np.ndarray: A 2D binary mask reconstructed from the RLE string.
    """
    # Parse the RLE string into start positions and lengths
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1  # Convert to zero-based indexing
    ends = starts + lengths

    # Create an empty mask
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # Fill the mask based on start and end positions
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape the mask to the desired dimensions
    return img.reshape(shape).T

def mask2string(dir):
    """
    Converts all binary masks in a directory to their RLE string representations.

    Args:
        dir (str): Path to the directory containing mask images.

    Returns:
        dict: A dictionary containing:
              - 'ids': List of image IDs with channel indices (e.g., 'image_1_0').
              - 'strings': List of RLE string representations for each channel.
    """
    strings = []  # List to store RLE strings
    ids = []      # List to store corresponding image IDs
    ws, hs = [[] for _ in range(2)]  # Lists to store image widths and heights

    # Iterate through all images in the directory
    for image_file in os.listdir(dir):
        # Extract the base ID of the image (excluding the extension)
        image_id = image_file.split('.')[0]
        path = os.path.join(dir, image_file)
        print(path)

        # Read the image and convert it from BGR to RGB
        img = cv2.imread(path)[:, :, ::-1]
        h, w = img.shape[:2]

        # Process each channel of the image
        for channel in range(2):  # Assumes the image has at least 2 channels
            ws.append(w)
            hs.append(h)
            ids.append(f'{image_id}_{channel}')

            # Encode the mask for the current channel into RLE
            string = rle_encode_one_mask(img[:, :, channel])
            strings.append(string)

    # Return the results as a dictionary
    return {
        'ids': ids,
        'strings': strings,
    }
