# **UNet-for-Colonoscopy-Polyp-Segmentation**

This repository provides an implementation of the UNet model for semantic segmentation tasks, specifically for detecting polyps in colonoscopy images. The repository includes training scripts, pretrained model checkpoints, and an inference script for generating segmentation masks.

## **Features**
* Implements a UNet architecture with a configurable encoder (e.g., ResNet).
* Saves the best model checkpoint during training.
* Includes a user-friendly `infer.py` script to run inference on test images.

## **Repository Structure**
```perl
└── data
    ├── test
    ├── train
    ├── train_gt
    ├── sample_submission.csv

└── src
    ├── dataset.py
    ├── model.py
    ├── training.ipynb

└── utils
    ├── log.py
    ├── mask2rgb.py
    ├── mask2string.py
    ├── train.py
```

## **Installation**
Clone the repository and navigate to the project directorty 

```bash
git clone fisherman611/UNet-for-Colonoscopy-Polyp-Segmentation
cd UNet-for-Colonoscopy-Polyp-Segmetation
```

Install the required dependencies: 

```bash
pip install -r requirements.txt
```

## **Using the pretrained model**
### **Download the pretrained model**
Download the pretrained model checkpoint from this [Google Drive link](https://drive.google.com/drive/folders/18kWRcWNJSeOZNZdteDuI8rdYpDkq1CL_?usp=sharing).

Place the downloaded checkpoint in the `checkpoint/` directory within the repository. The expected path is:


## **Inference**
Use the provided `infer.py` script to generate segmentation masks for test images.

### **Command**
Run the following command, replace `image.jpeg` with the path to your input image:
```bash
python3 infer.py --image_path image.jpeg
```

## **License**
This project is licensed under the [MIT License](LICENSE).
