# **UNet-for-Colonoscopy-Polyp-Segmentation**

This repository offers an implementation of the UNet model tailored for semantic segmentation tasks, focusing on detecting polyps in colonoscopy images. It includes comprehensive training scripts, pretrained model checkpoints, and an inference script to generate accurate segmentation masks. The dataset utilized for this project, [BKAI-IGH NeoPolyp](https://www.kaggle.com/competitions/bkai-igh-neopolyp/data)

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
git clone https://github.com/fisherman611/UNet-for-Colonoscopy-Polyp-Segmentation.git
```
Navigate to the project directory:
   ```bash
   cd UNet-for-Colonoscopy-Polyp-Segmentation
   ```
(Optional) Install the required dependencies: 

```bash
pip install -r requirements.txt
```
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
