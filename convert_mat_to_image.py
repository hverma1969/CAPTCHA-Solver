import os
import numpy as np
import gzip
import shutil
from PIL import Image

def extract_images_labels(img_path, lbl_path):
    with gzip.open(img_path, 'rb') as imgf:
        imgf.read(16)  # skip magic number & dimensions
        img_data = np.frombuffer(imgf.read(), dtype=np.uint8).reshape(-1, 28, 28)

    with gzip.open(lbl_path, 'rb') as labelf:
        labelf.read(8)  # skip magic number & item count
        labels = np.frombuffer(labelf.read(), dtype=np.uint8)

    return img_data, labels

def save_images(img_data, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, (img, label) in enumerate(zip(img_data, labels)):
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        img = Image.fromarray(np.transpose(img))  # EMNIST images are transposed
        img = img.resize((64, 64))
        img.save(os.path.join(label_dir, f"{label}_{i}.jpg"))

# === Paths (Change if needed)
train_img_path = "emnist-balanced-train-images-idx3-ubyte.gz"
train_lbl_path = "emnist-balanced-train-labels-idx1-ubyte.gz"
test_img_path = "emnist-balanced-test-images-idx3-ubyte.gz"
test_lbl_path = "emnist-balanced-test-labels-idx1-ubyte.gz"

save_images(*extract_images_labels(train_img_path, train_lbl_path), output_dir="new_data/train_char")
save_images(*extract_images_labels(test_img_path, test_lbl_path), output_dir="new_data/test_char")
