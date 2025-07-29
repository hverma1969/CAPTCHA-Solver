import gzip
import numpy as np
import pandas as pd

def convert_emnist_to_csv(image_file, label_file, output_csv):
    with gzip.open(label_file, 'rb') as lbpath:
        lbpath.read(4)  # magic number
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with gzip.open(image_file, 'rb') as imgpath:
        imgpath.read(4)  # magic number
        num_images = int.from_bytes(imgpath.read(4), 'big')
        rows = int.from_bytes(imgpath.read(4), 'big')
        cols = int.from_bytes(imgpath.read(4), 'big')
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(num_images, rows * cols)

    min_len = min(len(labels), len(images))
    data = np.column_stack((labels[:min_len], images[:min_len]))
    # data = np.column_stack((labels, images))
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, header=False)
    print(f"Saved CSV to {output_csv}")

# Convert train and test datasets
convert_emnist_to_csv(
    "emnist-balanced-train-images-idx3-ubyte.gz",
    "emnist-balanced-train-labels-idx1-ubyte.gz",
    "emnist-balanced-train.csv"
)

convert_emnist_to_csv(
    "emnist-balanced-test-images-idx3-ubyte.gz",
    "emnist-balanced-test-labels-idx1-ubyte.gz",
    "emnist-balanced-test.csv"
)
