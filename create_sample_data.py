import os
import shutil
from glob import glob
from tqdm import tqdm

def create_sample_dataset(src_dir, dst_dir, limit_per_class=100):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)

    class_dirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    
    for class_name in tqdm(class_dirs, desc=f'Copying from {src_dir}'):
        src_class_dir = os.path.join(src_dir, class_name)
        dst_class_dir = os.path.join(dst_dir, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)

        # Get limited number of images
        images = glob(os.path.join(src_class_dir, '*'))[:limit_per_class]

        for img_path in images:
            shutil.copy(img_path, dst_class_dir)

# Set your original and sample paths
train_src = 'new_data/train_char'
test_src = 'new_data/test_char'
train_dst = 'sample_data/train_char'
test_dst = 'sample_data/test_char'

# Create sample dataset (adjust limit_per_class if needed)
create_sample_dataset(train_src, train_dst, limit_per_class=10)
create_sample_dataset(test_src, test_dst, limit_per_class=5)
