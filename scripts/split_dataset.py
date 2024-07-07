import os
import shutil
from sklearn.model_selection import train_test_split

DATA_PATH = 'data'
PATH = 'data/CASIA2'

# Directories for authentic and tampered images
authentic_dir = os.path.join(PATH, 'Au')
tampered_dir = os.path.join(PATH, 'Tp2')

# Output directories
test_dir = os.path.join(DATA_PATH, 'test')
train_dir = os.path.join(DATA_PATH, 'train')

# Create output directories
os.makedirs(test_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(os.path.join(test_dir, 'authentic'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'tampered'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'authentic'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'tampered'), exist_ok=True)


def get_file_list_and_labels(directory, label):
    file_list = []
    labels = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_list.append(os.path.join(root, file))
                labels.append(label)
    return file_list, labels


# Get file lists and labels
authentic_files, authentic_labels = get_file_list_and_labels(authentic_dir, 0)
tampered_files, tampered_labels = get_file_list_and_labels(tampered_dir, 1)

# Combine authentic and tampered data
all_files = authentic_files + tampered_files
all_labels = authentic_labels + tampered_labels

# Split the data into training, validation, and test sets
TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.2

train_files, test_files, train_labels, test_labels = train_test_split(
    all_files, all_labels, test_size=TEST_SPLIT, random_state=42, stratify=all_labels
)

print(f"Train set: {len(train_files)} images ({TRAIN_SPLIT * 100:.2f}%)")
print(f"Test set: {len(test_files)} images ({TEST_SPLIT * 100:.2f}%)")

# Copy files to the appropriate directories
for src_file, label, dst_dir in zip(
    train_files, train_labels,
    [os.path.join(train_dir, 'authentic')] * sum('Au' in src_file for src_file in train_files) +
    [os.path.join(train_dir, 'tampered')] * sum('Tp' in src_file for src_file in train_files)
):
    if 'Au' in src_file:
        dst_file = os.path.join(os.path.join(train_dir, 'authentic'), f"{os.path.basename(src_file)}")
    elif 'Tp' in src_file:
        dst_file = os.path.join(os.path.join(train_dir, 'tampered'), f"{os.path.basename(src_file)}")
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    shutil.copy(src_file, dst_file)

for src_file, label, dst_dir in zip(
    test_files, test_labels, [os.path.join(test_dir, 'authentic')] * sum('Au' in src_file for src_file in test_files) +
                             [os.path.join(test_dir, 'tampered')] * sum('Tp' in src_file for src_file in test_files)
):
    if 'Au' in src_file:
        dst_file = os.path.join(os.path.join(test_dir, 'authentic'), f"{os.path.basename(src_file)}")
    elif 'Tp' in src_file:
        dst_file = os.path.join(os.path.join(test_dir, 'tampered'), f"{os.path.basename(src_file)}")
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    shutil.copy(src_file, dst_file)