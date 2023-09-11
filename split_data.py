import os
import random
import shutil

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

dirs = ["train/images","train/labels","test/images","test/labels","validation/images","validation/labels"]

for p in dirs:
    create_directory_if_not_exists(p)

# Path to the folder in Google Drive
source_folder = 'ap_vid1/images'  # Replace with your folder path

# Read all file names in the folder
file_names = os.listdir(source_folder)

# Shuffle the file names
random.shuffle(file_names)

# Calculate the sizes of the three sets (80%, 10%, 10%)
total_files = len(file_names)
# train_size = 1000
# val_size = 200
# test_size = 200
train_size = int(0.8 * total_files)
val_size = int(0.1 * total_files)
test_size = total_files - train_size - val_size

# Divide file names into three setsap_vid1/images'
train_files = file_names[:train_size]
val_files = file_names[train_size:train_size + val_size]
test_files = file_names[train_size + val_size: train_size + val_size + test_size]

# Print the sizes of the three sets
print(f"Train files: {len(train_files)}")
print(f"Validation files: {len(val_files)}")
print(f"Test files: {len(test_files)}")


def copy_files(file_names, source_folder, destination_folder):
    for file_name in file_names:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.copy(source_path, destination_path)


def copy_and_change_extension(file_names, source_folder, destination_folder, new_extension):
    for file_name in file_names:
        base_name = os.path.splitext(file_name)[0]
        
        new_file_name = base_name + new_extension
        source_path = os.path.join(source_folder, new_file_name)
        destination_path = os.path.join(destination_folder, new_file_name)
        
        shutil.copy(source_path, destination_path)


# Paths to the folders in Google Drive
destination_train = './train/images'   # Replace with your train folder path
destination_val = './validation/images'  # Replace with your validation folder path
destination_test = './test/images'    # Replace with your test folder path

# Copy files to respective folders
copy_files(train_files, source_folder, destination_train)
copy_files(val_files, source_folder, destination_val)
copy_files(test_files, source_folder, destination_test)

print("Done Folder architecture images")

source_folder = './ap_vid1/annotations_ap/'  # Replace with your folder path

destination_train = './train/labels'   # Replace with your train folder path
destination_val = './validation/labels'  # Replace with your validation folder path
destination_test = './test/labels'    # Replace with your test folder path


# Define the new extension
new_extension = '.txt'

# Copy files to respective folders and change the extension
copy_and_change_extension(train_files, source_folder, destination_train, new_extension)
copy_and_change_extension(val_files, source_folder, destination_val, new_extension)
copy_and_change_extension(test_files, source_folder, destination_test, new_extension)



print("Done Folder architecture labels")

