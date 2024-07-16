import os
import shutil

base_dir = "D:\\chest_xray\\chest_xray_original"
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

def organize_pneumonia_files(directory):
    pneumonia_dir = os.path.join(directory, 'PNEUMONIA')
    bacterial_dir = os.path.join(directory, 'BACTERIAL')
    viral_dir = os.path.join(directory, 'VIRAL')

    os.makedirs(bacterial_dir, exist_ok=True)
    os.makedirs(viral_dir, exist_ok=True)

    for filename in os.listdir(pneumonia_dir):
        if 'bacteria' in filename:
            shutil.move(os.path.join(pneumonia_dir, filename), os.path.join(bacterial_dir, filename))
        elif 'virus' in filename:
            shutil.move(os.path.join(pneumonia_dir, filename), os.path.join(viral_dir, filename))

    # Remove the now empty PNEUMONIA directory
    os.rmdir(pneumonia_dir)

# Organize train, val, and test directories
organize_pneumonia_files(train_dir)
organize_pneumonia_files(val_dir)
organize_pneumonia_files(test_dir)
