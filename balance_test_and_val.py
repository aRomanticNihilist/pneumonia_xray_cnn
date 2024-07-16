import os
import shutil
from random import sample

def move_images(source_dir, target_dir, num_images_to_move):
    images = os.listdir(source_dir)
    images_to_move = sample(images, num_images_to_move)
    
    for image in images_to_move:
        shutil.move(os.path.join(source_dir, image), os.path.join(target_dir, image))

def balance_folders(test_dir, val_dir, class_names):
    for class_name in class_names:
        test_class_dir = os.path.join(test_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        
        test_images = os.listdir(test_class_dir)
        val_images = os.listdir(val_class_dir)
        
        num_test_images = len(test_images)
        num_val_images = len(val_images)
        
        if num_test_images > num_val_images:
            num_images_to_move = (num_test_images - num_val_images) // 2
            move_images(test_class_dir, val_class_dir, num_images_to_move)
        elif num_val_images > num_test_images:
            num_images_to_move = (num_val_images - num_test_images) // 2
            move_images(val_class_dir, test_class_dir, num_images_to_move)

# Define the directories
test_dir = "D:\\chest_xray\\chest_xray_original\\test"
val_dir = "D:\\chest_xray\\chest_xray_original\\val"

# Define the class names
class_names = ['NORMAL', 'BACTERIAL', 'VIRAL']

# Balance the folders
balance_folders(test_dir, val_dir, class_names)
