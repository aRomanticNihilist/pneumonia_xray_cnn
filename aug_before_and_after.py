import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

image_dir = "D:\\chest_xray\\chest_xray_original\\train\\NORMAL"

all_images = os.listdir(image_dir)

random_image = random.choice(all_images)
image_path = os.path.join(image_dir, random_image)

original_img = load_img(image_path)
original_img_array = img_to_array(original_img)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

img = np.expand_dims(original_img_array, axis=0)

augmented_img = next(train_datagen.flow(img, batch_size=1))[0]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(original_img_array.astype('uint8'))
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(augmented_img)
ax[1].set_title('Augmented Image')
ax[1].axis('off')

plt.show()
