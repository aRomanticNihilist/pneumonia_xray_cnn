import numpy as np
import os
import argparse
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, LambdaCallback
import logging
from sklearn.utils import class_weight

# Argument parser for log file
parser = argparse.ArgumentParser()
parser.add_argument('--log_file', type=str, required=True, help='Log file path')
args = parser.parse_args()

# Setup logging
logging.basicConfig(filename=args.log_file, level=logging.INFO, format='%(message)s')

# Define directories
base_dir = "D:\\chest_xray\\chest_xray_original"
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Create ImageDataGenerator instances
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

val_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load the datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Learning rate scheduler and early stopping
initial_learning_rate = 0.0001

def lr_schedule(epoch):
    return initial_learning_rate * 0.95 ** epoch

optimizer = Adam(learning_rate=initial_learning_rate)
lr_scheduler = LearningRateScheduler(lr_schedule)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Logging callback for training and validation metrics
log_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: logging.info(f"Epoch {epoch+1}: loss={logs['loss']}, accuracy={logs['accuracy']}, val_loss={logs['val_loss']}, val_accuracy={logs['val_accuracy']}")
)

# Compile model
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    class_weight=class_weights,
    callbacks=[lr_scheduler, early_stopping, log_callback]
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
logging.info(f'Test accuracy: {test_acc}')
logging.info(f'Test loss: {test_loss}')

print('Test accuracy: ', test_acc)
print('Test loss: ', test_loss)

# Save the model
model.save('pneumonia_classification_model.keras')

# # Plotting the training and validation loss
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 4))

# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()

# # Plotting the training and validation accuracy
# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()

# plt.show()
