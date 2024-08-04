import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model


# Load and preprocess an image
def preprocess_image(image_path):
    img = load_img(
        image_path, target_size=(150, 150)
    )  # Adjust the target size as per your model input
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale as you did for training
    return img_array


# Path to a new image
image_path = (
    "D:\\chest_xray\\chest_xray_original\\test\\BACTERIAL\\person78_bacteria_380.jpeg"
)

# Preprocess the image
img_array = preprocess_image(image_path)

# Load the model
model = load_model("pneumonia_classification_model.keras")

# Make a prediction
prediction = model.predict(img_array)

# Print the prediction
print("Prediction:", prediction)

# Define the class labels in the order of the indices
class_labels = ["BACTERIAL", "NORMAL", "VIRAL"]

# Get the class label
predicted_class_label = class_labels[np.argmax(prediction)]

print("Predicted Class:", predicted_class_label)
