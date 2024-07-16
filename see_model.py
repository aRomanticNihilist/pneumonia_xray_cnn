from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Load the trained model
model = load_model('D:\\xray_classification_model\\pneumonia_classification_model.keras')

# Print model summary
model.summary()

# Access and print configuration of the first layer
print(model.layers[0].get_config())

# Save model architecture diagram
plot_model(model, to_file='model.png', show_shapes=True)
# Test accuracy: 0.6661290526390076