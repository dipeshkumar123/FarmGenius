import tensorflow as tf
import numpy as np

# Create a simple Sequential model that takes an image and outputs 21 classes
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(8, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(21, activation='softmax')
])

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to the flutter assets directory
with open(r'd:\Projects\FarmGenius\app\assets\models\disease_model_quant.tflite', 'wb') as f:
    f.write(tflite_model)

print("Valid TFLite model generated successfully.")
