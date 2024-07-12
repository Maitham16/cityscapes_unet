import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('cityScapes.keras')

# Convert the model to TensorFlow Lite format with quantization for Edge TPU
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('cityScapes.tflite', 'wb') as f:
    f.write(tflite_model)