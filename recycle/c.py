import os
import tensorflow as tf
from tensorflow.keras import backend as K

# Function to match image files with annotation files
def get_matched_files(image_dir, annotation_dir, img_suffix='_leftImg8bit.png', ann_suffix='_gtFine_color.png'):
    image_files = {}
    annotation_files = {}
    for dirpath, _, filenames in os.walk(image_dir):
        for filename in filenames:
            if filename.endswith(img_suffix):
                base_name = filename.replace(img_suffix, '')
                full_path = os.path.join(dirpath, filename)
                image_files[base_name] = full_path
    for dirpath, _, filenames in os.walk(annotation_dir):
        for filename in filenames:
            if filename.endswith(ann_suffix):
                base_name = filename.replace(ann_suffix, '')
                full_path = os.path.join(dirpath, filename)
                annotation_files[base_name] = full_path
    matched_images = []
    matched_annotations = []
    for base_name, img_path in image_files.items():
        ann_path = annotation_files.get(base_name)
        if ann_path:
            matched_images.append(img_path)
            matched_annotations.append(ann_path)
        else:
            print(f"Warning: No matching annotation file for {img_path}")
    return matched_images, matched_annotations

# Define the directories
image_dir = 'H:\\debian\\Cityscapes\\leftImg8bit_trainvaltest\\train'
annotation_dir = 'H:\\debian\\Cityscapes\\gtFine_trainvaltest\\train'

# Get matched files
image_files, annotation_files = get_matched_files(image_dir, annotation_dir)

# Clear previous session to free up memory
K.clear_session()

# Load the .keras model
model = tf.keras.models.load_model('cityScapes.keras')

# Prepare a simplified representative dataset
def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(image_files).batch(1).take(10):  # Use fewer samples
        img = tf.io.read_file(input_value[0])
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [64, 128])
        img = tf.cast(img, tf.float32) / 255.0
        yield [tf.expand_dims(img, axis=0)]

# Convert the model using a concrete function
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 64, 128, 3], model.inputs[0].dtype))

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Enable verbose logging
tf.get_logger().setLevel('INFO')

# Perform the conversion
tflite_model = converter.convert()

# Save the converted and quantized model to a .tflite file
tflite_model_path = 'cityScapes_model_quant.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Quantized TensorFlow Lite model saved to {tflite_model_path}")
