import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import csv
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.applications import EfficientNetB0


# PIL.Image.open(str("./data/train_images/001.Black_footed_Albatross/1.jpg"))

batch_size = 50
img_height = 224
img_width = 224

training_data_dir = "./data/train_images/"

train_ds = tf.keras.utils.image_dataset_from_directory(
  training_data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  training_data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

normalization_layer = layers.Rescaling(1./255)

num_classes = len(class_names)

data_augmentation = keras.Sequential(
  [
    
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.15),
    layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
    # layers.RandomTranslation(height_factor=(-0.03, 0.03), width_factor=(-0.03, 0.03)),
    layers.RandomContrast((0.1, 0.3)),
    # layers.RandomBrightness((-0.3, 0.3)),
   
  ]
)

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
# Freeze the base model
base_model.trainable = False

model = Sequential([
  layers.Input(shape=(img_height, img_width, 3)),
  base_model,
  layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.0001)),
  layers.Dropout(0.5),
  layers.GlobalAveragePooling2D(),
  layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
  layers.Dropout(0.5),
  layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=10
# Fine-tune the model
fine_tune_epochs = 10
total_epochs = epochs + fine_tune_epochs

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=total_epochs,
  initial_epoch=epochs
)

results = []
with open('./data/test_images_path.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_path = "./data/test_images" + row['image_path']
        img_array = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img_array)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        # Use the correct preprocessing for EfficientNetB0
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        results.append({'id': row['id'], 'label': class_names[np.argmax(score)]})
        
df_results = pd.DataFrame(results)
df_results.to_csv('image_predictions.csv', index=False)



