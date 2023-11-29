import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import csv

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# PIL.Image.open(str("./data/train_images/001.Black_footed_Albatross/1.jpg"))

batch_size = 32
img_height = 180
img_width = 180

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

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

results = []
with open('test_images_path.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_path = "./data/test_images" + row['image_path']
        img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        results.append({'id': row['id'], 'label': predicted_label})
        



