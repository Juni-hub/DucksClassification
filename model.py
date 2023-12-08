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



# PIL.Image.open(str("./data/train_images/001.Black_footed_Albatross/1.jpg"))

batch_size = 100
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

model = Sequential([
  layers.Resizing(img_height, img_width),
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  data_augmentation,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  # layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  # layers.BatchNormalization(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  # layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  # layers.BatchNormalization(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  # layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  # layers.BatchNormalization(),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  # layers.Conv2D(256, 3, padding='same', activation='relu'),
  # layers.Conv2D(256, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  # layers.Conv2D(256, 3, padding='same', activation='relu'),
  # layers.Conv2D(512, 3, padding='same', activation='relu'),
  # layers.Conv2D(512, 3, padding='same', activation='relu'),
  # layers.MaxPooling2D(),
  # layers.Conv2D(512, 3, padding='same', activation='relu'),
  # layers.Conv2D(512, 3, padding='same', activation='relu'),
  # layers.Conv2D(512, 3, padding='same', activation='relu'),
  # layers.MaxPooling2D(),
  # layers.BatchNormalization(),
  layers.Flatten(),
  layers.Dense(512, activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(350, activation='relu'),
  layers.Dropout(0.4),
  layers.Dense(200, activation='relu'),
  # layers.Dense(512, activation='relu'),
  # layers.Dense(200, activation='relu'),
  layers.Dense(num_classes, activation='softmax' )
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.summary()

epochs=30
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# results = []
# with open('./data/test_images_path.csv', newline='') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         image_path = "./data/test_images" + row['image_path']
#         img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
#         img_array = tf.keras.utils.img_to_array(img)
#         img_array = tf.expand_dims(img_array, 0) # Create a batch

#         predictions = model.predict(img_array)
#         score = tf.nn.softmax(predictions[0])
        
#         results.append({'id': row['id'], 'label': class_names[np.argmax(score)]})
        
# df_results = pd.DataFrame(results)
# df_results.to_csv('image_predictions.csv', index=False)



