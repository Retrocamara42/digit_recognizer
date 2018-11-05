import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten, Conv2D, Dropout

img_rows, img_cols = 28, 28
num_classes = 10

train_file = "./input/digits/train.csv"
raw_data = pd.read_csv(train_file)

# Preparing the data
y = keras.utils.to_categorical(raw_data.label, num_classes)
num_images = raw_data.shape[0]
x_as_array = raw_data.values[:,1:]
x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
x = x_shaped_array / 255

# The model
model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(40, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

# Training the model
model.fit(x, y,
          batch_size=64,
          epochs=10,
          validation_split = 0.3)

scores = model.evaluate(x, y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Saving the model
model_json = model.to_json()
with open("./model/model_digit.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("./model/model_digit.h5")
