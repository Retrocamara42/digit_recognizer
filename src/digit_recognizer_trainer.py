import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

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

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.3)

# The model
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(MaxPool2D(pool_size=(2,2),strides=(1,1)))
model.add(Dropout(0.2))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(1,1)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

generator = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10)

# Annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)



# Training the model
model.fit_generator(generator.flow(x_train,y_train,batch_size=75),
          epochs=25,
	  steps_per_epoch = 392,
	  verbose = 1,
          validation_data = generator.flow(x_val,y_val,batch_size=63),
	  validation_steps = 200,
	  callbacks=[learning_rate_reduction])

scores = model.evaluate(x, y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Saving the model
model_json = model.to_json()
with open("./model/model_digit.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("./model/model_digit.h5")

