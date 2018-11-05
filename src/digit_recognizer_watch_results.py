import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tensorflow.python import keras
from keras.models import Sequential, model_from_json

# Loading the model
print("Abriendo modelo")
json_file = open('./model/model_digit.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("./model/model_digit.h5")

# Evaluating model
print("Compilando...")
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

# Testing
print("Preparando...")
img_rows, img_cols = 28, 28
num_classes = 10

test_file = "./input/digits/test.csv"
raw_data = pd.read_csv(test_file)

# Preparing the data
num_images = raw_data.shape[0]
x_as_array = raw_data.values[:,:]
x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
x = x_shaped_array / 255

# Showing some predictions
print("Predicciones")
random.seed()
for i in range(1,8):
    n = random.randint(0,num_images-1)
    pred = model.predict(np.array([x[n]]))
    num=np.where(pred[0]==max(pred[0]))[0][0]
    print("El numero es: %d" %num)
    plt.imshow(x[n,:,:,0], cmap=plt.cm.binary)
    plt.show()
