import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

# Initial variables
img_rows, img_cols = 28, 28
num_classes = 10

train_file = "./input/digits/train.csv"
raw_data = pd.read_csv(train_file)


# Model counting
model_count = 0
with open('./evaluations/model_count.txt', 'r') as count_file:
    model_count = int(count_file.read())

with open('./evaluations/model_count.txt', 'w') as count_file:
    count_file.write(str(model_count+1))


# Preparing the data
y = keras.utils.to_categorical(raw_data.label, num_classes)
num_images = raw_data.shape[0]
x_as_array = raw_data.values[:,1:]
x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
x = x_shaped_array / 255

x_train, x_val_, y_train, y_val_ = train_test_split(
    x, y, test_size=0.3)
x_val, x_test, y_val, y_test = train_test_split(
    x_val_, y_val_, test_size=0.3)


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

generator = ImageDataGenerator(zoom_range = 0.2,
                            height_shift_range = 0.2,
                            width_shift_range = 0.2,
                            rotation_range = 15)

# Annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)



# Training the model
history = model.fit_generator(generator.flow(x_train,y_train,batch_size=75),
          epochs=5,
	      steps_per_epoch = 350,
	      verbose = 1,
          validation_data = generator.flow(x_val,y_val,batch_size=63),
	      validation_steps = 200,
	      callbacks=[learning_rate_reduction])

scores = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Saving the model
model_json = model.to_json()
with open("./model/model_digit" + str(model_count) + ".json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("./model/model_digit" + str(model_count) + ".h5")

# Saving accuracies
with open('./evaluations/model' + str(model_count) + '.txt', 'w') as accuracy_file:
    accuracy_file.write('Accuracies:\n')
    accuracy_file.write(str(history.history['acc']) + '\n')
    accuracy_file.write('Accuracies from Validation:\n')
    accuracy_file.write(str(history.history['val_acc']) + '\n')
    accuracy_file.write('Losses:\n')
    accuracy_file.write(str(history.history['loss']) + '\n')
    accuracy_file.write('Losses from Validation:\n')
    accuracy_file.write(str(history.history['val_loss']))
    accuracy_file.write('\nmodel' + str(model_count) + ': ' + str(scores[1]))


# Plotting accuracy history
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('accuracies')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'])
plt.show()

# Plotting loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('losses')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'])
plt.show()
