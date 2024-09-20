# Packages
import numpy as np
import pandas as pd
import glob
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Data
(training_dataset_x, training_dataset_y), (test_dataset_x, test_dataset_y) = mnist.load_data()

# Draw mnist data examples
plt.figure(figsize=(20, 40))
for i in range(50):
    plt.subplot(10, 5, i + 1)
    plt.title(str(training_dataset_y[i]), fontsize=14)
    plt.imshow(training_dataset_x[i], cmap='gray')
plt.show()

# One-hot encoding
ohe_training_dataset_y = to_categorical(training_dataset_y)
ohe_test_dataset_y = to_categorical(test_dataset_y)

# Min-max scaling
training_dataset_x = training_dataset_x.reshape(-1, 784) / 255
test_dataset_x = test_dataset_x.reshape(-1, 784) / 255

# Model
model = Sequential(name='MNIST')
model.add(Dense(256, activation='relu', input_dim=784, name='Hidden-1'))
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(10, activation='softmax', name='Output'))
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train
hist = model.fit(training_dataset_x, ohe_training_dataset_y, epochs=20, batch_size=32, validation_split=0.2)

# Epoch Graphs
plt.figure(figsize=(15, 5))
plt.title('Epoch-Loss Graph', fontsize=14, fontweight='bold')
plt.xticks(range(0, 210, 10))
plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(15, 5))
plt.title('Epoch-Categorical Binary Accuracy Graph', fontsize=14, fontweight='bold')
plt.xticks(range(0, 210, 10))
plt.plot(hist.epoch, hist.history['categorical_accuracy'])
plt.plot(hist.epoch, hist.history['val_categorical_accuracy'])
plt.legend(['Categorical Binary Accuracy', 'Validation Categorical Binary Accuracy'])
plt.show()

# Test
eval_result = model.evaluate(test_dataset_x, ohe_test_dataset_y)
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

# Predict
for path in glob.glob('test-images/*.jpg'):
    image_data = plt.imread(path)
    gray_scaled_image_data = np.average(image_data, axis=2, weights=[0.3, 0.59, 0.11])
    gray_scaled_image_data = gray_scaled_image_data / 255
    predict_result = model.predict(gray_scaled_image_data.reshape(1, 784))
    result = np.argmax(predict_result)
    print(f'{path}:{result}')