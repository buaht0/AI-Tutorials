# Packages
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

# Data
x_lst = []
y_lst = []

for path in glob.glob('cifar-10-batches-py/data_batch_*'):
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
        x_lst.append(d[b'data'])
        y_lst.append(d[b'labels'])


training_dataset_x = np.concatenate(x_lst)
training_dataset_y = np.concatenate(y_lst)

print(training_dataset_x.shape) # (50000, 3072)
print(training_dataset_y.shape) # (50000,)

with open('cifar-10-batches-py/test_batch', 'rb') as f:
    d = pickle.load(f, encoding='bytes')
    test_dataset_x = d[b'data']
    test_dataset_y = np.array(d[b'labels'])

print(test_dataset_x.shape) # (10000, 3072)
print(test_dataset_y.shape) # (10000,)

# Feature Scaling
training_dataset_x = training_dataset_x / 255
test_dataset_x = test_dataset_x / 255

# Transpose
training_dataset_x = training_dataset_x.reshape(-1, 3, 32, 32)
training_dataset_x = np.transpose(training_dataset_x, [0, 2, 3, 1])
test_dataset_x = test_dataset_x.reshape(-1, 3, 32, 32)
test_dataset_x = np.transpose(test_dataset_x, [0, 2, 3, 1])

# One hot encoding
ohe_training_dataset_y = to_categorical(training_dataset_y)
ohe_test_dataset_y = to_categorical(test_dataset_y)

# Show sample image
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(5, 30))
for i in range(30):
    plt.subplot(10, 3, i + 1)
    plt.title(class_names[training_dataset_y[i]], pad=10)
    plt.imshow(training_dataset_x[i])
plt.show()


# Model
model = Sequential(name='CIFAR-10')
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', name='Conv2D-1'))
model.add(MaxPooling2D(name='Pooling-1'))
model.add(Conv2D(64, (3, 3), activation='relu', name='Conv2D-2'))
model.add(MaxPooling2D(name='Pooling-2'))
model.add(Conv2D(128, (3, 3), activation='relu', name='Conv2D-3'))
model.add(MaxPooling2D(name='Pooling-3'))
model.add(Flatten(name='Flatten'))
model.add(Dense(512, activation='relu', name='Hidden-1'))
model.add(Dense(256, activation='relu', name='Hidden-2'))
model.add(Dense(10, activation='softmax', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
hist = model.fit(training_dataset_x, ohe_training_dataset_y, epochs=20, batch_size=32, validation_split=0.2)

# Epoch-Loss Graph
plt.figure(figsize=(15, 5))
plt.title('Epoch-Loss Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0, 210, 10))
plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

# Epoch - Accuracy Graph
plt.figure(figsize=(15, 5))
plt.title('Epoch-Categorical Accuracy Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Categorical Accuracy')
plt.xticks(range(0, 210, 10))
plt.plot(hist.epoch, hist.history['categorical_accuracy'])
plt.plot(hist.epoch, hist.history['val_categorical_accuracy'])
plt.legend(['Categorical Accuracy', 'Validation Categorical Accuracy'])
plt.show()

# Evaluate
eval_result = model.evaluate(test_dataset_x, ohe_test_dataset_y)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

# Predict
for path in glob.glob('test-images/*.jpg'):
    image_data = plt.imread(path)
    scaled_image_data = image_data / 255
    predict_result = model.predict(scaled_image_data.reshape(1, 32, 32, 3))
    result = np.argmax(predict_result)
    print(f'{path}: {class_names[result]}')