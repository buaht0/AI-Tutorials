# Packages
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
from tensorflow.keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# Const. variable
TEXT_SIZE = 250

# Data
df = pd.read_csv('IMDB Dataset.csv')
cv = CountVectorizer()
cv.fit(df['review'])
dataset_y = np.zeros(len(df), dtype='int8')
dataset_y[df['sentiment'] == 'positive'] = 1
text_vectors = [[cv.vocabulary_[word] for word in re.findall(r'(?u)\b\w\w+\b', text.lower())] for text in df['review']]
dataset_x = pad_sequences(text_vectors, TEXT_SIZE, padding='post', truncating='post')
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y)

# Model
model = Sequential(name='IMDBEmbbedding')
model.add(Input((TEXT_SIZE,), name='Input'))
model.add(Embedding(len(cv.vocabulary_), 64, input_length=TEXT_SIZE, name='Embedding'))
model.add(Conv1D(64, 3, padding='same', activation='relu', name='Conv1D-1'))
model.add(MaxPooling1D(2, name='Pooling-1'))
model.add(Conv1D(64, 3, padding='same', activation='relu', name='Conv1D-2'))
model.add(MaxPooling1D(2, name='Pooling-2'))
model.add(Flatten(name='Flatten'))
model.add(Dense(128, activation='relu', name='Dense-1'))
model.add(Dense(128, activation='relu', name='Dense-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
esc = EarlyStopping(patience=5, restore_best_weights=True)
hist = model.fit(training_dataset_x, training_dataset_y, epochs=10, batch_size=32, validation_split=0.2, callbacks=[esc])

# Epoch-Loss Graph
plt.figure(figsize=(15, 5))
plt.title('Epoch Loss Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0, len(hist.epoch), 20))
plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

# Epoch - Accuracy Graph
plt.figure(figsize=(15, 5))
plt.title('Epoch Binary Accuracy Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Binary Accuracy')
plt.xticks(range(0, len(hist.epoch), 20))
plt.plot(hist.epoch, hist.history['binary_accuracy'])
plt.plot(hist.epoch, hist.history['val_binary_accuracy'])
plt.legend(['Binary Accuracy', 'Validation Binary Accuracy'])
plt.show()

# Evaluate
eval_result = model.evaluate(test_dataset_x, test_dataset_y)
for i in range(len(eval_result)):
    print(f' {model.metrics_names[i]}: {eval_result[i]}')

# Predict
predict_texts = ['the movie was very good. The actors played perfectly. I would recommend it to everyone.',
         'this film is awful. The worst film i have ever seen']
predict_vectors = [[cv.vocabulary_[word] for word in re.findall(r'(?u)\b\w\w+\b', text.lower())] for text in predict_texts]
predict_data = pad_sequences(predict_vectors, TEXT_SIZE, padding='post', truncating='post')
predict_result = model.predict(predict_data)

for result in predict_result[:, 0]:
    if result > 0.5:
        print('Positive')
    else:
        print('Negative')