# Library 
import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Data 
df = pd.read_csv("IMDB-Dataset.csv")

# Vocabulary
cv = CountVectorizer(dtype = "int8")
dataset_x = cv.fit_transform(df["review"]).todense()
dataset_x.shape
dataset_y = np.zeros(len(df), dtype = "int8")
dataset_y[df["sentiment"] == "positive"] = 1
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y)

# Model 
model = Sequential(name = "IMDB") 
model.add(Dense(64, activation = "relu", input_dim = dataset_x.shape[1], name = "Hidden-1"))
model.add(Dense(64, activation = "relu", name = "Hidden-2"))
model.summary()
model.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics = ["binary_accuracy"])
hist = model.fit(training_dataset_x, training_dataset_y, batch_size = 32, epochs = 5, validation_split = 0.2)

# Epoch Graphs
plt.figure(figsize = (15,5))
plt.title("Epoch-Loss Graph", fontsize = 14, fontweight = "bold")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(hist.epoch, hist.history["loss"])
plt.plot(hist.epoch, hist.history["val_loss"])
plt.legend(["Loss", "Validation Loss"])
plt.show()

plt.figure(figsize =(15,5))
plt.title("Epoch-Binary Accuracy Graph", fontsize = 14, fontweight = "bold")
plt.xlabel("Epochs")
plt.ylabel("Binary Accuracy")
plt.plot(hist.epoch, hist.history["binary_accuracy"])
plt.plot(hist.epoch, hist.history["val_binary_accuracy"])
plt.legend(["Binary Accuracy", "Validation Binary Accuracy"])
plt.show()

# Test 
eval_result = model.evaluate(test_dataset_x, test_dataset_y)

for i in range(len(eval_result)):
    print(f"{model.metrics_names[i]}: {eval_result[i]}")
    
# Prediction 
texts = ["the movie was very good. The actors played perfectly. I would recommend it everyone.", "this film is awful. The worst film i have ever seen", "movie was bad very bad"]

transformed_predict_data = cv.transform(texts).todense()
predict_result = model.predict(transformed_predict_data)

for i in range(0, len(predict_result[:,0])):
    if predict_result[i, 0] > 0.5:
        print("Positive")
    else:
        print("Negative")
