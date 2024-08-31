# Library
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler


# Data 
df = pd.read_csv("Iris.csv")
dataset_x = df.iloc[:,1:-1].to_numpy(dtype = "float32")
dataset_y = pd.get_dummies(df.iloc[:,-1]).to_numpy(dtype = "int8")
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size = 0.2)
df.iloc[:,1:-1].hist()

# Scaling
ss = StandardScaler()
ss.fit(training_dataset_x)
scaled_training_dataset_x =  ss.transform(training_dataset_x)
scaled_test_dataset_x = ss.transform(test_dataset_x)

# Model
model = Sequential(name = "Iris")
model.add(Dense(64, activation = "relu", input_dim = dataset_x.shape[1], name = "Hidden-1"))
model.add(Dense(64, activation = "relu", name = "Hidden-2"))
model.add(Dense(3, activation="softmax", name = "Output"))
model.summary()
model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = ["categorical_accuracy"])
hist = model.fit(scaled_training_dataset_x, training_dataset_y, batch_size = 32, epochs = 200, validation_split = 0.2)


# Epoch Graphics
plt.figure(figsize = (15,5))
plt.title("Epoch-Loss Graph", fontsize = 14, fontweight="bold")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(range(0,210,10))
plt.plot(hist.epoch, hist.history["loss"])
plt.plot(hist.epoch, hist.history["val_loss"])
plt.legend(["Loss","Validation Loss"])


plt.figure(figsize = (15,5))
plt.title("Epoch-Categorical Accuracy Graph", fontsize = 14, fontweight = "bold")
plt.xlabel("Epochs")
plt.ylabel("Categorical Accuracy")
plt.xticks(range(0, 210, 10))
plt.plot(hist.epoch, hist.history["categorical_accuracy"])
plt.plot(hist.epoch, hist.history["val_categorical_accuracy"])
plt.legend(["Categorical Accuracy", "Validation Categorical Accuracy"])
plt.show()

# Test
eval_result = model.evaluate(scaled_test_dataset_x, test_dataset_y)
for i in range(len(eval_result)):
    print(f"{model.metrics_names[i]}:{eval_result[i]}")

# Prediction
predict_data = pd.read_csv("iris-predict.csv", header = None).to_numpy(dtype = "float32")
scaled_predict_data = ss.transform(predict_data)
predict_result = model.predict(scaled_predict_data)
predict_result_names = df.iloc[:,-1].unique()[np.argmax(predict_result, axis = 1)]
print(predict_result_names)
