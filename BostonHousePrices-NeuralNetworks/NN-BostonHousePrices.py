# Library 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Data 
df = pd.read_csv("housing.csv", delimiter = r"\s+", header = None)
df.hist()
dataset_x = df.iloc[:,:-1].to_numpy(dtype = "float32")
dataset_y = df.iloc[:,-1].to_numpy(dtype = "float32")
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size = 0.2)

# Scaling
mms = MinMaxScaler()
mms.fit(training_dataset_x)
scaled_training_dataset_x = mms.transform(training_dataset_x)
scaled_test_dataset_x = mms.transform(test_dataset_x)

# Model 
model = Sequential(name = "Boston-Housing-Price")
model.add(Dense(64, activation = "relu", input_dim = training_dataset_x.shape[1], name = "Hidden-1"))
model.add(Dense(64, activation = "relu", name = "Hidden-2"))
model.add(Dense(1, activation= "linear", name = "Output"))
model.summary()
model.compile(optimizer = "rmsprop", loss = "mse", metrics = ["mae"])
hist = model.fit(scaled_training_dataset_x, training_dataset_y, batch_size = 32, epochs = 200, validation_split = 0.2)


# Epoch Graphics
plt.figure(figsize = (15,5))
plt.title("Epoch-Loss Graph", fontsize=14, fontweight = "bold")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(range(0,210,10))
plt.plot(hist.epoch, hist.history["loss"])
plt.plot(hist.epoch, hist.history["val_loss"])
plt.legend(["Loss", "Validation Loss"])
plt.show()

plt.figure(figsize = (15,5))
plt.title("Epoch-Mean Absolute Error Graph", fontsize = 14, fontweight = "bold")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(range(0, 210, 10))
plt.plot(hist.epoch, hist.history["mae"])
plt.plot(hist.epoch, hist.history["val_mae"])
plt.legend(["Mean Absolute Error", "Validation Mean Absolute Error"])
plt.show()

# Test 
eval_result = model.evaluate(scaled_test_dataset_x, test_dataset_y)

for i in range(len(eval_result)):
    print(f"{model.metrics_names[i]}: {eval_result[i]}")
    

# Prediction
predict_data = np.array([[0.06905, 0.0, 2.18, 0, 0.458, 7.147, 54.2, 6.0622, 3, 222.0, 18.7, 396.9, 5.33 ]])
scaled_predict_data = mms.transform(predict_data)
predict_result = model.predict(scaled_predict_data)

for val in predict_result[:,0]:
    print(val)
