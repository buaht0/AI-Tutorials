# Library 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder

# Data
df = pd.read_csv("auto-mpg.data", delimiter = r"\s+", header = None)
df.hist()
df = df.iloc[:, :-1]
len(df)

# Remove Missing Value
(df.iloc[:,3] == "?").sum()
df = df[df.iloc[:,3]!= "?"]

# One Hot Encoding
ohe = OneHotEncoder(sparse = False, categories = [[1,2,3]])
df[["ohe1", "ohe2", "ohe3"]] = ohe.fit_transform(df.iloc[:,7].to_numpy(dtype = "float").reshape(-1,1))
df.drop(7, axis = 1, inplace = True)


# Train Test Split
dataset_x = df.iloc[:,1:].to_numpy(dtype = "float32")
dataset_y = df.iloc[:,0].to_numpy(dtype = "float32")
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size = 0.2)
training_dataset_x.shape
test_dataset_x.shape

# Scaling
mms = MinMaxScaler() 
mms.fit(training_dataset_x)
scaled_training_dataset_x = mms.transform(training_dataset_x)
scaled_test_dataset_x = mms.transform(test_dataset_x)

# Model
model = Sequential(name = "Auto-MPG")
model.add(Dense(64, activation='relu', input_dim = training_dataset_x.shape[1], name = "Hidden-1"))
model.add(Dense(64, activation = 'relu', name = "Hidden-2"))
model.add(Dense(1, activation = "linear", name = "Output"))
model.summary()
model.compile(optimizer = "rmsprop", loss ="mse", metrics = ["mae"])
hist = model.fit(scaled_training_dataset_x, training_dataset_y, batch_size = 32, epochs = 200, validation_split= 0.2)

# Epoch Graphics
plt.figure(figsize = (15,5))
plt.title("Epoch-Loss Graph", fontsize = 14, fontweight = "bold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(range(0,210, 10))
plt.plot(hist.epoch, hist.history["mae"])
plt.plot(hist.epoch, hist.history["val_mae"])
plt.legend(["Mean Absolute Error", "Validation Mean Absolute Error"])
plt.show()

plt.figure(figsize = (15,5))
plt.title("Epoch-Mean Absolute Graph", fontsize = 14, fontweight = "bold")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(range(0,210,10))
plt.plot(hist.epoch, hist.history["mae"])
plt.plot(hist.epoch, hist.history["val_mae"])
plt.legend(["Mean Absolute Error", "Validation Mean Absolute  Error"])
plt.show()

# Test
eval_result = model.evaluate(scaled_test_dataset_x, test_dataset_y)

for i in range(len(eval_result)):
    print(f"{model.metrics_names[i]}: {eval_result[i]}")
    
# Prediction
predict_data = np.array([[14,81,83,2130,17.9,71,1,0,0],[4,152,91,3164,19.5,70,0,1,0],[3,111,90,2430,27,79,0,0,1]])

scaled_predict_data = mms.transform(predict_data)
predict_result = model.predict(scaled_predict_data)

for val in predict_result[:,0]:
    print(val)
