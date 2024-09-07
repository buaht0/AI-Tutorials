from tensorflow.keras.datasets import reuters
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import regex

# Data 
(training_dataset_x, training_dataset_y),(test_dataset_x, test_dataset_y) = reuters.load_data()

# Vocabulary 
word_dict = reuters.get_word_index()
rev_word_dict = {value: key for key, value in word_dict.items()}
convert_text = lambda text_numbers:" ".join([rev_word_dict[tn-3] for tn in text_numbers if tn > 2])
print(convert_text(training_dataset_x[0]))

# Vectorization
def vectorize(sequence, colsize):
    dataset_x = np.zeros((len(sequence), colsize), dtype = "int8")
    for index, vals in enumerate(sequence):
        dataset_x[index, vals] = 1
    return dataset_x

training_dataset_x = vectorize(training_dataset_x, len(word_dict) + 3)
test_dataset_x = vectorize(test_dataset_x, len(word_dict) + 3)

# Encoder 
ohe_training_dataset_y = vectorize(training_dataset_y, len(np.unique(training_dataset_y)))
ohe_test_dataset_y = vectorize(test_dataset_y, len(np.unique(test_dataset_y)))


# Model 
model = Sequential(name = "Reuters")
model.add(Dense(64, activation = "relu", input_dim = training_dataset_x.shape[1], name = "Hidden-1"))
model.add(Dense(64, activation = "relu", name = "Hidden-2"))
model.add(Dense(len(np.unique(training_dataset_y)), activation = "softmax", name = "Output"))
model.summary()
model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = ["categorical_accuracy"])
hist = model.fit(training_dataset_x, ohe_training_dataset_y, batch_size = 32, epochs = 5, validation_split = 0.2)

# Epoch Graphics
plt.figure(figsize = (15,5))
plt.title("Epoch-Loss Graph", fontsize = 14, fontweight="bold")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(hist.epoch, hist.history["loss"])
plt.plot(hist.epoch, hist.history["val_loss"])
plt.legend(["Loss", "Validation Loss"])
plt.show()


plt.figure(figsize = (15,5))
plt.title("Epoch - Categorical Accuracy Graph", fontsize = 14, fontweight = "bold")
plt.xlabel("Epochs")
plt.ylabel("Categorical Accuracy")
plt.plot(hist.epoch, hist.history["categorical_accuracy"])
plt.plot(hist.epoch, hist.history["val_categorical_accuracy"])
plt.legend(["Categorical accuracy", "Validation categorical accuracy"])
plt.show()


# Test 
eval_result = model.evaluate(test_dataset_x, ohe_test_dataset_y)

for i in range(len(eval_result)):
    print(f"{model.metrics_names[i]}: {eval_result[i]}")
    
# Prediction
texts = ["countries trade volume is expanding. Trade is very important for countries.", "It is known that drinking a lot of coffee is harmful to health. However, according to some, a little coffee is beneficial."]  

predict_data_list = []

for text in texts:
    predict_data_list.append([word_dict[word]+3 for word in regex.findall("[a-zA-Z!0-9']+", text.lower())])
    
predict_data = vectorize(predict_data_list,len(word_dict)+3)

category_list = ["cocoa", "grain", "veg-oil", "earn", "acq", "wheat", "copper", "housing", "money-supply", "coffee", "sugar", "trade", "reserves","ship", "cotton", "carcass","crude","nat-gas","cpi", "money-fx", "interest","gnp", "meal-feed", "alum", "oilseed", "gold", "tin", "strategic-metal", "livestock", "retail","ipi", "iron-steel", "rubber", "heat", "jobs", "lei", "bop", "zinc", "orange", "pet-chem", "dlr", "gas", "silver", "wpi", "hog", "lead"] 

predict_result = model.predict(predict_data)
predict_result = np.argmax(predict_result , axis = 1)

for pr in predict_result:
    print(category_list[pr])
