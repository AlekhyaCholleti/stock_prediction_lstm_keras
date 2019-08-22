import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import load_model

dataset_train = pd.read_csv("/home/welcome/Downloads/ml_ai_dl/google_stockprice_data/Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values
#print(training_set)

#feature scaling
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

#creating a dataset with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
	X_train.append(training_set_scaled[i-60:i, 0])
	y_train.append(training_set_scaled[i, 0])
#convert the lists to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)
#print(X_train.shape)
#print(y_train.shape)

#reshaping into (no.of samples, seq_len, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

def training():
	#building the RNN
	regressor = Sequential()
	regressor.add(LSTM(units=50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
	regressor.add(Dropout(0.2))
	regressor.add(LSTM(units = 50, return_sequences = True))
	regressor.add(Dropout(0.2))
	regressor.add(LSTM(units = 50, return_sequences = True))
	regressor.add(Dropout(0.2))
	regressor.add(LSTM(units = 50))
	regressor.add(Dropout(0.2))
	regressor.add(Dense(units = 1))
	#compiling the rnn
	regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
	#fitting the rnn to the training set
	regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

	regressor.save('/home/welcome/python_files/project/stockpred_singlefeature.h5')
	del regressor

#training()	

def testing():
	regressor = load_model('/home/welcome/python_files/project/stockpred_singlefeature.h5')
	#making the predictions and visualising the results
	dataset_test = pd.read_csv("/home/welcome/Downloads/ml_ai_dl/google_stockprice_data/Google_Stock_Price_Test.csv")
	real_stock_price = dataset_test.iloc[:, 1:2].values

	# Getting the predicted stock price of 2017
	dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
	inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
	#print(inputs.shape)
	inputs = inputs.reshape(-1,1)
	#print(inputs.shape)
	inputs = sc.transform(inputs)
	X_test = []
	for i in range(60, 80):
	    X_test.append(inputs[i-60:i, 0])
	X_test = np.array(X_test)
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
	predicted_stock_price = regressor.predict(X_test)
	predicted_stock_price = sc.inverse_transform(predicted_stock_price)
	#print predicted_stock_price
	#Visualising the results
	plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
	plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
	plt.title('Google Stock Price Prediction')
	plt.xlabel('Time')
	plt.ylabel('Google Stock Price')
	plt.legend()
	plt.show()

testing()	