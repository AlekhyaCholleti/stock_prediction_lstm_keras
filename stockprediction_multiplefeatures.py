import pandas as pd
import numpy as np
from matplotlib  import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras import optimizers
from keras.models import load_model

df_ge = pd.read_csv("/home/welcome/Downloads/ml_ai_dl/aal.txt", usecols = [1, 2, 3, 4, 5])


plt.figure()
plt.plot(df_ge["Open"])
plt.plot(df_ge["High"])
plt.plot(df_ge["Low"])
plt.plot(df_ge["Close"])
plt.title('GE stock price history')
plt.ylabel('Price (USD)')
plt.xlabel('Days')
plt.legend(['Open','High','Low','Close'], loc='upper left')
#plt.show()

plt.figure()
plt.plot(df_ge["Volume"])
plt.title('GE stock volume history')
plt.ylabel('Volume')
plt.xlabel('Days')
#plt.show()

#print(df_ge.isna().sum())

train_cols = ["Open","High","Low","Close","Volume"]
df_train, df_test = train_test_split(df_ge, train_size=0.8, test_size=0.2, shuffle=False)


min_max_scaler = MinMaxScaler(feature_range=(0,1))
x_train = min_max_scaler.fit_transform(df_train.loc[:,train_cols])
x_test = min_max_scaler.transform(df_test.loc[:,train_cols])


def build_data(mat, y_col_index):
	# y_col_index is the index of column that would act as output column
	# total number of batches would be len(mat) - TIME_STEPS
	x = []
	y = []
	TIME_STEPS = 60
	noof_features = mat.shape[1]
	#x = np.zeros((TIME_STEPS, noof_features))
	#y = np.zeros((1,))
	#print(y)
	
	#print(len(mat))
	for i in range(len(mat)-TIME_STEPS):
		x.append(mat[i:TIME_STEPS+i])
		#print(len(x[i]))
		y.append(mat[TIME_STEPS+i,y_col_index])
		#print(y[i])
	x = np.array(x)
	y = np.array(y)	

	print("shape of i/o ",x.shape,y.shape)
	return x,y	
#print(build_data(x_train,3))  

def trim_dataset(mat,batch_size):
	no_of_rows_drop = mat.shape[0]%batch_size
	if(no_of_rows_drop>0):
		return mat[:-no_of_rows_drop]
	else:
		return mat		

BATCH_SIZE = 20
lr = 0.001
x_tr,y_tr = build_data(x_train,3)
x_tr = trim_dataset(x_tr,BATCH_SIZE)
y_tr = trim_dataset(y_tr,BATCH_SIZE)

x_te,y_te = build_data(x_test,3)
x_te = trim_dataset(x_te,BATCH_SIZE)
y_te = trim_dataset(y_te,BATCH_SIZE)
#print(x_tr.shape)
#print(y_te.shape)

def training():
	lstm_model = Sequential()
	lstm_model.add(LSTM(units=100, input_shape=(x_tr.shape[1], x_tr.shape[2]), dropout=0.0, recurrent_dropout=0.0,kernel_initializer='random_uniform'))
	lstm_model.add(Dropout(0.5))
	lstm_model.add(Dense(20,activation='relu'))
	lstm_model.add(Dense(1,activation='sigmoid'))
	optimizer = optimizers.RMSprop(lr=lr)
	lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)

	lstm_model.fit(x_tr, y_tr, epochs=100, batch_size=BATCH_SIZE)
	lstm_model.save('/home/welcome/python_files/project/stockpred_multifeatures.h5')
	del lstm_model

def testing():
	lstm_model = load_model('/home/welcome/python_files/project/stockpred_multifeatures.h5')
	predicted_y = lstm_model.predict(x_te)
	#print(predicted_y.shape)
	#predicted_y = min_max_scaler.inverse_transform(predicted_y)
	#print(y_te.shape)
	predicted_y = predicted_y.flatten()
	predicted_y = (predicted_y * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3] 
	true_y = (y_te* min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3] 
	#print(predicted_y)
	#print(true_y)

	plt.figure()
	plt.plot(predicted_y)
	plt.plot(true_y)
	plt.title('Prediction vs Real Stock Price')
	plt.ylabel('Price')
	plt.xlabel('Days')
	plt.legend(['Prediction', 'Real'], loc='upper left')
	plt.show()

testing()	