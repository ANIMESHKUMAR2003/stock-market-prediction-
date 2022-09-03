import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

st.title('STOCK TREND PREDICTION')
user_input=st.text_input('Enter stock ticker','AAPL')


user_start = st.text_input('Enter the start yyyy-mm-dd','2010-01-01')
user_end = st.text_input('Enter the end yyyy-mm-dd','2018-12-31')
df = data.DataReader(user_input,'yahoo',user_start,user_end)

#describing data
st.subheader('DATA FROM 2010-2022')
st.write(df.describe())


#visualisations
st.subheader('Closing price vs Time graph')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)




st.subheader('Closing price vs Time graph with 100 MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)



st.subheader('Closing price vs Time graph with 100MA and 200 MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

#splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


#minmax scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))


data_training_array = scaler.fit_transform(data_training)




#load my ml model
model = load_model('keras_aniduuu.h5')



#testing part
past_100_days = data_training.tail(100)

final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.transform(final_df)

x_test = []
y_test = []

for i in range (100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor



#final graph plotting
st.subheader('PREDICTIONS VS ORIGINAL')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test,'b', label = 'original price')
plt.plot(y_predicted,'r', label = 'Predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

