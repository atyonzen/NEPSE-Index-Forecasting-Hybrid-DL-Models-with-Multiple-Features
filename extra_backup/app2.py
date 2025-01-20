import numpy as np
import time as tm
import datetime as dt
import tensorflow as tf
import pandas as pd

# Data preparation
from yahoo_fin import stock_info as yf
from sklearn.preprocessing import MinMaxScaler
from collections import deque

# AI
from keras.src.models import Sequential
from keras.src.layers import Dense, LSTM, Dropout

# Graphics library
import matplotlib.pyplot as plt


# SETTINGS

# Window size or the sequence length, 7 (1 week)
# N_STEPS = 7
N_STEPS = 14

# Lookup steps, 1 is the next day, 3 = after tomorrow
LOOKUP_STEPS = [1, 2, 3, 4, 5, 6, 7]
# LOOKUP_STEPS = [1, 2, 3]

# Stock ticker, GOOGL
STOCK = 'NEPSE Index'

# Current date
# date_now = tm.strftime('%Y-%m-%d')
# date_3_years_back = (dt.date.today() - dt.timedelta(days=1104)).strftime('%Y-%m-%d')
# print(date_now, date_3_years_back)

# LOAD DATA 
# from yahoo_fin 
# for 1104 bars with interval = 1d (one day)
# init_df = yf.get_data(
#     STOCK, 
#     start_date=date_3_years_back, 
#     end_date=date_now, 
#     interval='1d')

# remove columns which our neural network will not use
# init_df = init_df.drop(['open', 'high', 'low', 'adjclose', 'ticker', 'volume'], axis=1)
# create the column 'Date' based on index column
# init_df['Date'] = init_df.index

# reac csv file
init_df = pd.read_csv("nepsealpha3.csv")
# remove columns which our neural network will not use
init_df = init_df.drop(['Symbol', 'Open', 'High', 'Low', 'Percent Change', 'Volume'], axis=1)
# create the column 'Date' based on index column
# init_df['Date'] = init_df.index

# print(init_df)
# exit()

# Scale data for ML engine
scaler = MinMaxScaler()
init_df['Close'] = scaler.fit_transform(np.expand_dims(init_df['Close'].values, axis=1))

# df = init_df.copy()
# df['future'] = df['Close'].shift(-2)
# last_sequence = np.array(df[['Close']].tail(2))
# df.dropna(inplace=True)
# sequence_data = []
# sequences = deque(maxlen=N_STEPS)
# for entry, target in zip(df[['Close'] + ['Date']].values, df['future'].values):
#     sequences.append(entry)
#     if len(sequences) == N_STEPS:
#         sequence_data.append([np.array(sequences), target])

# print(tuple(zip(df[['Close'] + ['Date']].values, df['future'].values)))
# print(df)
# last_sequence = list([s[:len(['Close'])] for s in sequences]) + list(last_sequence)
# last_sequence = np.array(last_sequence).astype(np.float32)
# print(sequence_data)
# print(sequences)
# print(last_sequence)
# exit()
def PrepareData(days):
  df = init_df.copy()
  df['future'] = df['Close'].shift(-days)
  last_sequence = np.array(df[['Close']].tail(days))
  df.dropna(inplace=True)
  sequence_data = []
  # pop outs from right if append items more than maxlen
  sequences = deque(maxlen=N_STEPS)
  for entry, target in zip(df[['Close'] + ['Date']].values, df['future'].values):
      sequences.append(entry)
      if len(sequences) == N_STEPS:
          sequence_data.append([np.array(sequences), target])

  last_sequence = list([s[:len(['Close'])] for s in sequences]) + list(last_sequence)
  last_sequence = np.array(last_sequence).astype(np.float32)

  # construct the X's and Y's
  X, Y = [], []
  for seq, target in sequence_data:
      X.append(seq)
      Y.append(target)

  # convert to numpy arrays
  X = np.array(X)
  Y = np.array(Y)

  return df, last_sequence, X, Y


def GetTrainedModel(x_train, y_train):
  model = Sequential()
  model.add(LSTM(60, return_sequences=True, input_shape=(N_STEPS, len(['Close']))))
  model.add(Dropout(0.01))
  model.add(LSTM(120, return_sequences=False))
  model.add(Dropout(0.01))
  model.add(Dense(20))
  model.add(Dense(1))

  BATCH_SIZE = 8
  EPOCHS = 80

  model.compile(loss='mean_squared_error', optimizer='adam')

  model.fit(x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1)

  # model.summary()

  return model


# GET PREDICTIONS
predictions = []

for step in LOOKUP_STEPS:
  df, last_sequence, x_train, y_train = PrepareData(step)
  # print(x_train)
  # exit()
  x_train = x_train[:, :, :len(['Close'])].astype(np.float32)

  model = GetTrainedModel(x_train, y_train)

  last_sequence = last_sequence[-N_STEPS:]
  last_sequence = np.expand_dims(last_sequence, axis=0)
  prediction = model.predict(last_sequence)
  predicted_price = scaler.inverse_transform(prediction)[0][0]
  # print(predicted_price)
  # exit()
  predictions.append(round(float(predicted_price), 2))


# Execute model for the whole history range
copy_df = init_df.copy()
y_predicted = model.predict(x_train)
y_predicted_transformed = np.squeeze(scaler.inverse_transform(y_predicted))
# y_predicted_transformed = np.squeeze(y_predicted)
# print(y_predicted_transformed.shape)
first_seq = scaler.inverse_transform(np.expand_dims(y_train[:6], axis=1))
last_seq = scaler.inverse_transform(np.expand_dims(y_train[-3:], axis=1))
y_predicted_transformed = np.append(first_seq, y_predicted_transformed)
y_predicted_transformed = np.append(y_predicted_transformed, last_seq)
copy_df['predicted_close'] = pd.Series(y_predicted_transformed)
# copy_df['predicted_close'] = y_predicted_transformed
copy_df.dropna(inplace=True)

# Reversing the scaling operation to get the original values back
original_data = scaler.inverse_transform(init_df[['Close']])
# Convert to a pandas series if needed, similar to the original format
reversed_close = pd.Series(original_data.flatten())
copy_df['Close'] = reversed_close

# Add predicted results to the table
date_now = dt.date.today()
date_tomorrow = dt.date.today() + dt.timedelta(days=1)
date_after_tomorrow = dt.date.today() + dt.timedelta(days=2)

copy_df['date_now'] = pd.Series([predictions[0], f"{date_now}", 0, 0])
copy_df['date_tomorrow'] = pd.Series([predictions[1], f"{date_tomorrow}", 0, 0])
copy_df['date_after_tomorrow'] = pd.Series([predictions[2], f"{date_after_tomorrow}", 0, 0])

# Result chart
plt.style.use(style='ggplot')
plt.figure(figsize=(16,10))
plt.plot(copy_df['Close'][-150:].head(147))
plt.plot(copy_df['predicted_close'][-150:].head(147), linewidth=1, linestyle='dashed')
plt.plot(copy_df['Close'][-150:].tail(7))
plt.xlabel('days')
plt.ylabel('price')
plt.legend([f"Actual {STOCK}", 
            f"Predicted {STOCK}",
            f"Predicted {STOCK} 3 days"])
plt.show()

# Let's preliminary see our data on the graphic
# plt.style.use(style='ggplot')
# plt.figure(figsize=(16,10))
# plt.plot(init_df['Close'][-200:])
# plt.xlabel("days")
# plt.ylabel("price")
# plt.legend([f'Actual price for {STOCK}"])
# plt.show()