# Import necessary libraries
import numpy as np
import pandas as pd
import time as tm
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from yahoo_fin import stock_info as yf
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# reac csv file
df = pd.read_csv("nepsealpha3.csv")
# remove columns which our neural network will not use
df = df.drop(['Symbol', 'Date', 'Open', 'High', 'Low', 'Percent Change', 'Volume'], axis=1)

# Normalize the Close price column
scaler = MinMaxScaler()
df["Close"] = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

# Prepare the dataset
# look_back is the number of previous days' prices to consider for prediction
def prepare_data(data, look_back=1):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + look_back
        # handle last item which may be like 228+1 > 229-1
        if end_ix > len(data) - 1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def create_train_test_split(data, look_back = 7, train_test_split = 0.8):

    # Prepare the training data
    X, y = prepare_data(data, look_back)
    
    # Split the data into training and testing sets
    training_size = int(len(X) * train_test_split)
    X_train, X_test, y_train, y_test = X[:training_size], X[training_size:], y[:training_size], y[training_size:]
    # print(X_train.shape)
    # print(X_test.shape)
    # (172, 14)
    # (43, 14)

    # Reshape the input data for LSTM
    # Reshape input data into [samples, timesteps, features]
    X_train = X_train.reshape(-1, look_back, 1)
    X_test = X_test.reshape(-1, look_back, 1)

    return X_train, X_test, y_train, y_test


# Define the LSTM model
def create_stacked_lstm_model():
    model = Sequential()
    model.add(LSTM(60, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.001))
    model.add(LSTM(120, return_sequences=False))
    model.add(Dropout(0.001))
    model.add(Dense(20))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# Create train_test_split
look_back = 7  # Number of time steps to consider for prediction
data = df["Close"].values
X_train, X_test, y_train, y_test = create_train_test_split(data=data)

# Create model instance
model = create_stacked_lstm_model()
# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=8, verbose=1)

# Predict using the trained model
# Predict for x_train, X_test
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate RMSE performance matrics
# Train data RMSE
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train, train_predict))

# Test data RMSE
math.sqrt(mean_squared_error(y_test, test_predict))


# Plot the predicted prices

# Shift train predictions for plotting
trainPredictPlot = np.empty_like(df)
# Inserts nan values to empty df like dataframe
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back: len(train_predict) + look_back] = train_predict

# Shift test predictions for plotting
testPredictPlot = np.empty_like(df)
# Inserts nan values to empty df like dataframe
testPredictPlot[:, :] = np.nan
testPredictPlot[ len(train_predict) + look_back : len(df), :] = test_predict

# Predict for coming future
future_steps = 164  # Number of days to predict
future_data = data[-look_back:].reshape(-1, look_back, 1)
predicted_prices = []
for i in range(future_steps):
    prediction = model.predict(future_data)
    predicted_prices.append(prediction)
    # Rolls the first item of future_data to last along axis 1
    future_data = np.roll(future_data, -1, axis=1)
    # Predicted value is used for forecasting other values
    # For this, `prediction`` is used to replace the rolled item at the last of future_data
    future_data[0, -1] = prediction

# Inverse transform the predicted prices to original scale
# predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).flatten()
predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Shift future predictions for plotting
futurePredictPlot = np.empty_like(np.concatenate((df, predicted_prices)))
# Inserts nan values to empty df like dataframe
futurePredictPlot[:, :] = np.nan
futurePredictPlot[ len(df) : len(df) + future_steps, :] = predicted_prices

plt.plot(scaler.inverse_transform(df), label="Actual Prices", linestyle='dashed')
plt.plot(trainPredictPlot, label="Train Predicted Prices")
plt.plot(testPredictPlot, label="Test Predicted Prices")
plt.plot(futurePredictPlot, label="Future Predicted Prices")
plt.xlabel("Days")
plt.ylabel("Close")
plt.legend()
# plt.show()

# exit()

# Hyperparameter tuning with grid search
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
from scikeras.wrappers import KerasClassifier
from keras.src.callbacks import early_stopping

param_grid = {
    # 'neurons': [16, 32, 64, 80, 96, 112, 128], 
    # 'dropout_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6],
    'optimizer': ['SGD', 'RMSprop', 'Adam'],
    'loss': ['mean_squared_error', 'mean_absolute_error', 'root_mean_squared_error', 'huber'],
    'batch_size': [8, 16, 32, 48, 64, 80, 128],
    'epochs': [50, 100, 150, 250, 300, 400, 500],
    'callbacks': ['relu', 'tanh', 'sigmoid', 'linear', 'swish'],
    # 'second_layer': [0, 16, 32, 64, 80, 96, 112],  # 0 means no second LSTM layer
}

batch_size = [8, 16, 32, 48, 64, 96, 128],
# neurons = [16, 32, 64, 80, 96, 112, 128], 
# dropout_rate = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6],
optimizer = ['SGD', 'RMSprop', 'Adam'],
loss = ['mean_squared_error', 'mean_absolute_error', 'root_mean_squared_error', 'huber'],
epochs = [50, 100, 150, 250, 300, 400, 500],
callbacks = ['relu', 'tanh', 'sigmoid', 'linear', 'swish', 'softmax'],

# Create the LSTM model using KerasClassifier
from scikeras.wrappers import KerasClassifier
modelTuner = KerasClassifier(
    build_fn=create_stacked_lstm_model, 
    batch_size = batch_size,
    # neurons = neurons, 
    # dropout_rate = dropout_rate,
    optimizer = optimizer,
    loss = loss,
    epochs = epochs,
    callbacks = callbacks,
    verbose=1
    )

# regressor = KerasRegressor(
#     build_fn=create_stacked_lstm_model, 
#     batch_size = batch_size,
#     neurons = neurons, 
#     dropout_rate = dropout_rate,
#     optimizer = optimizer,
#     loss = loss,
#     epochs = epochs,
#     activation = activation,
#     verbose=1
# )

# param_grid = dict(
#     batch_size = batch_size,
#     neurons = neurons, 
#     dropout_rate = dropout_rate,
#     optimizer = optimizer,
#     loss = loss,
#     epochs = epochs,
#     activation = activation
# )

# early_stopper = early_stopping(monitor='loss', min_delta=0.01, patience=5)
# Apply grid search
# grid = GridSearchCV(estimator=modelTuner, param_grid=grid_params, n_jobs=1, cv=3)
grid_search = GridSearchCV(estimator=modelTuner, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
# grid_result = grid.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), callbacks=[early_stopper])
# grid_result = grid_search.fit(x=X_train, y=y_train, validation_data=(X_test, y_test))
# grid_result = grid_search.fit(X_train, y_train, sample_weight=None, validation_data=(X_test, y_test))
grid_result = grid_search.fit(X_train, y_train, validation_data=(X_test, y_test))
print(grid_result.best_params_)
