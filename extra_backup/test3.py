# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
# from keras.src.models import Sequential
# from keras.src.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras_tuner as kt
from tensorflow import keras

# reac csv file
df = pd.read_csv('nepsealpha2.csv')
# remove columns which our neural network will not use
df = df.drop(['Symbol', 'Date', 'Open', 'High', 'Low', 'Percent Change', 'Volume'], axis=1)

# Normalize the Close price column
scaler = MinMaxScaler()
df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

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

def create_train_test_split(data, look_back = 7, train_test_split = 0.8, val_split = 0.1):

    # Prepare the training data
    X, y = prepare_data(data, look_back)
    
    # Split the data into training and testing sets
    # It is assumed to create validation set from training set
    training_size = int(len(X) * train_test_split)
    validation_size = int(len(X) * val_split)
    X_train, y_train = X[:training_size - validation_size], y[:training_size - validation_size]
    X_val, y_val = X_train[-validation_size:], y_train[-validation_size:]
    X_test, y_test = X[training_size:], y[training_size:]
    # print(X_train.shape)
    # print(X_val.shape)
    # print(X_test.shape)
    # (155, 7)
    # (22, 7)
    # (45, 7)

    # Reshape the input data for LSTM
    # Reshape input data into [samples, timesteps, features]
    X_train = X_train.reshape(-1, look_back, 1)
    X_val = X_val.reshape(-1, look_back, 1)
    X_test = X_test.reshape(-1, look_back, 1)

    return X_train, X_val, X_test, y_train, y_val, y_test

# Define the LSTM model
def create_stacked_lstm_model(hp):
    model = keras.Sequential()

    activation = hp.Choice('activation', ['relu', 'tanh', 'sigmoid', 'linear', 'swish'])
    # 'loss': ['mean_squared_error', 'mean_absolute_error', 'root_mean_squared_error', 'huber'],
    # 'epochs': [50, 100, 150, 250, 300, 400, 500],
    learning_rate = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')

    look_back = 7

    num_layers = hp.Int('num_layers', min_value=1, max_value=20)
    epochs = hp.Int('epochs', min_value=10, max_value=200, step=5)

    for i in range(num_layers):
        model.add(
            keras.layers.LSTM(
                units=hp.Int(f'units_{i}', min_value=1, max_value=200, step=8),
                activation=activation, 
                return_sequences=True if i < num_layers-1 else False,
                input_shape=(look_back, 1)
            )
        )
        
        if hp.Boolean('dropout'):
            model.add(
                keras.layers.Dropout(
                    rate=hp.Float(f'dropout_{i}', min_value=1e-4, max_value=0.6, sampling='log')
                )
            )
    
    model.add(keras.layers.Dense(units=hp.Int('units', min_value=1, max_value=200, step=8)))
    model.add(keras.layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=['mean_absolute_error']
    )

    return model

# Create train and test data
look_back = 7  # Number of time steps to consider for prediction
data = df['Close'].values
X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(data=data)

# Keras-Tuner
tuner = kt.Hyperband(
    create_stacked_lstm_model,
    objective='val_loss',
    max_epochs=5,
    factor=3,
    overwrite=True,
    directory='dir',
    project_name='model-tuning'
)

# stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
# tuner.search(X_train, y_train, epochs=5, validation_data=(X_val, y_val), callbacks=[stop_early])
tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Get the top 1 model.
best_model = tuner.get_best_models(num_models=1)[0]
best_model.fit(X_train, y_train, batch_size=8, epochs=100, initial_epoch=6, validation_data=(X_val, y_val), verbose=1)
best_model.summary()
print(best_hps.values)

# exit()

# Predict using the trained model
# Predict for x_train, X_test
train_predict = best_model.predict(X_train)
val_predict = best_model.predict(X_val)
test_predict = best_model.predict(X_test)
# print(train_predict.shape)
# print(train_predict)
train_predict = scaler.inverse_transform(train_predict)
val_predict = scaler.inverse_transform(val_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate RMSE performance matrics
# Train data RMSE
# import math
# from sklearn.metrics import mean_squared_error
# math.sqrt(mean_squared_error(y_train, train_predict))

# Test data RMSE
# math.sqrt(mean_squared_error(y_test, test_predict))


# Plot the predicted prices
# Shift train predictions for plotting
trainPredictPlot = np.empty_like(df)
# Inserts nan values to empty df like dataframe
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back: len(train_predict) + look_back, :] = train_predict

valPredictPlot = np.empty_like(df)
# Inserts nan values to empty df like dataframe
valPredictPlot[:, :] = np.nan
valPredictPlot[len(train_predict) + look_back: len(train_predict) + len(val_predict) + look_back, :] = val_predict

# Shift test predictions for plotting
testPredictPlot = np.empty_like(df)
# Inserts nan values to empty df like dataframe
testPredictPlot[:, :] = np.nan
testPredictPlot[ len(train_predict) + len(val_predict) + look_back : len(df), :] = test_predict

# Predict for coming future
future_steps = 164  # Number of days to predict
future_data = data[-look_back:].reshape(-1, look_back, 1)
predicted_prices = []
for i in range(future_steps):
    prediction = best_model.predict(future_data)
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

plt.plot(scaler.inverse_transform(df), label='Actual Prices', linestyle='dashed')
plt.plot(trainPredictPlot, label='Train Predicted Prices')
plt.plot(valPredictPlot, label='Validation Predicted Prices')
plt.plot(testPredictPlot, label='Test Predicted Prices')
plt.plot(futurePredictPlot, label='Future Predicted Prices')
plt.xlabel('Days')
plt.ylabel('Close')
plt.legend()
plt.show()