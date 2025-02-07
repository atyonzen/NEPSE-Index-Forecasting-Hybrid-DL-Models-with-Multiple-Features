# Import necessary libraries
import os
import sys
sys.path.append(os.path.abspath('') + os.path.sep + 'model_helpers')
sys.path.append(os.path.abspath('') + os.path.sep + 'model_builder')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from variable_config import LOOK_BACK as look_back, file_name, EPOCHS, future_steps, k_fold, UNSEEN
from data_wrangler import data_wrangler, split_into_datasets
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from model_builder import hb_tuner_lstm, hb_tuner_gru
# from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
 
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set the global random seed to control the entire process of randomization.
# Create determinism and model reproducibility
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

# This code make sure that each time you run your code, your neural network weights will be initialized equally.
np.random.seed(42)
tf.random.set_seed(42)
import random
random.seed(42)

# Call data_wrangler to create features and label
X, y, scaler, raw_data, prep_data, data, unseen_data = data_wrangler(file_name, look_back, unseen=UNSEEN)

# Hold out validation data
X_train, X_val, X_test, y_train, y_val, y_test = split_into_datasets(X, y, look_back, get_val_set=True)

# Early stoping callbacks for best epoch
early_stop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, mode='min')

def evaluate_model(tuner):
    # Extract model type from tuner object.
    model_type = tuner.hypermodel.layer.__module__.split('.')[-1]

    # Load saved model
    model = keras.models.load_model(f'hyper_model/best_model/best_{model_type}_model.keras')

    # Model summary
    print(model.summary())
    
    # Evaluate the hypermodel on the test data.
    # test_loss, test_mae, test_mape, test_r2 = model.evaluate(X_test, y_test)
    test_result = model.evaluate(X_test, y_test)
    print(f'Evaluation of {model_type.upper()} model:\n')
    print(f'Test Metrics Loss(MSE), MSE, MAE, MAPE, and R2: {[round(elem, 6) for elem in test_result]}')

    # https://forecastegy.com/posts/time-series-cross-validation-python/
    # So make sure your data is sorted before using this method.
    # This tool automates the expanding window method, that expands the training set while keeping constant the test set.
    # TimeSeriesSplit respects the temporal order of your data, ensuring that the ‘future’ data is not used to train your model.
    # TimeSeriesSplit is for k-fold time series validation for time series data
    # The data will be split into 5 consecutive folds, where each fold trains on 
    # a progressively larger portion of the dataset and tests on the subsequent time period.

    tscv = TimeSeriesSplit(n_splits=k_fold)

    # Initialize an array to store the results of each fold
    fold_results = {
        'mse': [],
        'mae': [],
        'mape': [],
        'r2_score': []
    }

    # Iterate over each split in TimeSeriesSplit
    for train_index, test_index in tscv.split(X):
        # Split data into training and testing sets
        X_train_k, X_test_k = X[train_index], X[test_index]
        y_train_k, y_test_k = y[train_index], y[test_index]

        # Train the model
        model.fit(X_train_k, y_train_k, epochs=EPOCHS, callbacks=[early_stop], verbose=0)

        # Evaluate the model on the test set
        metrics = model.evaluate(X_test_k.reshape(-1, look_back, 1), y_test_k)
        # print('Metrics: MSE, MAE, MAPE, and R2', [round(elem, 6) for elem in metrics])

        # Store the result
        fold_results['mse'].append(round(metrics[1], 6))
        fold_results['mae'].append(round(metrics[2], 6))
        fold_results['mape'].append(round(metrics[3], 6))
        fold_results['r2_score'].append(round(metrics[4], 6))

    print(f'{k_fold}-Fold results of {model_type.upper()} model: \n', fold_results)
    # Average performance across all folds
    average_mse = round(np.mean(fold_results['mse']), 6)
    average_mae = round(np.mean(fold_results['mae']), 6)
    average_mape = round(np.mean(fold_results['mape']), 6)
    average_r2_score = round(np.mean(fold_results['r2_score']), 6)
    print(f"Average Test MSE across all folds: {average_mse}")
    print(f"Average Test MAE across all folds: {average_mae}")
    print(f"Average Test MAPE across all folds: {average_mape}")
    print(f"Average Test R2_Score across all folds: {average_r2_score}")
    
    # Create Dataframes of time series validation results and average results of metrics
    df_ts_results = pd.DataFrame(fold_results)
    df_ts_average = pd.DataFrame({'average': [average_mse, average_mae, average_mape, average_r2_score]})
    # Concate the DataFrames with transposition of later and save to .csv file
    df_ts_average = df_ts_average.T
    df_ts_average.columns = ['mse', 'mae', 'mape', 'r2_score']
    df_ts_results = pd.concat([df_ts_results, df_ts_average], axis=0)
    df_ts_results.to_csv(f'hyper_model/best_model/{k_fold}_fold_ts_results_of_{model_type}_model.csv')

    # Max and min values
    max_mse = np.max(fold_results['mse'])
    min_mae = np.min(fold_results['mae'])
    max_mape = np.max(fold_results['mape'])
    min_r2_score = np.min(fold_results['r2_score'])

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, gridspec_kw={'wspace': 0.5, 'hspace': 0.4})
    ax1, ax2, ax3, ax4 = axes.flatten()
    fig.suptitle(f'{k_fold}-Fold Time Series Validation of {model_type.upper()}-Dense Model')
    x_range = range(1, len(fold_results['r2_score']) + 1)
    # Mean Squared Error
    ax1.plot(x_range, fold_results['mse'], marker='o', c='red', linestyle='--')
    # ax1.set_title('R2_Score on Each Fold')
    ax1.set_ylabel('MSE')

    # Mean Absolute Error
    ax2.plot(x_range, fold_results['mae'], marker='o', c='green', linestyle='--')
    # ax1.set_title('R2_Score on Each Fold')
    ax2.set_ylabel('MAE')

    # MAPE
    ax3.plot(x_range, fold_results['mape'], marker='o', c='blue', linestyle='--')
    # ax2.set_title('MAPE on Each Fold')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('MAPE')

    # R2 Score
    ax4.plot(x_range, fold_results['r2_score'], marker='o', c='gray', linestyle='--')
    # ax1.set_title('R2_Score on Each Fold')
    ax4.set_xlabel('Fold')
    ax4.set_ylabel('R2_Score')
    # MSE
    for i in range(len(fold_results['mse'])):
        # ax1.text(i+1, fold_results['mse'][i] + 0.01,  # Offset the y-position slightly
        #      f'({i+1}, {fold_results["mse"][i]:.2f})', fontsize=9, color='red', ha='center')
        if (fold_results['mse'][i] == max_mse):
            xytext_value = (0, 10)
        else:
            xytext_value = (0, -15)

        ax1.annotate(f'({fold_results["mse"][i]:.4f})', 
            xy=(i+1, fold_results['mse'][i]),      # Point being annotated
            xytext= xytext_value,                       # Offset the label slightly
            textcoords='offset points',                 # Use offset points for placement
            fontsize=9, color='red', ha='center')
    # MAE
    for i in range(len(fold_results['mae'])):
        if (fold_results['mae'][i] == min_mae):
            xytext_value = (0, -15)
        else:
            xytext_value = (0, 10)
        ax2.annotate(f'({fold_results["mae"][i]:.4f})', 
            xy=(i+1, fold_results['mae'][i]),     # Point being annotated
            xytext=xytext_value,                   # Offset the label slightly
            textcoords='offset points',            # Use offset points for placement
            fontsize=9, color='green', ha='center')
    # MAPE
    for i in range(len(fold_results['mape'])):
        if (fold_results['mape'][i] == max_mape):
            xytext_value = (0, -15)
        else:
            xytext_value = (0, 10)
        ax3.annotate(f'({fold_results["mape"][i]:.4f})', 
            xy=(i+1, fold_results['mape'][i]),     # Point being annotated
            xytext=xytext_value,                   # Offset the label slightly
            textcoords='offset points',            # Use offset points for placement
            fontsize=9, color='green', ha='center')      
    # R2 Score
    for i in range(len(fold_results['r2_score'])):
        # ax1.text(i+1, fold_results['r2_score'][i] + 0.01,  # Offset the y-position slightly
        #      f'({i+1}, {fold_results["r2_score"][i]:.2f})', fontsize=9, color='red', ha='center')
        if (fold_results['r2_score'][i] == min_r2_score):
            xytext_value = (0, 10)
        else:
            xytext_value = (0, -15)

        ax4.annotate(f'({fold_results["r2_score"][i]:.4f})', 
            xy=(i+1, fold_results['r2_score'][i]),      # Point being annotated
            xytext= xytext_value,                       # Offset the label slightly
            textcoords='offset points',                 # Use offset points for placement
            fontsize=9, color='red', ha='center')

    plt.show()

def plot_model_prediction(tuner):

    # This code make sure that each time you run your code, your neural network weights will be initialized equally.
    # from numpy.random import seed
    # seed(42)
    # from tensorflow import random
    # random.set_seed(42)
    
    # Extract model type from tuner object.
    model_type = tuner.hypermodel.layer.__module__.split('.')[-1]

    # model = keras.models.load_model('hyper_model/best_model/best_model.keras')
    model = keras.models.load_model(f'hyper_model/best_model/best_{model_type}_model.keras')

    # Create dataframe from 1D numy array
    df = pd.DataFrame(data, columns=['Close'])

    # Predict using the trained model
    # Predict for x_train, X_test
    print(X_train.shape, X_val.shape, X_test.shape)
    train_predict = model.predict(X_train)
    val_predict = model.predict(X_val)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    val_predict = scaler.inverse_transform(val_predict)
    test_predict = scaler.inverse_transform(test_predict)

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
    future_data = data[-look_back:].reshape(-1, look_back, 1)
    predicted_prices = []
    for _ in range(future_steps):
        prediction = model.predict(future_data)
        predicted_prices.append(prediction)
        # Rolls the first item of future_data to last along axis 1
        future_data = np.roll(future_data, -1, axis=1)
        # Predicted value is used for forecasting other values
        # For this, `prediction` is used to replace the rolled item at the last of future_data
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
    plt.title(f'NEPSE Index Prediction with {model_type.upper()}-Dense Model')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()

def test_plot_model_prediction(tuner, unseen_data=None):
   
    # This code make sure that each time you run your code, your neural network weights will be initialized equally.
    # from numpy.random import seed
    # seed(42)
    # from tensorflow import random
    # random.set_seed(42)
    
    # Extract model type from tuner object.
    model_type = tuner.hypermodel.layer.__module__.split('.')[-1]
    # Load the best model.
    model = keras.models.load_model(f'hyper_model/best_model/best_{model_type}_model.keras')

    # Create dataframe from 1D numy array
    df = pd.DataFrame(y_test, columns=['Close'])

    # Predict using the trained model
    # print(X_train.shape, X_val.shape, X_test.shape, y_test.shape)
    test_predict = model.predict(X_test)

    # test_predict = scaler.inverse_transform(test_predict)
    # Create a placeholder array with the same shape as the scaler's expected input
    placeholder = np.zeros((test_predict.shape[0], scaler.min_.shape[0]))
    # Replace the relevant column with test_predict values
    placeholder[:, 0] = test_predict[:, 0]

    # Perform inverse_transform
    test_predict_rescaled = scaler.inverse_transform(placeholder)
    # Extract the relevant column after scaling
    test_predict = test_predict_rescaled[:, 0]
    
    # Plot the predicted prices
    # Shift train predictions for plotting
    testPredictPlot = np.empty_like(df)
    # Inserts nan values to empty df like dataframe
    testPredictPlot[:, :] = np.nan
    testPredictPlot[:, :] = test_predict.reshape(-1,1)
    
    # Original transformation
    # Replace the relevant column with original close values
    placeholder[:, 0] = y_test
    original_features = scaler.inverse_transform(placeholder)
    original_close = original_features[:, 0]

    plt.plot(original_close.reshape(-1, 1), label='Actual Index', linestyle='dashed')
    plt.plot(testPredictPlot, label='Predicted Index')   
    
    # Walk Forward Validation to future prices.
    # If unseen is True, then only plot the unseen prices and predicted prices
    if unseen_data is not None:
        unseen_data = scaler.inverse_transform(np.array(unseen_data))
        unseen_data = unseen_data[:, 3].reshape(-1, 1)
        # Shift unseen_data for plotting
        unseenDataPlot = np.empty_like(np.concatenate((df, unseen_data)))
        # Inserts nan values to empty df like dataframe.
        unseenDataPlot[:, :] = np.nan
        unseenDataPlot[ len(df) : len(df) + future_steps, :] = unseen_data
        plt.plot(unseenDataPlot, marker='o', label=f"{future_steps} Day's Unseen Prices", linestyle='dashed')

    # Split whole data into training and test sets
    # test set contains single observation and remaining in the training set.
    X_wfv_train, X_wfv_test = train_test_split(X, test_size=1, random_state=42, shuffle=False)
    y_wfv_train, _ = train_test_split(y, test_size=1, random_state=42, shuffle=False)
    
    # Predict future prices with walk forward validation.
    model.fit(x=X_wfv_train, y=y_wfv_train, epochs=EPOCHS, callbacks=[early_stop])
    y_pred = model.predict(X_wfv_test.reshape(-1, look_back, 1))

    # Inverse transform the predicted prices to original scale
    predicted_price = placeholder[-future_steps:, :]
    predicted_price[:, 0] = np.array(y_pred)
    predicted_price = scaler.inverse_transform(predicted_price)
    predicted_price = predicted_price[:, 0].reshape(-1, 1)
    # Shift future predictions for plotting
    futurePredictPlot = np.empty_like(np.concatenate((df, predicted_price)))
    # Inserts nan values to empty df like dataframe
    futurePredictPlot[:, :] = np.nan
    futurePredictPlot[ len(df) : len(df) + future_steps, :] = predicted_price
    plt.plot(futurePredictPlot, marker='o', label=f"{future_steps} Day's Future Prices")

    plt.xlabel('Days')
    plt.ylabel('Close')
    plt.title(f'NEPSE Index Prediction with {model_type.upper()}-Dense Model')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()

def plot_training_history():
        
    # Load the training history of the model.
    history_lstm = pd.read_csv('hyper_model/best_model/best_lstm_model_history.csv')
    history_gru = pd.read_csv('hyper_model/best_model/best_gru_model_history.csv')

    # Plot Learning Curves
    # Training Learning Curves
    
    fig, (ax1, ax2) = plt.subplots(2, sharex=False)
    fig.suptitle(f'Training Learning Curves of Hybrid Models')
    x_range_lstm = range(1, len(history_lstm['epoch']) + 1)
    x_range_gru = range(1, len(history_gru['epoch']) + 1)
    
    # LSTM-Dense Model Loss
    ax1.plot(x_range_lstm, history_lstm['loss'], label='LSTM-Dense Model', marker='o', c='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('loss (MSE)')
    ax1.legend()

    # GRU-Dense Model Loss
    ax2.plot(x_range_gru, history_gru['loss'], label='GRU-Dense Model', marker='o', c='green', linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (MSE)')
    ax2.legend()

    plt.show()

# plot_model_prediction(hb_tuner_lstm)
# plot_training_history()
# evaluate_model(hb_tuner_lstm)
test_plot_model_prediction(hb_tuner_lstm, unseen_data)
# evaluate_model(hb_tuner_gru)
test_plot_model_prediction(hb_tuner_gru, unseen_data)
