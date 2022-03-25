# imports
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from pandas_datareader.data import DataReader
import yfinance as yf

print('Imports Loaded')


# the stock Predictor Class

class Stock:

    # constructor
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = yf.download(self.ticker, period="max")

        # all variables to be created after preprocessing
        self.data = None
        self.dataset = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.scaler = None
        self.training_data_len = None

        # for the model
        self.model = None
        self.predictions = None
        self.valid = None


    # preprocessing function to define all of the variables
    def preprocess(self):
        self.data = self.df.filter(['Close'])
        self.dataset = self.data.values

        self.training_data_len = int(np.ceil( len(self.dataset) * .95 ))
        self.scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = self.scaler.fit_transform(self.dataset)

        train_data = scaled_data[0:int(self.training_data_len), :]
        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
                
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        self.x_train = x_train
        self.y_train = y_train

        test_data = scaled_data[self.training_data_len - 60: , :]

        # Create the data sets x_test and y_test
        x_test = []
        y_test = self.dataset[self.training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
            
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

        self.x_test = x_test
        self.y_test = y_test


    def set_model(
        self, 
        optimizer = tf.keras.optimizers.Adam(), 
        loss = tf.keras.losses.MeanSquaredError(), 
        model=None
    ):
        if model:
            self.model = model
        else:
            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(128, return_sequences=True, input_shape= (self.x_train.shape[1], 1)))
            model.add(LSTM(64, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))

            self.model = model

        self.model.compile(optimizer=optimizer, loss=loss)

        
    def train_model(self, epochs=1):
        self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=1,
            epochs=epochs
        )


    def predict(self):
        self.predictions = self.model.predict(self.x_test)
        self.predictions = self.scaler.inverse_transform(self.predictions)
        rmse = np.sqrt(np.mean(((self.predictions - self.y_test) ** 2)))
        return self.predictions, rmse


    # get access to all of the ticker data
    def get_all_data(self):
        return self.df


    # plot Close Price History
    def plot_cp_hist(self):
        plt.figure(figsize=(16,6))
        plt.title('Close Price History')
        plt.plot(self.df['Close'])
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.show()


    # plot the prediction
    def plot_pred(self):
        train = self.data[:self.training_data_len]
        self.valid = self.data[self.training_data_len:]
        self.valid['Predictions'] = self.predictions
        # Visualize the data
        plt.figure(figsize=(16,6))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(self.valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower left')
        plt.show()
